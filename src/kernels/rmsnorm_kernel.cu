#include <stdio.h>
#include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/rmsnorm_kernel.h"
//bugs1: 2nd warpreducesum returns 0, because blockDim.x < 32, blockDim.x / 32=0
//bugs2: output buffer valuse is the same as ones before call, thats because we didn't successfully write into the output address
//bugs3: output buffer's 1st 32 values are right, the latter is wrong, because when we use vec, the ele nums of a row is hiddenunits/vecsize, we should note the row stride to move the ptr carefully
//bugs4: remeber add __syncthreads() in fp32/fp16 kernel, or we cant get the right res, ep, here we didnt add it, we get some res equal to 0 

// 每个线程都拿到了warp内的总和
template<typename T>
__device__ T warpReduceSum(T val){
    for(int i = 32 / 2; i > 0; i >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val; // 32 threads return val, but only 0th thread is sum val
}

//note:!!!when blocksize < 32, use blockDim.x/32 to get warp nums is wrong, we should instead ceil it
template<typename T>
__device__ T blockReduceSum(T val){
    int tid = threadIdx.x;
    int wid = tid / 32;
    int laneid = tid % 32;
    int warpnum = (blockDim.x + 31) / 32;

    // 保险起见直接变成64，但其实没必要，写32就行
    static __shared__ T warpsum[64];
    val = warpReduceSum<T>(val);
    if(laneid == 0){
        warpsum[wid] = val;
    }
    __syncthreads();
    
    // 因为warpshuffle是针对于寄存器的，所以先要取出来
    // 让前warpnum个线程拿到数，再做warpshuffle，最后前warpnum个线程都拿到了这个block的总和
    T sum = tid < warpnum ? warpsum[tid] : (T)0;
    sum = warpReduceSum<T>(sum); //though 0th own the sum, but dont need to shfl sync
    return sum;
}


// 1.this kernel is used at the begin of every decoder layer and the end of 32 decoder layers
// 2.I allocate threads number by assuming head size can be divided by 4 and 2
// q_hidden_units的目的是区分不同的头，q表示q的头
// hidden_units = headnum * head_size
template <typename T>
__global__ void RMSNorm(T* decoder_out, // [num tokens, q_hidden_units]
                        T* decoder_residual,
                        T* scale, //[q_hidden_units], RMSNorm weights
                        float eps, //RMSNorm eps
                        int num_tokens, 
                        int hidden_units){
  int vec_size = Vec<T>::size;
  // 编译器无法确定 Vec<T>::Type 到底是一个类型还是一个变量/值
  using Vec_t = typename Vec<T>::Type;
  float thread_sum = 0.0f;

  // 定位到每个block处理的位置
  // float4数组
  Vec_t* dout = reinterpret_cast<Vec_t*>(decoder_out + blockIdx.x * hidden_units);
  Vec_t* rsd;
  rsd = reinterpret_cast<Vec_t*>(decoder_residual + blockIdx.x * hidden_units);
  
  // 取float4的数据，处理任意数量的数据
  for (int idx = threadIdx.x; idx < hidden_units / vec_size; idx += blockDim.x) {
    Vec_t vec = dout[idx];
    rsd[idx] = vec;
    thread_sum += vec.x * vec.x;
    thread_sum += vec.y * vec.y;
    thread_sum += vec.z * vec.z;
    thread_sum += vec.w * vec.w;
  }
  thread_sum = blockReduceSum<float>(thread_sum);

  // 设成shared可以减少寄存器的使用，并且不会有bank conflict
  __shared__ float inv_mean;
  // 一个写
  
  if (threadIdx.x == 0) {
    inv_mean = rsqrtf((float)thread_sum / hidden_units + eps);
  }
  __syncthreads();
  
  // scale也可以向量化读取
  Vec_t* s = reinterpret_cast<Vec_t*>(scale);
  for (int idx = threadIdx.x; idx < hidden_units / vec_size; idx += blockDim.x) {
    Vec_t out = dout[idx];// note the offset should divide vec size
    
    dout[idx].x = out.x * inv_mean * s[idx].x;
    dout[idx].y = out.y * inv_mean * s[idx].y;
    dout[idx].z = out.z * inv_mean * s[idx].z;
    dout[idx].w = out.w * inv_mean * s[idx].w;
  }
}

template <>
__global__ void RMSNorm(half* decoder_out, // [num tokens, q_hidden_units]
                        half* decoder_residual,
                        half* scale, //[q_hidden_units], RMSNorm weights
                        float eps, //RMSNorm eps
                        int num_tokens, 
                        int hidden_units){
    int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    Vec_t* s; 
    Vec_t* dout = reinterpret_cast<Vec_t*>(decoder_out + batch_id * hidden_units);
    Vec_t* rsd;

    if (decoder_residual != nullptr) {
        rsd = reinterpret_cast<Vec_t*>(decoder_residual + batch_id * hidden_units);
    }
    // 累加这里还是用fp32
    float thread_accm = 0.0f;
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        Vec_t out = dout[i];// note the offset should divide vec size
        if (decoder_residual != nullptr) {
            rsd[i] = out;
        }
        thread_accm += __half2float(out.x) * __half2float(out.x);
        thread_accm += __half2float(out.y) * __half2float(out.y);
    } //x^2
    
    // mean(x^2)
    float blocksum = blockReduceSum<float>(thread_accm);
    __shared__ float inv_fenmu;
    if(tid == 0){
        inv_fenmu = rsqrtf(float(blocksum / hidden_units) + eps);
    }
    __syncthreads();
    // rmsnorm
    s = reinterpret_cast<Vec_t*>(scale);
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        Vec_t dout_h2 =dout[i];
        // 先转成fp32，两个fp32相乘后再转换为fp16再相乘
        dout[i].x = s[i].x * __float2half(__half2float(dout_h2.x) * inv_fenmu);
        dout[i].y = s[i].y * __float2half(__half2float(dout_h2.y) * inv_fenmu);
    }    
}

// [token num, hidden_size]  ===> [token num, hidden_size] 
// eps: RMSNorm计算中的epsilon值，用于数值稳定性
// attn_norm_weight: RMSNorm层的权重参数，只读不写
// decoder_out: 既是输入也是输出
// decoder_residual: 输出参数，存储decoder输出的残差连接数据（拷贝自decoder_out）
template<typename T>
void launchRMSNorm( TensorWrapper<T>* decoder_out, // [num tokens, hidden_units]
                    TensorWrapper<T>* decoder_residual,
                    LayerNormWeight<T>& attn_norm_weight, //RMSNorm weights
                    float eps, //RMSNorm eps
                    bool is_last // for print last rmsnorm output to debug
                    )
{
    int num_tokens = decoder_out->shape[0];
    int hidden_units = decoder_out->shape[1];
    int vec_size = Vec<T>::size;
    int num_threads = hidden_units / 4; //vec size // assume head size can be divided by 4 and 2
    T* rsd = decoder_residual->data;
    dim3 grid(num_tokens);
    dim3 block(num_threads);
    RMSNorm<T><<<grid, block>>>(decoder_out->data,
                            rsd,
                            attn_norm_weight.gamma,
                            eps,
                            num_tokens,
                            hidden_units);
#ifdef PRINT_DATA
    printf("rmsnorm kernel top2 result:\n");
    print_data<<<1, 1>>>(decoder_out->data);
#else
#endif
}

// 模板实例化
// 模板函数只有在被调用时才会生成实际代码
// 如果只在当前文件中定义模板，而在其他文件中使用，链接器可能找不到实现
// 通过显式实例化，确保编译器为这些常用类型（float和half）生成代码
template void launchRMSNorm( TensorWrapper<float>* decoder_out, // [num tokens, hidden_units]
                    TensorWrapper<float>* decoder_residual,
                    LayerNormWeight<float>& attn_norm_weight, //RMSNorm weights
                    float eps, //RMSNorm eps
                    bool is_last
                    );
template void launchRMSNorm( TensorWrapper<half>* decoder_out, // [num tokens, hidden_units]
                    TensorWrapper<half>* decoder_residual,
                    LayerNormWeight<half>& attn_norm_weight, //RMSNorm weights
                    float eps, //RMSNorm eps
                    bool is_last
                    );
