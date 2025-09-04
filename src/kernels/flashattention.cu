#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 最关键的点是什么? Q*K^T , scale得到S, 顺便算个局部最大值 -> 局部softmax得到P,  顺便算个局部l -> o += P*V， 这里两边都需要用l和m更新 -> 最后把l和m写回

// 目前是不支持N不能整除Bc和Br的?
// 且目前是仅支持Bc==Br的
// 开了Bc个线程，负责[Br, d] * [d, Bc] = [Br, Bc],  scale, softmax, [Br, Bc] *[Bc, d]
// 在QK相乘的时候，一个线程负责一行的S计算，一个线程拿到一行的之前的l和m，一个线程将Q的一行去和K一个矩阵去相乘
// S是QK相乘之后，再算完scale的，所以才叫S
// 算l和m的主要原因是S在shared mem里并不是一个完整的矩阵，只是矩阵的一个分块，行和列都分块了
// o的shape是和QKV一样的，所以不用算o的offset，shape是什么来着
// 老的l也是要算的，也挺重要的
// 算出来的中间o确实可以直接放到寄存器里面去，不用放到smem里面，然后再和老o加和变成新o，放到寄存器里也不会用很多寄存器
// 因为o不会被复用，而KVS会，Q也不会被复用
// 计算和访存的逻辑可以分开，这里loadKV是让一个线程去load一行
// 因为loadKV是一个线程去load一行，而在计算中一个线程是要用到K的一行，和V的一列，这V是需要用到别的线程load的，所以内层循环需要__syncthreads()，是有冲突所以才会需要__syncthreads()
// 而loadQ就是一个线程去load Q的一行
__global__ void FlashAttention(const float* Q, const float* K, const float* V, const int N, const int d, const int Tc, const int Tr,
    const int Bc, const int Br, const float softmax_scale, float* l, float* m, float* o){
        int bx = blockIdx.x; int by = blockIdx.y;
        int tx = threadIdx.x;
        
        extern __shared__ float smem[];

        // 让block指向数据，计算offset
        // 这个计算需要注意，按照grid的角度，是一个nh * N的矩阵，是列主序的
        int qkv_offset = (bx * gridDim.y + by) * (N * d);
        int lm_offset = (bx * gridDim.y + by) * N;

        // 让QKVS指向shared mem，计算offset
        int tile_size = Bc * d;
        float* Qi = smem;
        float* Kj = smem + tile_size;
        float* Vi = smem + tile_size * 2;
        float* S = smem + tile_size * 3;
        
        for(int j = 0; j < Tc; j++){
            
            // load KV
            for(int x = 0; x < d; x++){
                // 注意tile_size * j 
                // 这里Kj还是行主序的，并没有转置
                Kj[tx * d + x] = K[qkv_offset + tile_size * j + tx * d + x];
                Vj[tx * d + x] = V[qkv_offset + tile_size * j + tx * d + x];
            }

            // 为了后面能取到正确的Kj, Vj
            __syncthreads();

            // load Q, 内层循环
            for(int i = 0; i < Tr; i++){
                // load Qi to smem, load l,m to registers
                // 一个线程去load一个Q的一行
                for(int x = 0; x < d; x++){
                    Qi[tx * d + x] = Q[qkv_offset + tile_size * i + tx * d +x]; 
                }
                
                // 为什么是之前的，因为可能是上个外层循环的
                float row_l_prev = l[lm_offset + i * Br + tx];
                float row_m_prev = m[lm_offset + i * Br + tx];
                
                // 不需要，因为是一个线程去load Q，load完之后和整个K专职相乘
                // 这里不需要__syncthreads()吗 
                
                // 计算Q*K^T
                // 这里可以顺便计算出一个局部最大值
                // 并且把scale也做了，这里是不能做softmax的，因为局部最大值还没算出来
                float row_max = -INFINITY;
                for(int y = 0; y < Bc; y++){
                    float sum = 0;
                    for(int x = 0; x < d; x++){
                        // 这里Kj是跨步很多的，没有转置
                        sum += Qi[tx * d + x] * Kj[tx + y*d]
                    }
                    // 算scale
                    sum *= softmax_scale;

                    // 经过for循环之后，每次可以得到S的一行的一个值
                    S[tx * Bc +y] = sum;
                    
                    // 算局部最大值
                    row_max = max(row_max, sum);
                }
                
                // 此时已经拿到了局部最大值，开始做局部softmax, 并且算局部的加和
                // 还是一样，一个thread去计算S的一行
                // 这里得到了P波浪
                float row_l = 0.0f;
                float tmp = 0.0f;
                for(int x = 0; x < Bc; x++){
                    // __expf(x)更快
                    tmp = S[tx * Bc + x];
                    S[tx * Bc + x] = __expf(tmp - row_max); 
                    row_l += tmp;
                }
                
                // 更新m，得到当下的最大的m
                float new_row_m = max(row_max, row_m_prev);
                // 更新l，老l和刚计算出来的l都要更新，因为都不是当下最大的m
                float new_row_l = row_l_prev * __expf(row_m_prev - new_row_m) + row_l * __expf(row_max - new_row_m);
                // 因为之前的P已经融入o里面去了， 所以不用更新P

                // o += 乘法, 所以o是需要先处理下
                // 处理o

                // P*V，还是一样，一个线程算一行, o分块的shape是[Br, d]，所以还是一个线程算一行
                // o的offset和QKV一样
                float tmp_2 = 0.0f;
                for(int x = 0; x < d; x++){
                    float pv = 0.0f;
                    for(int y = 0; y < Bc; y++){
                        pv = S[tx * Bc + y] * Vi[x + y * d];
                    }

                    tmp_2 = o[qkv_offset + tile_size * i + tx * d  + x];
                    // 处理o，先乘以prev_l，再乘以m，然后再除以现在的l
                    tmp_2 = tmp_2 * row_l_prev * __expf(row_m_prev - new_row_m) / new_row_l;                 
                        
                    // pv不能直接加入，需要把p变成正确的p
                    pv = pv * __expf(row_max - new_row_m);
                    o[qkv_offset + tile_size * i + tx * d  + x] = tmp_2 + pv;
                    // 也是同样的原因，不需要__syncthreads()
                }

                // 最后将l和r更新了
                l[lm_offset + i * Br + tx] = new_row_l;
                m[lm_offset + i * Br + tx] = new_row_m;
            }
            // 看最上面的分析，这里是需要__syncthreads()的
            __syncthreads();
        }
        
}


// qk的buffer应该去掉，新建一个l,m的buffer，在context_attention.cpp里面的allocator搞，allocated一份，然后把B，nh，N，d都拿出来
// 然后加一点check的信息，必须要能整除
// qkv就是o，还需要加上l和m，flashAttn_l和flashAttn_m

// max_q_len和max_k_len?
// 不能n是不是能整除Bc和Br，n的shape是多少不知道
template <typename T>
void launchFlashAttention(TensorWrapper<T> *q, TensorWrapper<T> *k, TensorWrapper<T> *v, TensorWrapper<T> *l, 
        TensorWrapper<T> *m, TensorWrapper<T> qkv, TensorWrapper<T> *mask,float scale)
{
    const int Bc = 32; const int Br = 32;
    const int B = q->shape[0]; const int nh = q->shape[1];
    const int N = q->shape[2]; const int d = q->shape[3];

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);

    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));

    int max_sram_size;
    // TODO：需要check下，不能超过max_sram_size
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);
    
    dim3 grid(B, nh);
    dim3 block(Bc); // 这里先写死32，后面可以改成动态的
    FlashAttention<<<grid, block, sram_size>>>(q, k, v, N, d, Tc, Tr, Bc, Br, scale, l, m, qkv);
#ifdef PRINT_DATA
    printf("FlashAttention top2 result:\n");
    print_data<<<1, 1>>>(qkv->data);
#else
#endif
}