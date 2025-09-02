#pragma once
#include <vector>
#include <cstdint>
#include <cuda_fp16.h>

// 枚举类
enum class WeightType
{
    FP32_W,
    FP16_W,
    INT8_W,
    UNSUPPORTED_W
};

// 返回T对应的WeightType
// std::is_same<T, float>::value是比较T和float是不是一样类型，然后返回false或者true
template<typename T>
inline WeightType getWeightType()
{
    if (std::is_same<T, float>::value || std::is_same<T, const float>::value) {
        // 必须使用：：来访问枚举类里面的成员
        return WeightType::FP32_W;
    }
    else if (std::is_same<T, half>::value || std::is_same<T, const half>::value) {
        return WeightType::FP16_W;
    }
    else if (std::is_same<T, int8_t>::value || std::is_same<T, const int8_t>::value) {
        return WeightType::INT8_W;
    }
    else {
        return WeightType::UNSUPPORTED_W;
    }
}
template<typename T>
struct BaseWeight {

    std::vector<int> shape;
    T*   data;
    // 类型
    WeightType type;
    // bias是非空则data为空
    T*   bias;
};

/*
 * 枚举类和枚举类型
 * 枚举类（enum class）是C++11引入的强类型枚举，相比传统枚举具有更好的类型安全性、作用域控制和避免命名冲突的优点
 * 枚举类型相当于是定义了宏，枚举类相当于是加了命名空间
 * 1.作用域区别
 *   - 传统枚举：枚举成员在枚举类型外部可直接访问，容易造成命名冲突
            enum Device { CPU, GPU };
            // 直接访问枚举值   
            Device device = CPU;  // 正确
            Device device2 = GPU; // 正确
            // 命名冲突
            enum Color { RED, GREEN, BLUE };
            enum Fruit { APPLE, ORANGE, RED };  // 错误！RED重复定义
 *   - 枚举类：可以避免命名冲突，枚举成员在枚举类内部定义，外部访问时需要使用作用域运算符（::）限定
            enum class WeightType { FP32_W, FP16_W, INT8_W, UNSUPPORTED_W };
            // 必须使用作用域运算符访问
            WeightType type = WeightType::FP32_W;  // 正确
            WeightType type2 = FP32_W;             // 错误！必须使用作用域
 *  2.隐式转换为整数
     - 传统枚举 - 允许隐式转换为整数
            enum Device { CPU, GPU };
            Device device = CPU;
            int value = device;  // 隐式转换为 0
     - 枚举类   - 不允许隐式转换为整数
            enum class WeightType { FP32_W, FP16_W };
            WeightType type = WeightType::FP32_W;
            int value = type;  // 错误！不能隐式转换
            int value = static_cast<int>(type); // 必须显式转换
*/