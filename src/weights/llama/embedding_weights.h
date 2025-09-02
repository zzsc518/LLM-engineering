#pragma once
#include "src/weights/base_weights.h"
// 不需要修改，基类已经可以表达了
template<typename T>
struct EmbeddingWeight: public BaseWeight<T> {
};
