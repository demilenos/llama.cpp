#pragma once
// Minimal metadata double, not an installed GGML header.
#include <cstdint>
#include <cstddef>
enum ggml_type {GGML_TYPE_F32,GGML_TYPE_TQ2_0,GGML_TYPE_F16};
enum ggml_op {GGML_OP_NONE,GGML_OP_MUL_MAT};
struct ggml_tensor {ggml_type type=GGML_TYPE_F32;ggml_op op=GGML_OP_NONE;int64_t ne[4]{1,1,1,1};size_t nb[4]{};ggml_tensor*src[10]{};size_t view_offs=0;void*data=nullptr;};
