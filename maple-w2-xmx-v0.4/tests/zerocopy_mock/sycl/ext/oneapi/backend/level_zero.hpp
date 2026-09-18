#pragma once
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
namespace sycl {
template<backend B>ze_context_handle_t get_native(const context&){return reinterpret_cast<void*>(0x101);}
template<backend B>ze_device_handle_t get_native(const device&){return reinterpret_cast<void*>(0x102);}
}
