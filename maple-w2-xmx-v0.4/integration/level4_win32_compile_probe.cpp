// SPDX-License-Identifier: MIT
#include "level4_win32_memory.hpp"
// -fsycl -Iinclude -Iintegration -I<LevelZeroSDK/include> -c this file
// Compiling this file is NOT an import roundtrip or queue ownership test.
static_assert(sizeof(ze_device_handle_t)==sizeof(void*),"native handle type");
