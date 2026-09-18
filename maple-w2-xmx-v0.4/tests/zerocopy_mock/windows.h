#pragma once
// Host-only Win32 API double; never part of a production target.
#include <cstdint>
using HANDLE=void*;using DWORD=uint32_t;using BOOL=int;using LPCWSTR=const wchar_t*;
#define INVALID_HANDLE_VALUE reinterpret_cast<HANDLE>(intptr_t(-1))
extern "C" BOOL CloseHandle(HANDLE);
