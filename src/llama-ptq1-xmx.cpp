#include "llama-ptq1-xmx.h"

#include "ggml-backend.h"
#include "ggml-vulkan/ggml-vulkan-hybrid.h"
#include "dg2_validation.hpp"
#include "level4_device_identity_win32.hpp"
#include "maple_ptq1a8.hpp"
#include "maple_w2a8.hpp"

#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::invalid_argument(message);
}

bool enabled() {
    const char * value = std::getenv("GGML_VULKAN_PTQ1_XMX");
    return value && std::strcmp(value, "1") == 0;
}

uint32_t env_u32(const char * name, uint32_t fallback) {
    const char * value = std::getenv(name);
    if (!value || !*value) return fallback;
    char * end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    require(end && *end == '\0' && parsed <= UINT32_MAX, "PTQ1 XMX invalid numeric environment value");
    return static_cast<uint32_t>(parsed);
}

uint32_t token_tile_for(uint32_t tokens) {
    const char * forced = std::getenv("GGML_VULKAN_PTQ1_XMX_TILE");
    if (forced && *forced) {
        const uint32_t tile = env_u32("GGML_VULKAN_PTQ1_XMX_TILE", 1);
        require(tile == 1 || tile == 2 || tile == 4, "PTQ1 XMX tile must be 1, 2 or 4");
        return tile;
    }
    if (tokens == 2) return 2;
    if (tokens == 4) return 4;
    return 1;
}

uint32_t local_size_for(uint32_t token_tile) {
    const uint32_t local = env_u32("GGML_VULKAN_PTQ1_XMX_LOCAL_SIZE", token_tile > 1 ? 8u : 4u);
    require(local == 4 || local == 8 || local == 16 || local == 32,
            "PTQ1 XMX local size must be 4, 8, 16 or 32");
    return local;
}

sycl::queue & xmx_queue() {
    // Deliberately process-lifetime. The explicit runtime shutdown below drains
    // USM/imported allocations while this Level Zero context is still alive.
    static sycl::queue * result = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        sycl::device selected;
        bool found = false;
        for (const auto & platform : sycl::platform::get_platforms()) {
            for (const auto & device : platform.get_devices(sycl::info::device_type::gpu)) {
                if (device.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
                try {
                    maple_w2::validate_dg2_device(device);
                    selected = device;
                    found = true;
                    break;
                } catch (const std::exception &) {
                }
            }
            if (found) break;
        }
        if (!found) throw std::runtime_error("PTQ1 XMX requires Intel Arc A750/A770 Level Zero");

        result = new sycl::queue(
            sycl::context(selected),
            selected,
            [](sycl::exception_list errors) {
                for (const auto & error : errors) std::rethrow_exception(error);
            },
            sycl::property_list{sycl::property::queue::in_order{}});
    });
    return *result;
}

struct Handle {
    HANDLE value = nullptr;
    ~Handle() { if (value && value != INVALID_HANDLE_VALUE) CloseHandle(value); }
};

struct ImportedAllocation {
    ze_context_handle_t context = nullptr;
    void * base = nullptr;
    uint64_t generation = 0;
    size_t size = 0;
    std::shared_ptr<void> owner;

    ~ImportedAllocation() {
        if (base) {
            (void) zeMemFree(context, base);
        }
    }
};

template<typename T>
struct ReusableBuffer {
    sycl::queue * queue = nullptr;
    T * data = nullptr;
    size_t capacity = 0;

    ~ReusableBuffer() {
        if (data && queue) {
            try { queue->wait_and_throw(); } catch (...) {}
            sycl::free(data, *queue);
        }
    }

    void ensure(sycl::queue & q, size_t count) {
        if (count <= capacity) return;
        if (data) {
            q.wait_and_throw();
            sycl::free(data, q);
        }
        data = sycl::malloc_device<T>(count, q);
        if (!data) throw std::bad_alloc();
        queue = &q;
        capacity = count;
    }

    void reset() {
        if (!data || !queue) return;
        queue->wait_and_throw();
        sycl::free(data, *queue);
        data = nullptr;
        queue = nullptr;
        capacity = 0;
    }
};

struct Runtime {
    sycl::queue * queue = nullptr;
    ze_context_handle_t ze_context = nullptr;
    ze_device_handle_t ze_device = nullptr;
    bool identity_checked = false;
    std::unordered_map<uint64_t, std::shared_ptr<ImportedAllocation>> imports;
    ReusableBuffer<int32_t> ids;
    ReusableBuffer<int32_t> status;
    ReusableBuffer<int8_t> a8_q;
    ReusableBuffer<float> a8_scales;
    ReusableBuffer<int32_t> invalid;
    uint64_t cache_hits = 0;
    uint64_t imports_created = 0;
};

std::mutex & runtime_mutex() {
    static std::mutex mutex;
    return mutex;
}

Runtime *& runtime_slot() {
    static Runtime * state = nullptr;
    return state;
}

Runtime & runtime() {
    auto *& state = runtime_slot();
    if (!state) state = new Runtime();
    auto & q = xmx_queue();
    if (!state->queue) {
        state->queue = &q;
        state->ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        state->ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());

        ze_device_external_memory_properties_t props{};
        props.stype = ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES;
        require(zeDeviceGetExternalMemoryProperties(state->ze_device, &props) == ZE_RESULT_SUCCESS,
                "PTQ1 XMX cannot query Level Zero external-memory properties");
        require((props.memoryAllocationImportTypes & ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32) != 0,
                "PTQ1 XMX driver lacks OPAQUE_WIN32 import");
    }
    return *state;
}

void verify_identity(Runtime & state, const ggml_vk_external_span & span) {
    if (state.identity_checked) return;

    ze_device_luid_ext_properties_t luid{};
    luid.stype = ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES;
    ze_device_properties_t props{};
    props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    props.pNext = &luid;
    require(zeDeviceGetProperties(state.ze_device, &props) == ZE_RESULT_SUCCESS,
            "PTQ1 XMX cannot query Level Zero device identity");

    maple_w2::level4::zc::DeviceIdentity vk_identity{
        span.vendor, span.device, span.node_mask, {}, span.luid_valid};
    maple_w2::level4::zc::DeviceIdentity ze_identity{
        props.vendorId, props.deviceId, luid.nodeMask, {}, true};
    std::memcpy(vk_identity.luid.data(), span.luid, 8);
    std::memcpy(ze_identity.luid.data(), luid.luid.id, 8);

    maple_w2::level4::zc::require_same_windows_device(
        vk_identity,
        ze_identity,
        span.device_uuid,
        props.uuid.id,
        (props.flags & ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE) != 0);
    state.identity_checked = true;
}

void * import_tensor(
        Runtime & state,
        ggml_vk_external_lease * lease,
        const ggml_vk_external_api * api,
        const ggml_tensor * tensor,
        size_t alignment) {
    ggml_vk_external_span span{};
    require(api && api->abi_version == 1 && api->get_span &&
            api->get_span(lease, tensor, &span),
            "PTQ1 XMX Vulkan tensor is not exportable");
    Handle handle{static_cast<HANDLE>(span.memory_handle)};
    require(handle.value != nullptr && handle.value != INVALID_HANDLE_VALUE,
            "PTQ1 XMX external handle is invalid");

    auto found = state.imports.find(span.allocation_id);
    if (found == state.imports.end()) {
        verify_identity(state, span);

        auto allocation = std::make_shared<ImportedAllocation>();
        allocation->context = state.ze_context;
        allocation->generation = span.generation;
        allocation->size = span.allocation_size;
        allocation->owner = span.owner;

        ze_external_memory_import_win32_handle_t import_desc{};
        import_desc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32;
        import_desc.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32;
        import_desc.handle = handle.value;

        ze_device_mem_alloc_desc_t alloc_desc{};
        alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
        alloc_desc.pNext = &import_desc;

        require(zeMemAllocDevice(
                    state.ze_context,
                    &alloc_desc,
                    span.allocation_size,
                    0,
                    state.ze_device,
                    &allocation->base) == ZE_RESULT_SUCCESS,
                "PTQ1 XMX Level Zero import failed");

        found = state.imports.emplace(span.allocation_id, std::move(allocation)).first;
        ++state.imports_created;
    } else {
        ++state.cache_hits;
    }

    const auto & allocation = *found->second;
    require(allocation.generation == span.generation &&
            allocation.size == span.allocation_size &&
            allocation.owner,
            "PTQ1 XMX stale external allocation");

    require(span.bind_offset <= allocation.size &&
            span.tensor_offset <= allocation.size - span.bind_offset,
            "PTQ1 XMX external offset overflow");
    const size_t offset = span.bind_offset + span.tensor_offset;
    require(alignment && offset % alignment == 0 &&
            span.bytes <= allocation.size - offset,
            "PTQ1 XMX external tensor range/alignment mismatch");

    return static_cast<uint8_t *>(allocation.base) + offset;
}

bool supported(const ggml_tensor * node) {
    if (!enabled() || !node || node->op != GGML_OP_MUL_MAT ||
        !node->src[0] || !node->src[1]) return false;

    const auto * w = node->src[0];
    const auto * x = node->src[1];

    if (w->type != GGML_TYPE_PTQ1_0 ||
        (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_F16) ||
        node->type != GGML_TYPE_F32) return false;

    if (!ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(node)) return false;
    if (w->ne[0] <= 0 || w->ne[0] % 256 != 0 ||
        w->ne[1] <= 0 || w->ne[1] % 8 != 0 ||
        w->ne[2] != 1 || w->ne[3] != 1) return false;

    const int64_t ncols = x->ne[1];
    if (x->ne[0] != w->ne[0] ||
        ncols <= 0 || ncols > 2048 ||
        x->ne[2] != 1 || x->ne[3] != 1) return false;

    if (node->ne[0] != w->ne[1] || node->ne[1] != ncols ||
        node->ne[2] != 1 || node->ne[3] != 1 ||
        w->ne[0] > UINT32_MAX || w->ne[1] > UINT32_MAX) return false;

    try {
        const maple_w2::Shape shape{
            static_cast<uint32_t>(w->ne[0]),
            static_cast<uint32_t>(w->ne[1]),
            1};
        return ggml_nbytes(w) == maple_w2::ptq1_nbytes(shape);
    } catch (...) {
        return false;
    }
}

bool execute(
        ggml_tensor * node,
        ggml_vk_external_lease * lease,
        const ggml_vk_external_api * api) {
    try {
        if (!supported(node) || !lease || !api || !api->release_to_external) return false;
        std::lock_guard<std::mutex> lock(runtime_mutex());
        auto & state = runtime();
        auto & q = *state.queue;

        const auto * wt = node->src[0];
        const auto * xt = node->src[1];
        const uint32_t k = static_cast<uint32_t>(wt->ne[0]);
        const uint32_t m = static_cast<uint32_t>(wt->ne[1]);
        const uint32_t tokens = static_cast<uint32_t>(xt->ne[1]);

        auto * w = static_cast<const uint8_t *>(import_tensor(state, lease, api, wt, alignof(uint16_t)));
        const void * x = import_tensor(state, lease, api, xt, xt->type == GGML_TYPE_F32 ? alignof(float) : alignof(uint16_t));
        auto * y = static_cast<float *>(import_tensor(state, lease, api, node, alignof(float)));

        constexpr uint32_t a8_group = 32;
        state.ids.ensure(q, tokens);
        state.status.ensure(q, tokens);
        const size_t a8_elements = size_t(tokens) * k;
        const size_t a8_groups = a8_elements / a8_group;
        state.a8_q.ensure(q, a8_elements);
        state.a8_scales.ensure(q, a8_groups);
        state.invalid.ensure(q, a8_groups);
        q.memset(state.ids.data, 0, tokens * sizeof(int32_t));
        q.memset(state.status.data, 0, tokens * sizeof(int32_t));

        maple_w2::DeviceProblem problem{};
        problem.shape = {k, m, 1};
        problem.tokens = tokens;
        problem.topk = 1;
        problem.per_selection = false;
        problem.layout = maple_w2::Layout::native_ptq1;
        problem.w0 = w;
        problem.x = x;
        problem.ids = state.ids.data;
        problem.y0 = y;
        problem.status = state.status.data;
        problem.activation_type =
            xt->type == GGML_TYPE_F16 ? maple_w2::ActivationType::f16 : maple_w2::ActivationType::f32;

        maple_w2::A8Options options{};
        options.group = a8_group;
        options.mode = maple_w2::A8Mode::staged;
        options.kernel.split_k = 1;
        options.ptq1_token_tile = token_tile_for(tokens);
        options.kernel.local_size = local_size_for(options.ptq1_token_tile);

        maple_w2::A8Workspace workspace{};
        workspace.q = state.a8_q.data;
        workspace.scales = state.a8_scales.data;
        workspace.invalid = state.invalid.data;

        require(api->release_to_external(lease), "PTQ1 XMX Vulkan ownership release failed");

        auto run = maple_w2::enqueue_a8(q, problem, options, workspace, nullptr);
        run.done.wait_and_throw();

        std::vector<int32_t> status(tokens);
        std::vector<int32_t> invalid(a8_groups);
        q.memcpy(status.data(), state.status.data, tokens * sizeof(int32_t));
        q.memcpy(invalid.data(), state.invalid.data, a8_groups * sizeof(int32_t));
        q.wait_and_throw();
        for (int32_t value : status) require(value == 0, "PTQ1 XMX device status failure");
        for (int32_t value : invalid) require(value == 0, "PTQ1 XMX activation contains non-finite data");

        static std::atomic<uint32_t> logs{0};
        if (logs.fetch_add(1) < 8) {
            std::fprintf(stderr,
                "ptq1-xmx m=%u n=%u k=%u group=%u staged_a8=1 native=1 tile=%u local=%u imports=%llu cache_hits=%llu\n",
                m, tokens, k, a8_group,
                options.ptq1_token_tile, options.kernel.local_size,
                (unsigned long long) state.imports_created,
                (unsigned long long) state.cache_hits);
        }
        return true;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "PTQ1 XMX bridge failed: %s\n", error.what());
        return false;
    }
}

} // namespace

void llama_ptq1_xmx_shutdown() {
    std::lock_guard<std::mutex> lock(runtime_mutex());
    Runtime * state = runtime_slot();
    if (!state || !state->queue) return;

    try {
        state->queue->wait_and_throw();
        state->ids.reset();
        state->status.reset();
        state->a8_q.reset();
        state->a8_scales.reset();
        state->invalid.reset();
        // Imported mappings own the Vulkan allocations. Release the Level Zero
        // import before dropping those source owners.
        state->imports.clear();
        state->identity_checked = false;
        state->cache_hits = 0;
        state->imports_created = 0;
    } catch (const std::exception & error) {
        // Fail closed at teardown: keep the heap-owned runtime alive rather than
        // freeing Vulkan memory behind an uncertain Level Zero operation.
        std::fprintf(stderr, "PTQ1 XMX shutdown drain failed: %s\n", error.what());
    }
}

void llama_ptq1_xmx_register() {
    if (!enabled()) return;
    auto reg = ggml_backend_reg_by_name("Vulkan");
    require(reg != nullptr, "PTQ1 XMX requires Vulkan backend");
    auto install = reinterpret_cast<ggml_vk_ptq1_register>(
        ggml_backend_reg_get_proc_address(reg, "ggml_vk_ptq1_register_v1"));
    require(install && install(supported, execute, 1),
            "Vulkan backend lacks PTQ1 XMX executor ABI");
    std::fprintf(stderr, "PTQ1 XMX executor registered; set before model allocation=1\n");
}
