// SPDX-License-Identifier: MIT
#include "maple_ptq1_ggml_bridge.hpp"

#include "dg2_validation.hpp"
#include "level4_device_identity_win32.hpp"
#include "maple_ptq1a8.hpp"
#include "maple_w2a8.hpp"

#include "ggml.h"
#include "ggml-vulkan-external.h"

#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <array>
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
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

bool enabled() {
    const char * value = std::getenv("GGML_VULKAN_PTQ1_XMX");
    return value && std::strcmp(value, "1") == 0;
}

std::mutex & bridge_mutex() {
    static std::mutex mutex;
    return mutex;
}

sycl::queue & bridge_queue() {
    static std::unique_ptr<sycl::queue> queue;
    static std::once_flag once;

    std::call_once(once, [] {
        sycl::device selected;
        bool found = false;

        for (const auto & platform : sycl::platform::get_platforms()) {
            for (const auto & device : platform.get_devices(sycl::info::device_type::gpu)) {
                if (device.get_backend() != sycl::backend::ext_oneapi_level_zero) {
                    continue;
                }
                try {
                    maple_w2::validate_dg2_device(device);
                    selected = device;
                    found = true;
                    break;
                } catch (const std::exception &) {
                }
            }
            if (found) {
                break;
            }
        }

        if (!found) {
            throw std::runtime_error("PTQ1 XMX bridge requires an Intel Arc A750/A770 Level Zero GPU");
        }

        auto async_handler = [](sycl::exception_list errors) {
            for (const auto & error : errors) {
                std::rethrow_exception(error);
            }
        };
        queue = std::make_unique<sycl::queue>(
            sycl::context(selected),
            selected,
            async_handler,
            sycl::property_list{sycl::property::queue::in_order{}});
    });

    return *queue;
}

template<typename T>
struct DeviceBuffer {
    sycl::queue * queue = nullptr;
    T * data = nullptr;
    size_t count = 0;

    DeviceBuffer() = default;

    DeviceBuffer(sycl::queue & q, size_t n) : queue(&q), count(n) {
        if (n == 0) {
            return;
        }
        data = sycl::malloc_device<T>(n, q);
        if (!data) {
            throw std::bad_alloc();
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer & operator=(const DeviceBuffer &) = delete;

    ~DeviceBuffer() {
        if (data) {
            try {
                queue->wait_and_throw();
            } catch (...) {
            }
            sycl::free(data, *queue);
        }
    }
};

struct ImportedAllocation {
    ze_context_handle_t context = nullptr;
    void * base = nullptr;
    uint64_t generation = 0;
    size_t size = 0;

    ~ImportedAllocation() {
        if (base) {
            zeMemFree(context, base);
        }
    }
};

struct Handle {
    HANDLE value = nullptr;
    ~Handle() {
        if (value) {
            CloseHandle(value);
        }
    }
};

class ExternalMappings {
public:
    explicit ExternalMappings(sycl::queue & q) : q_(q) {
        device_ = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_.get_device());
        context_ = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_.get_context());
    }

    void * import_tensor(
        ggml_vk_external_lease * lease,
        const ggml_vk_external_api * api,
        const ggml_tensor * tensor) {

        ggml_vk_external_span span{};
        require(api && api->abi_version == 1 && api->get_span && api->get_span(lease, tensor, &span),
                "PTQ1 XMX Vulkan tensor is not exportable");

        Handle handle{static_cast<HANDLE>(span.memory_handle)};
        require(handle.value != nullptr, "PTQ1 XMX external handle is null");

        auto it = allocations_.find(span.allocation_id);
        if (it == allocations_.end()) {
            verify_identity(span);

            auto allocation = std::make_unique<ImportedAllocation>();
            allocation->context = context_;
            allocation->generation = span.generation;
            allocation->size = span.allocation_size;

            ze_external_memory_import_win32_handle_t import_desc{};
            import_desc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32;
            import_desc.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32;
            import_desc.handle = handle.value;

            ze_device_mem_alloc_desc_t alloc_desc{};
            alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
            alloc_desc.pNext = &import_desc;

            require(
                zeMemAllocDevice(
                    context_,
                    &alloc_desc,
                    span.allocation_size,
                    0,
                    device_,
                    &allocation->base) == ZE_RESULT_SUCCESS,
                "PTQ1 XMX Level Zero external-memory import failed");

            it = allocations_.emplace(span.allocation_id, std::move(allocation)).first;
        }

        const auto & allocation = *it->second;
        require(allocation.generation == span.generation && allocation.size == span.allocation_size,
                "PTQ1 XMX stale external allocation");
        require(span.tensor_offset <= allocation.size && span.bytes <= allocation.size - span.tensor_offset,
                "PTQ1 XMX external tensor range overflow");

        return static_cast<uint8_t *>(allocation.base) + span.tensor_offset;
    }

private:
    void verify_identity(const ggml_vk_external_span & span) {
        ze_device_luid_ext_properties_t luid{};
        luid.stype = ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES;

        ze_device_properties_t props{};
        props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        props.pNext = &luid;

        require(zeDeviceGetProperties(device_, &props) == ZE_RESULT_SUCCESS,
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
    }

    sycl::queue & q_;
    ze_device_handle_t device_ = nullptr;
    ze_context_handle_t context_ = nullptr;
    std::unordered_map<uint64_t, std::unique_ptr<ImportedAllocation>> allocations_;
};

bool shape_supported(const ggml_tensor * node) {
    if (!node || node->op != GGML_OP_MUL_MAT || !node->src[0] || !node->src[1]) {
        return false;
    }

    const ggml_tensor * weights = node->src[0];
    const ggml_tensor * activations = node->src[1];

    if (weights->type != GGML_TYPE_PTQ1_0 ||
        (activations->type != GGML_TYPE_F32 && activations->type != GGML_TYPE_F16) ||
        node->type != GGML_TYPE_F32) {
        return false;
    }

    if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(activations) || !ggml_is_contiguous(node)) {
        return false;
    }

    if (weights->ne[0] <= 0 || weights->ne[0] % 256 != 0 ||
        weights->ne[1] <= 0 || weights->ne[1] % 8 != 0 ||
        weights->ne[2] != 1 || weights->ne[3] != 1) {
        return false;
    }

    const int64_t ncols = activations->ne[1];
    if (activations->ne[0] != weights->ne[0] ||
        (ncols != 1 && ncols != 2 && ncols != 4) ||
        activations->ne[2] != 1 || activations->ne[3] != 1) {
        return false;
    }

    if (node->ne[0] != weights->ne[1] || node->ne[1] != ncols ||
        node->ne[2] != 1 || node->ne[3] != 1) {
        return false;
    }

    if (weights->ne[0] > UINT32_MAX || weights->ne[1] > UINT32_MAX || ncols > UINT32_MAX) {
        return false;
    }

    try {
        const maple_w2::Shape shape{
            static_cast<uint32_t>(weights->ne[0]),
            static_cast<uint32_t>(weights->ne[1]),
            1};
        return ggml_nbytes(weights) == maple_w2::ptq1_nbytes(shape);
    } catch (...) {
        return false;
    }
}

} // namespace

bool maple_ptq1_xmx_supported(const ggml_tensor * node) {
    return enabled() && shape_supported(node);
}

bool maple_ptq1_xmx_execute(
    ggml_tensor * node,
    ggml_vk_external_lease * lease,
    const ggml_vk_external_api * api) {

    if (!enabled() || !shape_supported(node) || !lease || !api || api->abi_version != 1 ||
        !api->release_to_external) {
        return false;
    }

    try {
        std::lock_guard<std::mutex> guard(bridge_mutex());

        auto & q = bridge_queue();
        ExternalMappings mappings(q);

        const ggml_tensor * weights = node->src[0];
        const ggml_tensor * activations = node->src[1];

        auto * w = static_cast<const uint8_t *>(mappings.import_tensor(lease, api, weights));
        const void * x = mappings.import_tensor(lease, api, activations);
        auto * y = static_cast<float *>(mappings.import_tensor(lease, api, node));

        const uint32_t k = static_cast<uint32_t>(weights->ne[0]);
        const uint32_t m = static_cast<uint32_t>(weights->ne[1]);
        const uint32_t tokens = static_cast<uint32_t>(activations->ne[1]);

        DeviceBuffer<int32_t> ids(q, tokens);
        DeviceBuffer<int32_t> status(q, tokens);
        const size_t invalid_count = size_t(tokens) * (k / 128);
        DeviceBuffer<int32_t> invalid(q, invalid_count);

        q.memset(ids.data, 0, ids.count * sizeof(int32_t));
        q.memset(status.data, 0, status.count * sizeof(int32_t));
        q.memset(invalid.data, 0, invalid.count * sizeof(int32_t));

        maple_w2::DeviceProblem problem{};
        problem.shape = {k, m, 1};
        problem.tokens = tokens;
        problem.topk = 1;
        problem.per_selection = false;
        problem.layout = maple_w2::Layout::native_ptq1;
        problem.w0 = w;
        problem.x = x;
        problem.ids = ids.data;
        problem.y0 = y;
        problem.status = status.data;
        problem.activation_type =
            activations->type == GGML_TYPE_F16 ? maple_w2::ActivationType::f16 : maple_w2::ActivationType::f32;

        maple_w2::A8Options options{};
        options.group = 128;
        options.mode = maple_w2::A8Mode::fused;
        options.kernel.split_k = 1;
        options.kernel.local_size = 4;

        maple_w2::A8Workspace workspace{};
        workspace.invalid = invalid.data;
        workspace.capture = false;

        // Importing a Vulkan allocation does not transfer queue-family ownership.
        // Release all spans only after every pointer has been imported, immediately
        // before Level Zero starts touching the memory.
        require(api->release_to_external(lease), "PTQ1 XMX Vulkan ownership release failed");

        auto run = maple_w2::enqueue_a8(q, problem, options, workspace, nullptr);
        run.done.wait_and_throw();

        std::vector<int32_t> host_status(tokens);
        std::vector<int32_t> host_invalid(invalid_count);
        q.memcpy(host_status.data(), status.data, host_status.size() * sizeof(int32_t));
        q.memcpy(host_invalid.data(), invalid.data, host_invalid.size() * sizeof(int32_t));
        q.wait_and_throw();

        for (int32_t value : host_status) {
            require(value == 0, "PTQ1 XMX device status reported invalid expert state");
        }
        for (int32_t value : host_invalid) {
            require(value == 0, "PTQ1 XMX activation contains non-finite data");
        }

        std::fprintf(
            stderr,
            "maple-ptq1-xmx complete m=%u n=%u k=%u group=128 layout=native_ptq1 host_weight_copy=0\n",
            m,
            tokens,
            k);
        return true;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "maple-ptq1-xmx failed: %s\n", error.what());
        return false;
    }
}
