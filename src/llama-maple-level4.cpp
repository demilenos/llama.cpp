#include "llama-maple-level4.h"

#include "ggml-backend.h"
#include "ggml-vulkan/ggml-vulkan-hybrid.h"
#include "level4_device_identity_win32.hpp"
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <windows.h>
#include "dg2_validation.hpp"
#include "maple_level4.hpp"
#include "w2a8_reference.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstring>
#include <cstdio>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using maple_w2::Shape;
using maple_w2::level4::Config;
using maple_w2::level4::Entry;

struct CachedWeight {
    const ggml_tensor * tensor = nullptr;
    size_t source_bytes = 0;
    Shape shape{};
    std::vector<uint8_t> packed;
};

struct ImportedAllocation {
    ze_context_handle_t context = nullptr;
    void * base = nullptr;
    uint64_t generation = 0;
    size_t size = 0;
    std::shared_ptr<void> owner;
    ~ImportedAllocation() { if (base) zeMemFree(context, base); }
};
struct RuntimeState {
    std::unordered_map<uint64_t, std::shared_ptr<ImportedAllocation>> imports;
    std::unordered_map<const ggml_tensor *, CachedWeight> weights;
};

std::mutex & bridge_mutex() {
    static std::mutex m;
    return m;
}

std::shared_ptr<RuntimeState> runtime_for(llama_maple_level4_params & p) {
    if (!p.runtime) {
        p.runtime = std::shared_ptr<void>(std::make_shared<RuntimeState>());
    }
    return std::static_pointer_cast<RuntimeState>(p.runtime);
}

sycl::queue & maple_queue() {
    static std::unique_ptr<sycl::queue> result;
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
            throw std::runtime_error("Maple Level4 requires an Intel Arc A750/A770 Level Zero GPU");
        }
        result = std::make_unique<sycl::queue>(sycl::context(selected), selected, [](sycl::exception_list errors) {
            for (const auto & error : errors) {
                std::rethrow_exception(error);
            }
        });
    });
    return *result;
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
    DeviceBuffer(DeviceBuffer && other) noexcept : queue(other.queue), data(other.data), count(other.count) {
        other.queue = nullptr;
        other.data = nullptr;
        other.count = 0;
    }
    ~DeviceBuffer() {
        if (data) {
            try { queue->wait_and_throw(); } catch (...) {}
            sycl::free(data, *queue);
        }
    }
};

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

void require_finite(const float * data, size_t count, const char * name) {
    require(data != nullptr, name);
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(data[i])) {
            throw std::invalid_argument(std::string(name) + " contains nonfinite data");
        }
    }
}

void require_f32_contiguous(const ggml_tensor * tensor, const char * name) {
    require(tensor != nullptr, name);
    require(tensor->type == GGML_TYPE_F32, "Maple Level4 boundary must be F32");
    require(ggml_is_contiguous(tensor), "Maple Level4 boundary must be contiguous");
    require(tensor->data != nullptr, "Maple Level4 boundary has no host data");
}

Shape tensor_shape(const ggml_tensor * tensor, uint32_t k, uint32_t m, uint32_t experts) {
    require(tensor != nullptr, "missing Maple Level4 weight tensor");
    require(tensor->type == GGML_TYPE_TQ2_0, "Maple Level4 projection must be GGML_TYPE_TQ2_0");
    require(tensor->ne[0] == k && tensor->ne[1] == m && tensor->ne[2] == experts,
            "Maple Level4 projection shape mismatch");
    require(tensor->ne[3] == 1, "Maple Level4 projection must be rank 3");
    return {k, m, experts};
}

const std::vector<uint8_t> & cached_tq2(RuntimeState & state, const ggml_tensor * tensor, Shape shape) {
    (void) tensor_shape(tensor, shape.k, shape.m, shape.experts);
    require(ggml_is_contiguous(tensor), "Maple Level4 projection must be contiguous");
    auto it = state.weights.find(tensor);
    const size_t source_bytes = ggml_nbytes(tensor);
    const size_t expected = maple_w2::nbytes(shape);
    if (it != state.weights.end() && it->second.source_bytes == source_bytes && it->second.shape.k == shape.k &&
        it->second.shape.m == shape.m && it->second.shape.experts == shape.experts) {
        return it->second.packed;
    }
    CachedWeight cached;
    cached.tensor = tensor;
    cached.source_bytes = source_bytes;
    cached.shape = shape;
    std::vector<uint8_t> native(source_bytes);
    require(source_bytes == expected, "Maple Level4 TQ2 byte size mismatch");
    ggml_backend_tensor_get(tensor, native.data(), 0, source_bytes);
    cached.packed = maple_w2::repack_s2tile8(native, shape);
    auto inserted = state.weights.insert_or_assign(tensor, std::move(cached));
    return inserted.first->second.packed;
}

std::vector<float> tensor_f32(const ggml_tensor * tensor, size_t count, const char * name) {
    require(tensor != nullptr, name);
    require(ggml_is_contiguous(tensor), "Maple Level4 small weight must be contiguous");
    require(tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16, "Maple Level4 small weight must be F16 or F32");
    require(ggml_nbytes(tensor) == count * (tensor->type == GGML_TYPE_F16 ? sizeof(uint16_t) : sizeof(float)), "Maple Level4 F16/F32 weight size mismatch");
    std::vector<float> result(count);
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, result.data(), 0, result.size() * sizeof(float));
    } else {
        std::vector<uint16_t> half(count);
        ggml_backend_tensor_get(tensor, half.data(), 0, half.size() * sizeof(uint16_t));
        for (size_t i = 0; i < count; ++i) {
            result[i] = maple_w2::half_to_float(half[i]);
        }
    }
    require_finite(result.data(), result.size(), name);
    return result;
}

template<typename T>
void upload(sycl::queue & q, DeviceBuffer<T> & dst, const T * source, size_t count) {
    require(dst.data != nullptr && dst.count == count, "Maple Level4 upload allocation mismatch");
    q.memcpy(dst.data, source, count * sizeof(T)).wait_and_throw();
}

template<typename T>
std::vector<T> download(sycl::queue & q, const T * source, size_t count) {
    std::vector<T> result(count);
    q.memcpy(result.data(), source, count * sizeof(T)).wait_and_throw();
    return result;
}

float * maple_import(RuntimeState & state, sycl::queue & q, ggml_vk_external_lease * lease, const ggml_vk_external_api * api, const ggml_tensor * tensor) {
    ggml_vk_external_span span{};
    require(api && api->abi_version == 1 && api->get_span(lease, tensor, &span), "Vulkan tensor is not exportable");
    struct Handle { HANDLE h; ~Handle() { if (h) CloseHandle(h); } } handle{span.memory_handle};
    auto zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    auto zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
    // bridge_mutex serializes this weak registry across graph-owned states.
    static std::unordered_map<uint64_t, std::weak_ptr<ImportedAllocation>> registry;
    auto & imported = state.imports[span.allocation_id];
    if (!imported) imported = registry[span.allocation_id].lock();
    if (!imported) {
        ze_device_luid_ext_properties_t luid{}; luid.stype = ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES;
        ze_device_properties_t props{}; props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES; props.pNext = &luid;
        require(zeDeviceGetProperties(zd, &props) == ZE_RESULT_SUCCESS, "Cannot query Level Zero identity");
        maple_w2::level4::zc::DeviceIdentity a{span.vendor, span.device, span.node_mask, {}, span.luid_valid};
        maple_w2::level4::zc::DeviceIdentity b{props.vendorId, props.deviceId, luid.nodeMask, {}, true};
        std::memcpy(a.luid.data(), span.luid, 8); std::memcpy(b.luid.data(), luid.luid.id, 8);
        std::fprintf(stderr, "Maple identity vk_mask=%u ze_mask=%u vk_uuid=", a.node_mask, b.node_mask);
        for (auto byte : span.device_uuid) std::fprintf(stderr, "%02x", unsigned(byte));
        std::fprintf(stderr, " ze_uuid="); for (auto byte : props.uuid.id) std::fprintf(stderr, "%02x", unsigned(byte)); std::fprintf(stderr, "\n");
        maple_w2::level4::zc::require_same_windows_device(a, b, span.device_uuid, props.uuid.id, (props.flags & ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE) != 0);
        auto candidate = std::make_shared<ImportedAllocation>();
        candidate->context = zc; candidate->owner = span.owner; candidate->size = span.allocation_size; candidate->generation = span.generation;
        ze_external_memory_import_win32_handle_t imp{}; imp.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32;
        imp.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32; imp.handle = span.memory_handle;
        ze_device_mem_alloc_desc_t desc{}; desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC; desc.pNext = &imp;
        require(zeMemAllocDevice(zc, &desc, span.allocation_size, 0, zd, &candidate->base) == ZE_RESULT_SUCCESS, "Level Zero external import failed");
        imported = std::move(candidate);
        registry[span.allocation_id] = imported;
        if (registry.size() > 256) {
            for (auto it = registry.begin(); it != registry.end();) {
                if (it->second.expired()) it = registry.erase(it); else ++it;
            }
        }
    }
    require(imported->context == zc && imported->generation == span.generation && imported->size == span.allocation_size, "Stale external allocation");
    require(span.bind_offset <= span.allocation_size && span.tensor_offset <= span.allocation_size - span.bind_offset, "External offset overflow");
    size_t offset = span.bind_offset + span.tensor_offset;
    require(offset % alignof(float) == 0 && span.bytes <= span.allocation_size - offset, "External tensor out of range");
    return reinterpret_cast<float *>(static_cast<uint8_t *>(imported->base) + offset);
}

} // namespace

static void maple_compute(ggml_tensor * dst, void * userdata, ggml_vk_external_lease * lease, const ggml_vk_external_api * api) {
    {
        require(dst != nullptr && userdata != nullptr, "Maple Level4 callback received null state");
        auto & p = *static_cast<llama_maple_level4_params *>(userdata);
        std::lock_guard<std::mutex> lock(bridge_mutex());
        const uint32_t tokens = static_cast<uint32_t>(dst->ne[1]);
        require(tokens > 0, "Maple Level4 invalid token count");
        require(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) && dst->data != nullptr,
                "Maple Level4 destination must be contiguous F32");
        require(p.hidden && p.attention && p.q_width && p.kv_width && p.ffn && p.experts && p.topk,
                "Maple Level4 dimensions are incomplete");
        require(p.token_tile == 1 || p.token_tile == 4 || p.token_tile == 8, "Maple Level4 token tile must be 1, 4, or 8");
        require(std::isfinite(p.epsilon) && p.epsilon > 0, "Maple Level4 epsilon must be positive and finite");
        const Entry entry = p.current_layer < 0 ? Entry::bootstrap : (p.next_layer < 0 ? Entry::terminal : Entry::advance);
        require(std::isfinite(p.clamp) && (entry == Entry::bootstrap || p.clamp == 7.0f), "Maple Level4 clamp must be 7 for FFN entries");
        require(dst->ne[1] <= 2048, "Maple Level4 token count exceeds module limit");

        const size_t packed_width = size_t(p.hidden) + (entry == Entry::terminal ? 0 : size_t(p.q_width) + 2 * p.kv_width);
        require(dst->ne[0] == int64_t(packed_width), "Maple Level4 packed destination width mismatch");
        require(dst->ne[0] * dst->ne[1] <= INT64_MAX / int64_t(sizeof(float)), "Maple Level4 destination size overflow");

        require_f32_contiguous(dst->src[0], "Maple Level4 residual input");
        require(dst->src[0]->ne[0] == p.hidden && dst->src[0]->ne[1] == tokens, "Maple Level4 residual shape mismatch");
        const ggml_tensor * attention = dst->src[1];
        if (entry != Entry::bootstrap) {
            require_f32_contiguous(attention, "Maple Level4 pre-O attention input");
            require(attention->ne[0] == p.attention && attention->ne[1] == tokens, "Maple Level4 attention shape mismatch");
        }

        Config config;
        config.tokens = tokens; config.hidden = p.hidden; config.attention = p.attention;
        config.q_width = p.q_width; config.kv_width = p.kv_width; config.ffn = p.ffn;
        config.experts = p.experts; config.topk = p.topk; config.entry = entry;
        config.ffn_epsilon = p.epsilon; config.next_epsilon = p.epsilon;
        auto plan = maple_w2::level4::make_plan(config, 1, 1, 1, 1, 1, 1, true);
        auto state = runtime_for(p);
        auto & q = maple_queue();

        float * imported_x = maple_import(*state, q, lease, api, dst->src[0]);
        float * imported_a = entry != Entry::bootstrap ? maple_import(*state, q, lease, api, attention) : nullptr;
        float * imported_out = maple_import(*state, q, lease, api, dst);
        const size_t hidden_n = size_t(tokens) * p.hidden;
        DeviceBuffer<uint8_t> arena(q, plan.bytes);
        auto workspace = maple_w2::level4::bind_workspace(arena.data, plan.bytes, std::move(plan));

        auto make_tq2 = [&](ggml_tensor * tensor, Shape shape) {
            const auto & packed = cached_tq2(*state, tensor, shape);
            DeviceBuffer<uint8_t> buffer(q, packed.size());
            upload(q, buffer, packed.data(), packed.size());
            return std::pair<DeviceBuffer<uint8_t>, const std::vector<uint8_t> *>(std::move(buffer), &packed);
        };
        auto qkv_q = entry == Entry::terminal ? std::pair<DeviceBuffer<uint8_t>, const std::vector<uint8_t> *>(DeviceBuffer<uint8_t>(), nullptr) : make_tq2(p.q, {p.hidden, p.q_width, 1});
        auto qkv_k = entry == Entry::terminal ? std::pair<DeviceBuffer<uint8_t>, const std::vector<uint8_t> *>(DeviceBuffer<uint8_t>(), nullptr) : make_tq2(p.k, {p.hidden, p.kv_width, 1});
        auto qkv_v = entry == Entry::terminal ? std::pair<DeviceBuffer<uint8_t>, const std::vector<uint8_t> *>(DeviceBuffer<uint8_t>(), nullptr) : make_tq2(p.v, {p.hidden, p.kv_width, 1});
        auto o = entry == Entry::bootstrap ? std::pair<DeviceBuffer<uint8_t>, const std::vector<uint8_t> *>(DeviceBuffer<uint8_t>(), nullptr) : make_tq2(p.o, {p.attention, p.hidden, 1});
        auto gate = entry == Entry::bootstrap ? std::pair<DeviceBuffer<uint8_t>, const std::vector<uint8_t> *>(DeviceBuffer<uint8_t>(), nullptr) : make_tq2(p.gate, {p.hidden, p.ffn, p.experts});
        auto up = entry == Entry::bootstrap ? std::pair<DeviceBuffer<uint8_t>, const std::vector<uint8_t> *>(DeviceBuffer<uint8_t>(), nullptr) : make_tq2(p.up, {p.hidden, p.ffn, p.experts});
        auto down = entry == Entry::bootstrap ? std::pair<DeviceBuffer<uint8_t>, const std::vector<uint8_t> *>(DeviceBuffer<uint8_t>(), nullptr) : make_tq2(p.down, {p.ffn, p.hidden, p.experts});
        auto ffn_norm_h = entry == Entry::bootstrap ? std::vector<float>() : tensor_f32(p.ffn_norm, p.hidden, "Maple Level4 FFN norm");
        auto next_norm_h = entry == Entry::terminal ? std::vector<float>() : tensor_f32(p.next_attn_norm, p.hidden, "Maple Level4 next attention norm");
        auto router_h = entry == Entry::bootstrap ? std::vector<float>() : tensor_f32(p.router, size_t(p.hidden) * p.experts, "Maple Level4 router");
        DeviceBuffer<float> ffn_norm(q, ffn_norm_h.size()), next_norm(q, next_norm_h.size()), router(q, router_h.size());
        if (!ffn_norm_h.empty()) upload(q, ffn_norm, ffn_norm_h.data(), ffn_norm_h.size());
        if (!next_norm_h.empty()) upload(q, next_norm, next_norm_h.data(), next_norm_h.size());
        if (!router_h.empty()) upload(q, router, router_h.data(), router_h.size());

        maple_w2::level4::Weights weights;
        weights.o = o.first.data; weights.q = qkv_q.first.data; weights.k = qkv_k.first.data; weights.v = qkv_v.first.data;
        weights.gate = gate.first.data; weights.up = up.first.data; weights.down = down.first.data;
        weights.ffn_norm = ffn_norm.data; weights.next_attn_norm = next_norm.data; weights.router = router.data;
        weights.current_layer = p.current_layer; weights.next_layer = p.next_layer;
        weights.dense_layout = maple_w2::Layout::s2tile8; weights.moe_layout = maple_w2::Layout::s2tile8;
        maple_w2::level4::Bindings bindings{};
        maple_w2::level4::Options options;
        options.boundary = maple_w2::level4::BoundaryPolicy::direct_required;
        options.moe.gate_tokens_per_tile = 1; options.moe.down_tokens_per_tile = 1; options.moe.token_tile = p.token_tile;
        options.moe.clamp = p.clamp; options.moe.input_group = 32; options.moe.hidden_group = 32;
        options.moe.expert_grouping = true; options.moe.schedule = maple_w2::MoeSchedule::grouped;
        options.moe.gate_kernel.split_k = 1; options.moe.gate_kernel.local_size = 4;
        options.moe.down_kernel.split_k = 1; options.moe.down_kernel.local_size = 4;
        {
            float * x = imported_x;
            float * a = imported_a;
            float * out = imported_out;
            bindings.residual = maple_w2::level4::contiguous_read(x, tokens, p.hidden);
            bindings.attention = entry == Entry::bootstrap ? maple_w2::level4::ReadView{} : maple_w2::level4::contiguous_read(a, tokens, p.attention);
            bindings.hidden = maple_w2::level4::contiguous_write(out, tokens, p.hidden);
            if (entry != Entry::terminal) {
                out += hidden_n;
                bindings.q = maple_w2::level4::contiguous_write(out, tokens, p.q_width); out += size_t(tokens) * p.q_width;
                bindings.k = maple_w2::level4::contiguous_write(out, tokens, p.kv_width); out += size_t(tokens) * p.kv_width;
                bindings.v = maple_w2::level4::contiguous_write(out, tokens, p.kv_width);
            }
            maple_w2::level4::validate(weights, bindings, options, workspace);
            require(api->release_to_external(lease), "Vulkan external release failed");
        }
        auto run = maple_w2::level4::enqueue(q, weights, bindings, options, workspace);
        run.done.wait_and_throw();
        auto status = download(q, run.status, tokens);
        for (int32_t value : status) require(value == 0, "Maple Level4 device status reported nonfinite or invalid data");
        {
            require(run.boundary_traffic.bytes() == 0, "Level4 unexpectedly copied a boundary");
            std::fprintf(stderr, "maple-level4 complete layer=%d tokens=%u token_tile=%u host_staged=0 boundary_copy_bytes=0 backend=Vulkan\n", p.current_layer, tokens, p.token_tile);
            return;
        }
    }
}

void llama_maple_level4_compute(ggml_tensor * dst, int ith, int nth, void * userdata) {
    // Zero-copy is required for this descriptor; do not silently stage on the CPU.
    GGML_UNUSED(dst); GGML_UNUSED(ith); GGML_UNUSED(nth); GGML_UNUSED(userdata);
    GGML_ABORT("Maple Level4 node was not claimed by Vulkan; zero-copy unavailable");
}
static bool maple_external_execute(ggml_tensor * dst, void * userdata, ggml_vk_external_lease * lease, const ggml_vk_external_api * api) {
    try { std::fprintf(stderr, "maple-level4 Vulkan executor node=%s CPU_callback=0\n", dst->name); maple_compute(dst, userdata, lease, api); return true; }
    catch (const std::exception & e) { std::fprintf(stderr, "Maple zero-copy failed: %s\n", e.what()); return false; }
}
void llama_maple_level4_register() {
    auto reg = ggml_backend_reg_by_name("Vulkan");
    require(reg != nullptr, "Maple zero-copy needs Vulkan backend");
    auto install = reinterpret_cast<ggml_vk_external_register>(ggml_backend_reg_get_proc_address(reg, "ggml_vk_external_register_v1"));
    require(install && install(llama_maple_level4_compute, maple_external_execute, 1), "Vulkan lacks the Level4 external executor ABI");
}
