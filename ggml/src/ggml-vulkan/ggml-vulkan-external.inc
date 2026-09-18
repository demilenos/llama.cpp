// Included after Vulkan backend types and submit helpers.
struct ggml_vk_external_lease {
    ggml_backend_vk_context * ctx;
    std::unique_lock<std::recursive_mutex> guard;
    std::vector<vk_buffer> buffers;
    vk::CommandPool pool;
    vk::Fence fence;
    bool released = false;
    bool pending = false;
    explicit ggml_vk_external_lease(ggml_backend_vk_context * c) : ctx(c), guard(c->device->compute_queue->handle->external_mutex) {
        pool = c->device->device.createCommandPool({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, c->device->compute_queue->queue_family_index});
        fence = c->device->device.createFence({});
    }
    ~ggml_vk_external_lease() {
        if (pending) {
            ctx->device->external_poisoned = true;
            // Retain native resources when completion cannot be proven.
            auto * retained = new std::vector<vk_buffer>(std::move(buffers));
            (void) retained;
            return;
        }
        ctx->device->device.destroyFence(fence);
        ctx->device->device.destroyCommandPool(pool);
    }
};
static bool ggml_vk_external_get_span(ggml_vk_external_lease * l, const ggml_tensor * tensor, ggml_vk_external_span * span) {
#if defined(_WIN32)
    if (!l || l->released || !tensor || !tensor->buffer || !span || !ggml_backend_buffer_is_vk(tensor->buffer)) return false;
    auto * bc = static_cast<ggml_backend_vk_buffer_context *>(tensor->buffer->context);
    auto b = bc->dev_buffer;
    if (!b || b->device != l->ctx->device || !b->external_exportable) return false;
    size_t offset = vk_tensor_offset(tensor);
    if (tensor->view_offs > SIZE_MAX - offset) return false;
    offset += tensor->view_offs;
    const size_t bytes = ggml_nbytes(tensor);
    if (offset > b->size || bytes > b->size - offset) return false;
    HANDLE handle = nullptr;
    if (!DuplicateHandle(GetCurrentProcess(), b->exported_handle, GetCurrentProcess(), &handle, 0, FALSE, DUPLICATE_SAME_ACCESS)) return false;
    *span = {};
    span->allocation_id = b->external_allocation_id;
    span->generation = b->external_allocation_id;
    span->allocation_size = b->memory_allocation_size;
    span->tensor_offset = offset;
    span->bytes = bytes;
    span->memory_handle = handle;
    span->owner = b;
    auto ids = l->ctx->device->physical_device.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceIDProperties>();
    const auto & id = ids.get<vk::PhysicalDeviceIDProperties>();
    const auto & p = ids.get<vk::PhysicalDeviceProperties2>().properties;
    span->vendor = p.vendorID; span->device = p.deviceID;
    span->node_mask = id.deviceNodeMask; span->luid_valid = id.deviceLUIDValid;
    memcpy(span->luid, id.deviceLUID.data(), 8); memcpy(span->device_uuid, id.deviceUUID.data(), 16);
    if (std::find(l->buffers.begin(), l->buffers.end(), b) == l->buffers.end()) l->buffers.push_back(std::move(b));
    return true;
#else
    GGML_UNUSED(l); GGML_UNUSED(tensor); GGML_UNUSED(span); return false;
#endif
}
static bool ggml_vk_external_barrier(ggml_vk_external_lease * l, bool release) {
    try {
        if (l->pending || l->buffers.empty() || l->released == release) return false;
        auto device = l->ctx->device;
        auto cmd = device->device.allocateCommandBuffers({l->pool, vk::CommandBufferLevel::ePrimary, 1}).front();
        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        std::vector<vk::BufferMemoryBarrier> barriers;
        for (auto & b : l->buffers) {
            barriers.emplace_back(release ? vk::AccessFlagBits::eMemoryWrite : vk::AccessFlags{},
                release ? vk::AccessFlags{} : vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
                release ? device->compute_queue->queue_family_index : VK_QUEUE_FAMILY_EXTERNAL,
                release ? VK_QUEUE_FAMILY_EXTERNAL : device->compute_queue->queue_family_index,
                b->buffer, 0, VK_WHOLE_SIZE);
        }
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands, {}, {}, barriers, {});
        cmd.end();
        device->device.resetFences(l->fence);
        vk::SubmitInfo submit; submit.setCommandBuffers(cmd);
        l->pending = true;
        device->compute_queue->handle->submit(submit, l->fence);
        if (device->device.waitForFences(l->fence, true, UINT64_MAX) != vk::Result::eSuccess) return false;
        l->pending = false;
        l->released = release;
        return true;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml_vulkan: external barrier failed: %s\n", e.what());
        l->ctx->device->external_poisoned = true;
        return false;
    }
}
static bool ggml_vk_external_release(ggml_vk_external_lease * l) { return ggml_vk_external_barrier(l, true); }
static bool ggml_vk_external_compute(ggml_backend_vk_context * ctx, ggml_tensor * node) {
    try {
        ggml_vk_external_lease lease(ctx);
        static const ggml_vk_external_api api{1, ggml_vk_external_get_span, ggml_vk_external_release};
        ggml_custom_op_params p; memcpy(&p, node->op_params, sizeof(p));
        const bool ok = vk_external_executor.load()(node, p.userdata, &lease, &api);
        if (!ok || !lease.released) {
            if (lease.released || lease.pending) ctx->device->external_poisoned = true;
            return false;
        }
        if (!ggml_vk_external_barrier(&lease, false)) {
            ctx->device->external_poisoned = true;
            return false;
        }
        return true;
    } catch (const std::exception & e) {
        ctx->device->external_poisoned = true;
        GGML_LOG_ERROR("ggml_vulkan: external execution failed: %s\n", e.what());
        return false;
    }
}
