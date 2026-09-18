// PTQ1 XMX external-memory lease. Included after Vulkan backend types/helpers.
struct ggml_vk_external_lease {
    ggml_backend_vk_context * ctx;
    std::unique_lock<std::recursive_mutex> guard;
    std::vector<vk_buffer> buffers;
    vk::CommandPool pool;
    vk::Fence fence;
    bool released = false;
    bool pending = false;

    explicit ggml_vk_external_lease(ggml_backend_vk_context * c)
        : ctx(c), guard(c->device->compute_queue->handle->external_mutex) {
        pool = c->device->device.createCommandPool(
            {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, c->device->compute_queue->queue_family_index});
        fence = c->device->device.createFence({});
    }

    ~ggml_vk_external_lease() {
        if (pending) {
            ctx->device->external_poisoned = true;
            // Completion cannot be proven; intentionally retain the native resources.
            auto * retained = new std::vector<vk_buffer>(std::move(buffers));
            (void) retained;
            return;
        }
        ctx->device->device.destroyFence(fence);
        ctx->device->device.destroyCommandPool(pool);
    }
};

static bool ggml_vk_external_get_span(
    ggml_vk_external_lease * lease,
    const ggml_tensor * tensor,
    ggml_vk_external_span * span) {
#if defined(_WIN32)
    if (!lease || lease->released || !tensor || !tensor->buffer || !span ||
        !ggml_backend_buffer_is_vk(tensor->buffer)) {
        return false;
    }

    auto * buffer_ctx = static_cast<ggml_backend_vk_buffer_context *>(tensor->buffer->context);
    auto buffer = buffer_ctx->dev_buffer;
    if (!buffer || buffer->device != lease->ctx->device || !buffer->external_exportable) {
        return false;
    }

    size_t offset = vk_tensor_offset(tensor);
    if (tensor->view_offs > SIZE_MAX - offset) {
        return false;
    }
    offset += tensor->view_offs;

    const size_t bytes = ggml_nbytes(tensor);
    if (offset > buffer->size || bytes > buffer->size - offset) {
        return false;
    }

    HANDLE handle = nullptr;
    if (!DuplicateHandle(
            GetCurrentProcess(),
            buffer->exported_handle,
            GetCurrentProcess(),
            &handle,
            0,
            FALSE,
            DUPLICATE_SAME_ACCESS)) {
        return false;
    }

    *span = {};
    span->allocation_id = buffer->external_allocation_id;
    span->generation = buffer->external_allocation_id;
    span->allocation_size = buffer->memory_allocation_size;
    span->tensor_offset = offset;
    span->bytes = bytes;
    span->memory_handle = handle;

    auto ids = lease->ctx->device->physical_device
                   .getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceIDProperties>();
    const auto & id = ids.get<vk::PhysicalDeviceIDProperties>();
    const auto & properties = ids.get<vk::PhysicalDeviceProperties2>().properties;

    span->vendor = properties.vendorID;
    span->device = properties.deviceID;
    span->node_mask = id.deviceNodeMask;
    span->luid_valid = id.deviceLUIDValid;
    std::memcpy(span->luid, id.deviceLUID.data(), 8);
    std::memcpy(span->device_uuid, id.deviceUUID.data(), 16);

    if (std::find(lease->buffers.begin(), lease->buffers.end(), buffer) == lease->buffers.end()) {
        lease->buffers.push_back(std::move(buffer));
    }
    return true;
#else
    GGML_UNUSED(lease);
    GGML_UNUSED(tensor);
    GGML_UNUSED(span);
    return false;
#endif
}

static bool ggml_vk_external_barrier(ggml_vk_external_lease * lease, bool release) {
    try {
        if (!lease || lease->pending || lease->buffers.empty() || lease->released == release) {
            return false;
        }

        auto device = lease->ctx->device;
        auto command = device->device
                           .allocateCommandBuffers(
                               {lease->pool, vk::CommandBufferLevel::ePrimary, 1})
                           .front();

        command.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        std::vector<vk::BufferMemoryBarrier> barriers;
        barriers.reserve(lease->buffers.size());
        for (const auto & buffer : lease->buffers) {
            barriers.emplace_back(
                release ? vk::AccessFlagBits::eMemoryWrite : vk::AccessFlags{},
                release ? vk::AccessFlags{} :
                          vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
                release ? device->compute_queue->queue_family_index : VK_QUEUE_FAMILY_EXTERNAL,
                release ? VK_QUEUE_FAMILY_EXTERNAL : device->compute_queue->queue_family_index,
                buffer->buffer,
                0,
                VK_WHOLE_SIZE);
        }

        command.pipelineBarrier(
            vk::PipelineStageFlagBits::eAllCommands,
            vk::PipelineStageFlagBits::eAllCommands,
            {},
            {},
            barriers,
            {});
        command.end();

        device->device.resetFences(lease->fence);
        vk::SubmitInfo submit;
        submit.setCommandBuffers(command);

        lease->pending = true;
        device->compute_queue->handle->submit(submit, lease->fence);
        if (device->device.waitForFences(lease->fence, true, UINT64_MAX) != vk::Result::eSuccess) {
            return false;
        }

        lease->pending = false;
        lease->released = release;
        return true;
    } catch (const std::exception & error) {
        GGML_LOG_ERROR("ggml_vulkan: PTQ1 external barrier failed: %s\n", error.what());
        lease->ctx->device->external_poisoned = true;
        return false;
    }
}

static bool ggml_vk_external_release(ggml_vk_external_lease * lease) {
    return ggml_vk_external_barrier(lease, true);
}

static bool ggml_vk_ptq1_external_compute(ggml_backend_vk_context * ctx, ggml_tensor * node) {
    try {
        ggml_vk_external_lease lease(ctx);
        static const ggml_vk_external_api api{
            1,
            ggml_vk_external_get_span,
            ggml_vk_external_release,
        };

        const bool ok = maple_ptq1_xmx_execute(node, &lease, &api);
        if (!ok || !lease.released) {
            if (lease.released || lease.pending) {
                ctx->device->external_poisoned = true;
            }
            return false;
        }

        if (!ggml_vk_external_barrier(&lease, false)) {
            ctx->device->external_poisoned = true;
            return false;
        }

        return true;
    } catch (const std::exception & error) {
        ctx->device->external_poisoned = true;
        GGML_LOG_ERROR("ggml_vulkan: PTQ1 XMX external execution failed: %s\n", error.what());
        return false;
    }
}
