// Included after Vulkan backend types and submit helpers.
struct ggml_vk_external_lease {
    ggml_backend_vk_context * ctx;
    std::unique_lock<std::recursive_mutex> guard;
    std::vector<vk_buffer> buffers;
    // Legacy host-fenced path allocates these lazily. PTQ1 async handoff never
    // creates a per-op command pool or fence.
    vk::CommandPool pool {};
    vk::Fence fence {};
    bool released = false;
    bool pending = false;
    uint64_t external_wait_value = 0;
    uint64_t external_signal_value = 0;

    explicit ggml_vk_external_lease(ggml_backend_vk_context * c)
        : ctx(c), guard(c->device->compute_queue->handle->external_mutex) {}

    ~ggml_vk_external_lease() {
        if (pending) {
            ctx->device->external_poisoned = true;
            auto * retained = new std::vector<vk_buffer>(std::move(buffers));
            (void) retained;
            return;
        }
        if (fence) ctx->device->device.destroyFence(fence);
        if (pool) ctx->device->device.destroyCommandPool(pool);
    }
};

static bool ggml_vk_external_get_span(
        ggml_vk_external_lease * l,
        const ggml_tensor * tensor,
        ggml_vk_external_span * span) {
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
    if (!DuplicateHandle(GetCurrentProcess(), b->exported_handle, GetCurrentProcess(),
                         &handle, 0, FALSE, DUPLICATE_SAME_ACCESS)) return false;

    *span = {};
    span->allocation_id = b->external_allocation_id;
    span->generation = b->external_allocation_id;
    span->allocation_size = b->memory_allocation_size;
    span->tensor_offset = offset;
    span->bytes = bytes;
    span->memory_handle = handle;
    span->owner = b;

    auto ids = l->ctx->device->physical_device.getProperties2<
        vk::PhysicalDeviceProperties2, vk::PhysicalDeviceIDProperties>();
    const auto & id = ids.get<vk::PhysicalDeviceIDProperties>();
    const auto & p = ids.get<vk::PhysicalDeviceProperties2>().properties;
    span->vendor = p.vendorID;
    span->device = p.deviceID;
    span->node_mask = id.deviceNodeMask;
    span->luid_valid = id.deviceLUIDValid;
    memcpy(span->luid, id.deviceLUID.data(), 8);
    memcpy(span->device_uuid, id.deviceUUID.data(), 16);

    if (std::find(l->buffers.begin(), l->buffers.end(), b) == l->buffers.end()) {
        l->buffers.push_back(std::move(b));
    }
    return true;
#else
    GGML_UNUSED(l); GGML_UNUSED(tensor); GGML_UNUSED(span); return false;
#endif
}

static void ggml_vk_external_ensure_legacy_sync(ggml_vk_external_lease * l) {
    if (!l->pool) {
        l->pool = l->ctx->device->device.createCommandPool({
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            l->ctx->device->compute_queue->queue_family_index});
    }
    if (!l->fence) {
        l->fence = l->ctx->device->device.createFence({});
    }
}

// ABI v1 host-fenced fallback. Kept for the generic Level4 bridge.
static bool ggml_vk_external_barrier(ggml_vk_external_lease * l, bool release) {
    try {
        if (l->pending || l->buffers.empty() || l->released == release) return false;
        ggml_vk_external_ensure_legacy_sync(l);

        auto device = l->ctx->device;
        auto cmd = device->device.allocateCommandBuffers({
            l->pool, vk::CommandBufferLevel::ePrimary, 1}).front();
        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        std::vector<vk::BufferMemoryBarrier> barriers;
        barriers.reserve(l->buffers.size());
        for (auto & b : l->buffers) {
            barriers.emplace_back(
                release ? vk::AccessFlagBits::eMemoryWrite : vk::AccessFlags{},
                release ? vk::AccessFlags{} :
                    vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
                release ? device->compute_queue->queue_family_index : VK_QUEUE_FAMILY_EXTERNAL,
                release ? VK_QUEUE_FAMILY_EXTERNAL : device->compute_queue->queue_family_index,
                b->buffer, 0, VK_WHOLE_SIZE);
        }
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eAllCommands,
            vk::PipelineStageFlagBits::eAllCommands,
            {}, {}, barriers, {});
        cmd.end();

        device->device.resetFences(l->fence);
        vk::SubmitInfo submit;
        submit.setCommandBuffers(cmd);
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

static bool ggml_vk_external_release(ggml_vk_external_lease * l) {
    return ggml_vk_external_barrier(l, true);
}

static bool ggml_vk_external_timeline_init(ggml_backend_vk_context * ctx) {
#if defined(_WIN32)
    auto & t = ctx->ptq1_external_timeline;
    if (t.semaphore && t.exported_handle) return true;
    if (!ctx->device->external_semaphore_win32) return false;

    try {
        vk::ExportSemaphoreCreateInfo export_info{
            vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32};
        vk::SemaphoreTypeCreateInfo type_info{vk::SemaphoreType::eTimeline, 0};
        type_info.setPNext(&export_info);
        vk::SemaphoreCreateInfo create_info{};
        create_info.setPNext(&type_info);
        t.semaphore = ctx->device->device.createSemaphore(create_info);

        auto get_handle = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
            vkGetDeviceProcAddr(static_cast<VkDevice>(ctx->device->device),
                                "vkGetSemaphoreWin32HandleKHR"));
        if (!get_handle) throw std::runtime_error("vkGetSemaphoreWin32HandleKHR unavailable");

        VkSemaphoreGetWin32HandleInfoKHR info{
            VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR,
            nullptr,
            static_cast<VkSemaphore>(t.semaphore),
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT};
        HANDLE handle = nullptr;
        if (get_handle(static_cast<VkDevice>(ctx->device->device), &info, &handle) != VK_SUCCESS ||
            !handle || handle == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Vulkan timeline semaphore Win32 export failed");
        }
        t.exported_handle = handle;
        t.value = 0;
        return true;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml_vulkan: external timeline init failed: %s\n", e.what());
        if (t.exported_handle) {
            CloseHandle(t.exported_handle);
            t.exported_handle = nullptr;
        }
        if (t.semaphore) {
            ctx->device->device.destroySemaphore(t.semaphore);
            t.semaphore = {};
        }
        return false;
    }
#else
    GGML_UNUSED(ctx);
    return false;
#endif
}

static bool ggml_vk_external_submit_async(
        ggml_vk_external_lease * l,
        bool release,
        uint64_t wait_value,
        uint64_t signal_value) {
    try {
        if (!l || l->buffers.empty()) return false;
        auto device = l->ctx->device;
        auto & timeline = l->ctx->ptq1_external_timeline;
        if (!timeline.semaphore) return false;

        vk_context sync_ctx = ggml_vk_create_context(l->ctx, l->ctx->compute_cmd_pool);
        ggml_vk_ctx_begin(device, sync_ctx);

        std::vector<vk::BufferMemoryBarrier> barriers;
        barriers.reserve(l->buffers.size());
        for (auto & b : l->buffers) {
            barriers.emplace_back(
                release ? vk::AccessFlagBits::eMemoryWrite : vk::AccessFlags{},
                release ? vk::AccessFlags{} :
                    vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
                release ? device->compute_queue->queue_family_index : VK_QUEUE_FAMILY_EXTERNAL,
                release ? VK_QUEUE_FAMILY_EXTERNAL : device->compute_queue->queue_family_index,
                b->buffer, 0, VK_WHOLE_SIZE);
        }

        if (!release) {
            sync_ctx->s->wait_semaphores.push_back({timeline.semaphore, wait_value});
        }

        sync_ctx->s->buffer->buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eAllCommands,
            vk::PipelineStageFlagBits::eAllCommands,
            {}, {}, barriers, {});

        if (release) {
            sync_ctx->s->signal_semaphores.push_back({timeline.semaphore, signal_value});
        }

        ggml_vk_ctx_end(sync_ctx);
        ggml_vk_submit(sync_ctx, {});
        l->ctx->submit_pending = true;
        return true;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml_vulkan: async external barrier failed: %s\n", e.what());
        l->ctx->device->external_poisoned = true;
        return false;
    }
}

static bool ggml_vk_external_release_async(
        ggml_vk_external_lease * l,
        ggml_vk_external_timeline * out) {
#if defined(_WIN32)
    if (!l || !out || l->released || l->buffers.empty()) return false;
    if (!ggml_vk_external_timeline_init(l->ctx)) return false;

    auto & t = l->ctx->ptq1_external_timeline;
    const uint64_t release_value = ++t.value;
    const uint64_t complete_value = ++t.value;

    if (!ggml_vk_external_submit_async(l, true, 0, release_value)) return false;

    l->released = true;
    l->external_wait_value = release_value;
    l->external_signal_value = complete_value;
    out->semaphore_handle = t.exported_handle;
    out->wait_value = release_value;
    out->signal_value = complete_value;
    return true;
#else
    GGML_UNUSED(l); GGML_UNUSED(out); return false;
#endif
}

static bool ggml_vk_external_acquire_async(ggml_vk_external_lease * l) {
    if (!l || !l->released || !l->external_signal_value) return false;
    if (!ggml_vk_external_submit_async(
            l, false, l->external_signal_value, 0)) return false;
    l->released = false;
    return true;
}

static void ggml_vk_external_timeline_cleanup(ggml_backend_vk_context * ctx) {
#if defined(_WIN32)
    if (!ctx) return;
    auto & t = ctx->ptq1_external_timeline;
    if (t.exported_handle) {
        CloseHandle(t.exported_handle);
        t.exported_handle = nullptr;
    }
    if (t.semaphore) {
        ctx->device->device.destroySemaphore(t.semaphore);
        t.semaphore = {};
    }
    t.value = 0;
#else
    GGML_UNUSED(ctx);
#endif
}

static bool ggml_vk_external_compute(ggml_backend_vk_context * ctx, ggml_tensor * node) {
    try {
        ggml_vk_external_lease lease(ctx);
        static const ggml_vk_external_api api{
            1, ggml_vk_external_get_span, ggml_vk_external_release, nullptr};
        ggml_custom_op_params p;
        memcpy(&p, node->op_params, sizeof(p));
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

static bool ggml_vk_ptq1_external_compute(ggml_backend_vk_context * ctx, ggml_tensor * node) {
    try {
        const auto executor = vk_ptq1_executor.load();
        if (!executor) return false;

        ggml_vk_external_lease lease(ctx);
        static const ggml_vk_external_api api{
            2, ggml_vk_external_get_span, ggml_vk_external_release,
            ggml_vk_external_release_async};

        const bool ok = executor(node, &lease, &api);
        if (!ok || !lease.released || !lease.external_signal_value) {
            ctx->device->external_poisoned = true;
            return false;
        }
        if (!ggml_vk_external_acquire_async(&lease)) {
            ctx->device->external_poisoned = true;
            return false;
        }
        return true;
    } catch (const std::exception & e) {
        ctx->device->external_poisoned = true;
        GGML_LOG_ERROR("ggml_vulkan: PTQ1 XMX async external execution failed: %s\n", e.what());
        return false;
    }
}
