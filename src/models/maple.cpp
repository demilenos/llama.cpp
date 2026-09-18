#include "models.h"
#include "llama-adapter.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#ifdef LLAMA_MAPLE_LEVEL4
#include "llama-maple-level4.h"
#endif

void llama_model_maple::load_arch_hparams(llama_model_loader & ml) {
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);

    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);

    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP, hparams.swiglu_clamp_exp, hparams.n_layer_all);

    switch (hparams.n_layer()) {
        case 24: type = LLM_TYPE_20B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_maple::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_ff_exp = hparams.n_ff_exp();
    const int64_t head_dim = hparams.n_embd_head_k();

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    if (n_expert == 0) {
        throw std::runtime_error("n_expert must be > 0 for Maple");
    }
    if (n_expert_used == 0) {
        throw std::runtime_error("n_expert_used must be > 0 for Maple");
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_head * head_dim, n_head_kv * head_dim, n_head_kv * head_dim, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * head_dim, n_embd}, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {head_dim}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {head_dim}, 0);
        layer.ffn_norm    = create_tensor(tn(LLM_TENSOR_FFN_NORM,    "weight", i), {n_embd}, 0);

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_maple::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_maple::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_k();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_v());

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv_iswa();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const char * level4_env = std::getenv("LLAMA_MAPLE_LEVEL4");
    const bool level4 = level4_env && std::strcmp(level4_env, "1") == 0;
#ifndef LLAMA_MAPLE_LEVEL4
    if (level4) {
        throw std::runtime_error("LLAMA_MAPLE_LEVEL4 requires a build with -DLLAMA_MAPLE_LEVEL4=ON");
    }
#else
    unsigned token_tile = 4;
    ggml_tensor * island = nullptr;
    if (level4) {
        const char * tile = std::getenv("LLAMA_MAPLE_TOKEN_TILE");
        if (tile) {
            if (!std::strcmp(tile, "1")) token_tile = 1;
            else if (!std::strcmp(tile, "4")) token_tile = 4;
            else if (!std::strcmp(tile, "8")) token_tile = 8;
            else throw std::runtime_error("LLAMA_MAPLE_TOKEN_TILE must be 1, 4 or 8");
        }
        if ((loras && !loras->empty()) || hparams.f_clamp_kqv > 0 || n_tokens > 2048) {
            throw std::runtime_error("Maple Level 4 does not support LoRA, QKV clamp or ubatch > 2048");
        }
        for (int il = 0; il < n_layer; ++il) {
            const auto & l = model.layers[il];
            if ((cvec && cvec->tensor_for(il)) || l.wqkv || l.wq_b || l.wk_b || l.wv_b ||
                l.wo_b || l.wq_s || l.wk_s || l.wv_s || l.wo_s || hparams.swiglu_clamp_exp[il] != 7.0f) {
                throw std::runtime_error("Maple Level 4 requires plain separate QKV/O and clamped SwiGLU without adapters");
            }
            for (auto * w : {l.wq, l.wk, l.wv, l.wo, l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps}) {
                if (!w || w->type != GGML_TYPE_TQ2_0 || !ggml_is_contiguous(w)) {
                    throw std::runtime_error("Maple Level 4 requires contiguous TQ2_0 projection/expert weights");
                }
            }
        }
    }
    if (level4) llama_maple_level4_register();
    auto make_island = [&](ggml_tensor * residual, ggml_tensor * attention, int current, int next) {
        auto data = std::make_shared<llama_maple_level4_params>();
        data->hidden = n_embd;
        data->attention = n_embd_head * n_head;
        data->q_width = n_embd_head * n_head;
        data->kv_width = n_embd_head * n_head_kv;
        data->ffn = hparams.n_ff_exp(current < 0 ? next : current);
        data->experts = n_expert;
        data->topk = n_expert_used;
        data->epsilon = hparams.f_norm_rms_eps;
        data->current_layer = current;
        data->next_layer = next;
        data->token_tile = token_tile;
        if (current >= 0) {
            const auto & l = model.layers[current];
            data->o = l.wo;
            data->ffn_norm = l.ffn_norm;
            data->router = l.ffn_gate_inp;
            data->gate = l.ffn_gate_exps;
            data->up = l.ffn_up_exps;
            data->down = l.ffn_down_exps;
            data->clamp = hparams.swiglu_clamp_exp[current];
        }
        if (next >= 0) {
            const auto & l = model.layers[next];
            data->q = l.wq;
            data->k = l.wk;
            data->v = l.wv;
            data->next_attn_norm = l.attn_norm;
        }
        ggml_tensor * args[] = {ggml_is_contiguous(residual) ? residual : ggml_cont(ctx0, residual),
            attention && !ggml_is_contiguous(attention) ? ggml_cont(ctx0, attention) : attention};
        const int64_t width = n_embd + (next >= 0 ? data->q_width + 2*data->kv_width : 0);
        auto * out = ggml_custom_4d(ctx0, GGML_TYPE_F32, width, residual->ne[1], 1, 1,
                args, attention ? 2 : 1, llama_maple_level4_compute, 1, data.get());
        res->custom_node_data.push_back(data);
        // The registered Vulkan executor owns the Level4 node before scheduling.
        cb(out, next < 0 ? "maple_l4_terminal" : current < 0 ? "maple_l4_bootstrap" : "maple_l4_advance", current);
        return out;
    };
    if (level4) {
        island = make_island(inpL, nullptr, -1, 0);
        inpL = ggml_view_2d(ctx0, island, n_embd, n_tokens, n_embd * sizeof(float), 0);
    }
#endif

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        ggml_tensor * cur = nullptr;
        if (!level4) {
            cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);
        }

        {
            ggml_tensor * Qcur, * Kcur, * Vcur;
#ifdef LLAMA_MAPLE_LEVEL4
            if (level4) {
                const size_t qoff = n_embd * n_tokens * sizeof(float);
                const size_t koff = qoff + n_embd_head * n_head * n_tokens * sizeof(float);
                const size_t voff = koff + n_embd_head * n_head_kv * n_tokens * sizeof(float);
                Qcur = ggml_view_3d(ctx0, island, n_embd_head, n_head, n_tokens,
                        n_embd_head * sizeof(float), n_embd_head * n_head * sizeof(float), qoff);
                Kcur = ggml_view_3d(ctx0, island, n_embd_head, n_head_kv, n_tokens,
                        n_embd_head * sizeof(float), n_embd_head * n_head_kv * sizeof(float), koff);
                Vcur = ggml_view_3d(ctx0, island, n_embd_head, n_head_kv, n_tokens,
                        n_embd_head * sizeof(float), n_embd_head * n_head_kv * sizeof(float), voff);
            } else
#endif
            {
                auto qkv = build_qkv(model.layers[il], cur, n_embd_head, n_head, n_head_kv, il);
                Qcur = qkv.q; Kcur = qkv.k; Vcur = qkv.v;
            }

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);
            cb(Kcur, "Kcur_normed", il);

            if (hparams.is_swa(il)) {
                const int64_t n_rot_l = hparams.n_rot(il);
                const float freq_base_l = model.get_rope_freq_base(cparams, il);
                const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot_l, rope_type, n_ctx_orig, freq_base_l,
                                     freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot_l, rope_type, n_ctx_orig, freq_base_l,
                                     freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
            }
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    level4 ? nullptr : model.layers[il].wo, nullptr, level4 ? nullptr : model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
            cb(cur, "attn_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

#ifdef LLAMA_MAPLE_LEVEL4
        if (level4) {
            const int next = il + 1 < n_layer ? il + 1 : -1;
            island = make_island(inpSA, cur, il, next);
            inpL = ggml_view_2d(ctx0, island, n_embd, island->ne[1], n_embd * sizeof(float), 0);
            cb(inpL, "l_out", il);
            continue;
        }
#endif
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true,
                1.0f,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                il);
        cb(cur, "ffn_moe_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
