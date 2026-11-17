#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <vector>
#include <string>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <cctype>

class CogniMoETelemetry {
public:
    static CogniMoETelemetry& instance() {
        static CogniMoETelemetry inst;
        return inst;
    }

    void init(int n_layers, int n_experts) {
        std::lock_guard<std::mutex> lock(mtx_);
        n_layers_ = n_layers;
        n_experts_ = n_experts;
        counts_.assign(n_layers, std::vector<uint64_t>(n_experts, 0));
        vram_resident_.assign(n_layers, std::vector<bool>(n_experts, false));
        bias_tensors_.assign(n_layers, nullptr);
        bias_uploaded_.assign(n_layers, false);
        has_vram_map_ = false;
        enabled_ = true;

        record_topk_ = (std::getenv("COGNIMOE_TELEMETRY_OUT") != nullptr);

        if (const char * pen_str = std::getenv("COGNIMOE_CACHE_MISS_PENALTY")) {
            miss_penalty_ = std::max(0.0f, (float) std::atof(pen_str));
        } else {
            miss_penalty_ = 0.0f;
        }
    }

    void set_vram_resident(int layer_idx, int expert_idx, bool resident = true) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (layer_idx >= 0 && layer_idx < n_layers_ && expert_idx >= 0 && expert_idx < n_experts_) {
            vram_resident_[layer_idx][expert_idx] = resident;
            has_vram_map_ = true;
            bias_uploaded_[layer_idx] = false;
        }
    }

    void register_bias_tensor(int layer_idx, struct ggml_tensor * t) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (layer_idx >= 0 && layer_idx < n_layers_) {
            bias_tensors_[layer_idx] = t;
            bias_uploaded_[layer_idx] = false;
        }
    }

    bool is_soft_mask_active() const {
        return enabled_ && has_vram_map_ && miss_penalty_ > 0.0f;
    }

    float get_miss_penalty() const {
        return miss_penalty_;
    }

    std::vector<float> get_layer_bias(int layer_idx) const {
        std::vector<float> bias(n_experts_, 0.0f);
        if (!is_soft_mask_active() || layer_idx < 0 || layer_idx >= n_layers_) {
            return bias;
        }
        for (int e = 0; e < n_experts_; ++e) {
            if (!vram_resident_[layer_idx][e]) {
                bias[e] = -miss_penalty_;
            }
        }
        return bias;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mtx_);
        for (auto& layer : counts_) {
            std::fill(layer.begin(), layer.end(), 0);
        }
    }

    static bool eval_callback(struct ggml_tensor * t, bool ask, void * /*user_data*/) {
        auto& self = instance();
        if (!self.enabled_) return true;

        // 1. Popola il tensore di bias in VRAM una sola volta per layer (restituendo SEMPRE true per non fermare il grafo)
        if (self.is_soft_mask_active() && std::strncmp(t->name, "ffn_moe_logits-", 15) == 0) {
            int layer_idx = std::atoi(t->name + 15);
            if (layer_idx >= 0 && layer_idx < self.n_layers_ && !self.bias_uploaded_[layer_idx]) {
                if (ask) {
                    return true;
                }
                struct ggml_tensor * bias_t = self.bias_tensors_[layer_idx];
                if (bias_t != nullptr && bias_t->buffer != nullptr) {
                    auto bias = self.get_layer_bias(layer_idx);
                    ggml_backend_tensor_set(bias_t, bias.data(), 0, bias.size() * sizeof(float));
                    self.bias_uploaded_[layer_idx] = true;
                }
            }
            return true;
        }

        // 2. Telemetria: registra gli indici Top-K solo se richiesta esportazione JSON (zero overhead in produzione)
        if (self.record_topk_ && std::strncmp(t->name, "ffn_moe_topk-", 13) == 0) {
            if (ask) {
                return true;
            }
            int layer_idx = std::atoi(t->name + 13);
            if (layer_idx >= 0 && layer_idx < self.n_layers_) {
                size_t n_elems = ggml_nelements(t);
                std::vector<int32_t> host_ids(n_elems);
                ggml_backend_tensor_get(t, host_ids.data(), 0, n_elems * sizeof(int32_t));

                std::lock_guard<std::mutex> lock(self.mtx_);
                for (int32_t exp_id : host_ids) {
                    if (exp_id >= 0 && exp_id < self.n_experts_) {
                        self.counts_[layer_idx][exp_id]++;
                    }
                }
            }
        }
        return true;
    }

    void export_json(const std::string& filepath) {
        std::lock_guard<std::mutex> lock(mtx_);
        std::ofstream out(filepath);
        if (!out.is_open()) return;

        out << "{\n  \"n_layers\": " << n_layers_ << ",\n  \"n_experts\": " << n_experts_ << ",\n  \"activations\": [\n";
        for (int l = 0; l < n_layers_; ++l) {
            out << "    [";
            for (int e = 0; e < n_experts_; ++e) {
                out << counts_[l][e] << (e + 1 < n_experts_ ? ", " : "");
            }
            out << "]" << (l + 1 < n_layers_ ? ",\n" : "\n");
        }
        out << "  ]\n}\n";
    }

private:
    CogniMoETelemetry() = default;
    int n_layers_ = 0;
    int n_experts_ = 0;
    bool enabled_ = false;
    bool record_topk_ = false;
    bool has_vram_map_ = false;
    float miss_penalty_ = 0.0f;
    std::vector<std::vector<uint64_t>> counts_;
    std::vector<std::vector<bool>> vram_resident_;
    std::vector<struct ggml_tensor *> bias_tensors_;
    std::vector<bool> bias_uploaded_;
    std::mutex mtx_;
};

inline std::vector<std::vector<int32_t>> cognimoe_load_top_experts_from_json(const std::string& filepath, int n_layers, int n_experts, int k_per_layer) {
    std::vector<std::vector<int32_t>> result(n_layers);
    std::ifstream in(filepath);
    if (!in.is_open()) return {};

    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    size_t pos = content.find("\"activations\"");
    if (pos == std::string::npos) return {};
    pos = content.find('[', pos);
    if (pos == std::string::npos) return {};
    pos++;

    for (int l = 0; l < n_layers; ++l) {
        size_t l_start = content.find('[', pos);
        size_t l_end = content.find(']', l_start);
        if (l_start == std::string::npos || l_end == std::string::npos) break;

        std::vector<uint64_t> counts;
        counts.reserve(n_experts);
        size_t cur = l_start + 1;
        while (cur < l_end) {
            while (cur < l_end && !std::isdigit(static_cast<unsigned char>(content[cur]))) cur++;
            if (cur >= l_end) break;
            uint64_t val = 0;
            while (cur < l_end && std::isdigit(static_cast<unsigned char>(content[cur]))) {
                val = val * 10 + (content[cur] - '0');
                cur++;
            }
            counts.push_back(val);
        }

        if ((int)counts.size() == n_experts) {
            std::vector<int32_t> indices(n_experts);
            std::iota(indices.begin(), indices.end(), 0);
            int actual_k = std::min(k_per_layer, n_experts);
            std::partial_sort(indices.begin(), indices.begin() + actual_k, indices.end(),
                [&](int32_t a, int32_t b) { return counts[a] > counts[b]; });
            indices.resize(actual_k);
            result[l] = std::move(indices);
        }
        pos = l_end + 1;
    }
    return result;
}
