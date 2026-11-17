#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <vector>
#include <string>
#include <fstream>
#include <cstring>
#include <mutex>

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
        enabled_ = true;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mtx_);
        for (auto& layer : counts_) {
            std::fill(layer.begin(), layer.end(), 0);
        }
    }

    // Callback invocata dal grafo GGML ad ogni nodo valutato
    static bool eval_callback(struct ggml_tensor * t, bool ask, void * /*user_data*/) {
        auto& self = instance();
        if (!self.enabled_) return true;

        // Filtra solo il tensore degli indici Top-K degli esperti
        // In llama.cpp il nome del tensore è formatato come "ffn_moe_topk-<layer>"
        if (std::strncmp(t->name, "ffn_moe_topk-", 13) == 0) {
            if (ask) {
                return true; // Chiedi al backend di eseguire la callback dopo il calcolo
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

    // Esporta la matrice di attivazione [40 x 256] su file JSON per l'agente RL e la GUI
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
    std::vector<std::vector<uint64_t>> counts_;
    std::mutex mtx_;
};

// Parser leggero e senza dipendenze esterne per caricare la matrice [n_layers x n_experts] dal JSON
#include <algorithm>
#include <numeric>
#include <cctype>

inline std::vector<std::vector<int32_t>> cognimoe_load_top_experts_from_json(const std::string& filepath, int n_layers, int n_experts, int k_per_layer) {
    std::vector<std::vector<int32_t>> result(n_layers);
    std::ifstream in(filepath);
    if (!in.is_open()) return {};

    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    size_t pos = content.find("\"activations\"");
    if (pos == std::string::npos) return {};
    pos = content.find('[', pos);
    if (pos == std::string::npos) return {};
    pos++; // Entra nell'array esterno

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
