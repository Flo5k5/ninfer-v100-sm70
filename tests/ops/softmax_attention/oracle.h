#pragma once

#include "ninfer/ops/attention_geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ninfer::test {

// Independent logical Softmax Attention oracle. Callbacks expose represented public values and
// the entry-specific visible set; no production staging cast, tile, cache address, or reduction
// tree is reproduced here.
template <typename QueryValue, typename KeyValue, typename ValueValue, typename Visible,
          typename Store>
void naive_dense_softmax_attention(ops::AttentionHeadGeometry geometry, int query_tokens,
                                   int key_tokens, double scale, QueryValue query_value,
                                   KeyValue key_value, ValueValue value_value, Visible visible,
                                   Store store) {
    if (!ops::valid_attention_head_geometry(geometry) || query_tokens < 0 || key_tokens < 0) {
        throw std::invalid_argument("invalid naive Softmax Attention geometry");
    }
    const int group = geometry.query_heads / geometry.kv_heads;
    // Each query row is independent and every callback writes or reads distinct elements, so large
    // problems split their rows across threads; every row keeps its sequential arithmetic.
    const auto rows = [&](int first_query, int last_query) {
        std::vector<double> scores(static_cast<std::size_t>(key_tokens));
        std::vector<double> numerators(static_cast<std::size_t>(geometry.head_dim));
        for (int query = first_query; query < last_query; ++query) {
            for (int query_head = 0; query_head < geometry.query_heads; ++query_head) {
                const int kv_head = query_head / group;
                double maximum    = -std::numeric_limits<double>::infinity();
                for (int key = 0; key < key_tokens; ++key) {
                    if (!visible(query, key)) {
                        scores[static_cast<std::size_t>(key)] =
                            -std::numeric_limits<double>::infinity();
                        continue;
                    }
                    double dot = 0.0;
                    for (int d = 0; d < geometry.head_dim; ++d) {
                        dot += query_value(d, query_head, query) * key_value(d, kv_head, key);
                    }
                    const double score                    = dot * scale;
                    scores[static_cast<std::size_t>(key)] = score;
                    maximum                               = std::max(maximum, score);
                }

                double denominator = 0.0;
                if (maximum != -std::numeric_limits<double>::infinity()) {
                    for (int key = 0; key < key_tokens; ++key) {
                        double& score = scores[static_cast<std::size_t>(key)];
                        if (score == -std::numeric_limits<double>::infinity()) continue;
                        score = std::exp(score - maximum);
                        denominator += score;
                    }
                }
                // Keys outer, dimensions inner: every numerator still sums its keys in ascending
                // order, and the value rows are read contiguously.
                std::fill(numerators.begin(), numerators.end(), 0.0);
                for (int key = 0; key < key_tokens; ++key) {
                    const double weight = scores[static_cast<std::size_t>(key)];
                    if (weight == -std::numeric_limits<double>::infinity()) continue;
                    for (int d = 0; d < geometry.head_dim; ++d) {
                        numerators[static_cast<std::size_t>(d)] +=
                            weight * value_value(d, kv_head, key);
                    }
                }
                for (int d = 0; d < geometry.head_dim; ++d) {
                    const double numerator = numerators[static_cast<std::size_t>(d)];
                    store(d, query_head, query,
                          denominator > 0.0 ? numerator / denominator : 0.0);
                }
            }
        }
    };

    constexpr long long kParallelWork = 1LL << 22; // query rows x heads x keys
    const long long work = static_cast<long long>(query_tokens) * geometry.query_heads * key_tokens;
    const int threads    = static_cast<int>(std::min<unsigned>(
        {std::max(1U, std::thread::hardware_concurrency()), 16U,
         static_cast<unsigned>(std::max(query_tokens, 1))}));
    if (threads <= 1 || work < kParallelWork) {
        rows(0, query_tokens);
        return;
    }
    const int per_thread = (query_tokens + threads - 1) / threads;
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(threads));
    for (int first = 0; first < query_tokens; first += per_thread) {
        pool.emplace_back(rows, first, std::min(query_tokens, first + per_thread));
    }
    for (std::thread& thread : pool) { thread.join(); }
}

} // namespace ninfer::test
