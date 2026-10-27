#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {
constexpr int64_t kWidth = 2560;

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "layer_major_kernel_test: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;

    constexpr int64_t rows = 3;
    std::vector<float> shared((size_t) (2 * kWidth));
    std::vector<float> sg{0.0f, 1.0f};
    std::vector<float> dm((size_t) (rows * kWidth));
    for (int64_t i = 0; i < 2 * kWidth; ++i) shared[(size_t) i] = (float) ((int) (i % 29) - 14) / 17.0f;
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t d = 0; d < kWidth; ++d) dm[(size_t) (r * kWidth + d)] = (float) (r + 1) + (float) (d % 7) / 11.0f;

    float *d_shared = nullptr, *d_sg = nullptr, *d_acc = nullptr, *d_dm = nullptr;
    int32_t *d_tok = nullptr, *d_off = nullptr, *d_rows = nullptr;
    float* d_w = nullptr;
    bool ok = ck(cudaMalloc(&d_shared, shared.size() * sizeof(float)), "alloc shared") &&
              ck(cudaMalloc(&d_sg, sg.size() * sizeof(float)), "alloc gate") &&
              ck(cudaMalloc(&d_acc, shared.size() * sizeof(float)), "alloc accumulator") &&
              ck(cudaMalloc(&d_dm, dm.size() * sizeof(float)), "alloc expert rows") &&
              ck(cudaMalloc(&d_tok, 2 * sizeof(int32_t)), "alloc token map") &&
              ck(cudaMalloc(&d_off, 3 * sizeof(int32_t)), "alloc offsets") &&
              ck(cudaMalloc(&d_rows, 3 * sizeof(int32_t)), "alloc row map") &&
              ck(cudaMalloc(&d_w, 3 * sizeof(float)), "alloc weights");
    if (!ok) goto done;

    {
        const int32_t tok[] = {0, 1}, off[] = {0, 2, 3}, row_map[] = {0, 1, 2};
        const float weights[] = {0.25f, 0.75f, 1.0f};
        ok = ck(cudaMemcpy(d_shared, shared.data(), shared.size() * sizeof(float), cudaMemcpyHostToDevice), "copy shared") &&
             ck(cudaMemcpy(d_sg, sg.data(), sg.size() * sizeof(float), cudaMemcpyHostToDevice), "copy gates") &&
             ck(cudaMemcpy(d_dm, dm.data(), dm.size() * sizeof(float), cudaMemcpyHostToDevice), "copy expert rows") &&
             ck(cudaMemcpy(d_tok, tok, sizeof(tok), cudaMemcpyHostToDevice), "copy token map") &&
             ck(cudaMemcpy(d_off, off, sizeof(off), cudaMemcpyHostToDevice), "copy offsets") &&
             ck(cudaMemcpy(d_rows, row_map, sizeof(row_map), cudaMemcpyHostToDevice), "copy row map") &&
             ck(cudaMemcpy(d_w, weights, sizeof(weights), cudaMemcpyHostToDevice), "copy weights");
        if (!ok) goto done;

        strata::prefill::moe_shared_init(d_shared, d_sg, d_acc, 2, nullptr);
        strata::prefill::moe_scatter_add(d_dm, d_tok, d_off, d_rows, d_w, 2, d_acc, nullptr);
        ok = ck(cudaDeviceSynchronize(), "run kernels");
        if (!ok) goto done;

        std::vector<float> got(shared.size());
        ok = ck(cudaMemcpy(got.data(), d_acc, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "read accumulator");
        if (!ok) goto done;
        for (int64_t t = 0; t < 2 && ok; ++t) {
            const float gate = 1.0f / (1.0f + std::exp(-sg[(size_t) t]));
            for (int64_t d = 0; d < kWidth; ++d) {
                const float routed = t == 0 ? 0.25f * dm[(size_t) d] + 0.75f * dm[(size_t) (kWidth + d)]
                                            : dm[(size_t) (2 * kWidth + d)];
                const float want = shared[(size_t) (t * kWidth + d)] * gate + routed;
                if (std::fabs(got[(size_t) (t * kWidth + d)] - want) > 2.0e-5f) {
                    std::fprintf(stderr, "layer_major_kernel_test: mismatch at token %lld column %lld: got %.8g, want %.8g\n",
                                 (long long) t, (long long) d, got[(size_t) (t * kWidth + d)], want);
                    ok = false;
                    break;
                }
            }
        }
    }

done:
    if (d_w) cudaFree(d_w);
    if (d_rows) cudaFree(d_rows);
    if (d_off) cudaFree(d_off);
    if (d_tok) cudaFree(d_tok);
    if (d_dm) cudaFree(d_dm);
    if (d_acc) cudaFree(d_acc);
    if (d_sg) cudaFree(d_sg);
    if (d_shared) cudaFree(d_shared);
    return ok ? 0 : 1;
}
