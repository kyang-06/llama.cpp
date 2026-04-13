// bench-linear.cpp — benchmark ggml_mul_mat (linear projection) for quantized weight types.
//
// Measures decoding-style matrix-vector multiply: weight [N×K] × input [K×1] → output [N×1].
// Reports average latency (ms), tokens/s, effective memory bandwidth (GB/s), and GFLOPS.
//
// Build:  cmake --build build --target bench-linear
// Run:    ./build/bin/bench-linear [--warmup N] [--iters N] [--cpu]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
static int  g_warmup   = 20;
static int  g_iters    = 200;
static bool g_cpu_only = false;

// Shapes: {N (output dim), K (input dim)}
struct Shape { int N, K; const char * label; };
static const Shape g_shapes[] = {
    {12288, 2048, "12288×2048"},
    { 9216, 2048,  "9216×2048"},
    { 2048, 4096,  "2048×4096"},
    { 5120, 2048,  "5120×2048"},
    { 2048,  512,   "2048×512"},
};

static const ggml_type g_types[] = {
    GGML_TYPE_Q4_0,
    GGML_TYPE_TQ2_0,
    GGML_TYPE_BPT1_0,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static const char * type_label(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:    return "F32";
        case GGML_TYPE_F16:    return "F16";
        case GGML_TYPE_Q4_0:   return "Q4_0";
        case GGML_TYPE_Q8_0:   return "Q8_0";
        case GGML_TYPE_TQ1_0:  return "TQ1_0";
        case GGML_TYPE_TQ2_0:  return "TQ2_0";
        case GGML_TYPE_BPT1_0: return "BPT1_0";
        default:               return ggml_type_name(t);
    }
}

static void fill_random(float * buf, size_t n, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) buf[i] = dist(rng);
}

static int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Benchmark one (shape, type) combination
// ---------------------------------------------------------------------------
static void bench_one(ggml_backend_t backend, ggml_type wtype, const Shape & s) {
    const int N = s.N, K = s.K;
    const size_t weight_bytes = (size_t)N * ggml_row_size(wtype, K);

    // --- prepare CPU-side weight data ---
    std::vector<float> wf32((size_t)N * K);
    fill_random(wf32.data(), wf32.size(), 1234);

    std::vector<uint8_t> wq(weight_bytes);
    for (int r = 0; r < N; ++r) {
        ggml_quantize_chunk(wtype, wf32.data() + r * K, wq.data() + r * ggml_row_size(wtype, K),
                            0, 1, K, nullptr);
    }

    // --- prepare CPU-side input data ---
    std::vector<float> inf32(K);
    fill_random(inf32.data(), K, 5678);

    // --- build graph (weight tensor + input tensor + mul_mat node) ---
    // context: tensors only, no_alloc=true; backend buffers hold the actual memory
    const size_t ctx_size = 4 * ggml_tensor_overhead() + ggml_graph_overhead();
    struct ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/true };
    struct ggml_context * ctx = ggml_init(ip);

    // Weight: ne[0]=K (cols), ne[1]=N (rows)  — ggml stores column-major
    struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, wtype,  K, N);
    struct ggml_tensor * input  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);

    // Allocate both tensors on the backend
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // Upload data
    ggml_backend_tensor_set(weight, wq.data(), 0, weight_bytes);
    ggml_backend_tensor_set(input,  inf32.data(), 0, K * sizeof(float));

    // Build forward graph
    struct ggml_cgraph * gf  = ggml_new_graph(ctx);
    struct ggml_tensor * out = ggml_mul_mat(ctx, weight, input);
    ggml_build_forward_expand(gf, out);

    // Allocate graph intermediates (the output tensor) via gallocr
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    // --- warmup ---
    for (int i = 0; i < g_warmup; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);

    // --- timed loop ---
    int64_t t0 = now_us();
    for (int i = 0; i < g_iters; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);
    int64_t t1 = now_us();

    double avg_ms  = (double)(t1 - t0) / g_iters / 1000.0;
    double tok_s   = 1000.0 / avg_ms;
    double gbps    = (double)weight_bytes / (avg_ms * 1e-3) / 1e9;
    double gflops  = 2.0 * N * K / (avg_ms * 1e-3) / 1e9;
    double bpw     = 8.0 * weight_bytes / ((double)N * K);

    printf("  %-6s  %10s  bpw=%4.2f  %7.3f ms  %8.1f tok/s  %6.2f GB/s  %5.2f GFLOPS\n",
           type_label(wtype), s.label, bpw, avg_ms, tok_s, gbps, gflops);

    ggml_gallocr_free(allocr);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--warmup") && i + 1 < argc) { g_warmup = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--iters")  && i + 1 < argc) { g_iters  = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--cpu"))  { g_cpu_only = true; }
        else {
            fprintf(stderr, "usage: %s [--warmup N] [--iters N] [--cpu]\n", argv[0]);
            return 1;
        }
    }

    // Load all backends (CUDA, CPU, etc.) from dynamic libraries
    ggml_backend_load_all();

    // Pick best available backend (GPU first, then CPU)
    ggml_backend_t backend = nullptr;
    if (!g_cpu_only) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (!backend) {
            fprintf(stderr, "note: no GPU backend available, using CPU\n");
        }
    }
    if (!backend) {
        backend = ggml_backend_cpu_init();
    }

    printf("Backend : %s\n", ggml_backend_name(backend));
    printf("Warmup  : %d   Iters: %d\n\n", g_warmup, g_iters);

    const int W = 88;
    printf("  %-6s  %10s  %-7s  %10s  %13s  %11s  %12s\n",
           "type", "shape", "bpw", "avg_ms", "tok/s", "bandwidth", "GFLOPS");
    printf("  %s\n", std::string(W, '-').c_str());

    for (const auto & sh : g_shapes) {
        printf("\n");
        for (ggml_type t : g_types) {
            bench_one(backend, t, sh);
        }
    }

    printf("\n  %s\n", std::string(W, '-').c_str());

    ggml_backend_free(backend);
    return 0;
}
