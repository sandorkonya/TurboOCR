// JFA (Jump Flooding Algorithm) + SDF for per-component Euclidean unclip.
//
// Pipeline (no component merging):
// 1. CCL on original bitmap → label map (uint32, 0=background)
// 2. JFA from foreground pixels: propagate nearest (x,y) coordinates
// 3. Expand: for each pixel where distance ≤ expand, look up the label
//    of the nearest foreground pixel from the original label map.
//    This assigns each expanded pixel to the CORRECT component,
//    preventing merge even when expanded regions overlap.
// 4. Extract bboxes from the expanded label map.

#include "turbo_ocr/kernels/kernels.h"
#include "turbo_ocr/common/cuda_check.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <climits>

namespace turbo_ocr::kernels {

__global__ void jfa_init_kernel(int2 *seeds, const uint8_t *bitmap, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    seeds[idx] = bitmap[idx] ? make_int2(x, y) : make_int2(-1, -1);
}

// Ping-pong: read from in_seeds, write to out_seeds. Avoids the in-place
// race where neighbor reads see partial writes from concurrent blocks.
__global__ void jfa_step_kernel(const int2 *in_seeds, int2 *out_seeds,
                                int w, int h, int s) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;
    int2 best = in_seeds[idx];
    long long best_d2 = (best.x < 0) ? LLONG_MAX :
        ((long long)(x - best.x) * (x - best.x) +
         (long long)(y - best.y) * (y - best.y));

    const int dx[8] = {s, s, 0, -s, -s, -s, 0, s};
    const int dy[8] = {0, -s, -s, -s, 0, s, s, s};

    #pragma unroll
    for (int d = 0; d < 8; d++) {
        int nx = x + dx[d], ny = y + dy[d];
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        int2 ns = in_seeds[ny * w + nx];
        if (ns.x < 0) continue;
        long long d2 = (long long)(x - ns.x) * (x - ns.x) +
                        (long long)(y - ns.y) * (y - ns.y);
        if (d2 < best_d2) { best = ns; best_d2 = d2; }
    }
    out_seeds[idx] = best;
}

// Step 3: Produce expanded label map using COMPACT labels (0..N-1).
// Per-component expand cutoff: expand_per_comp[cid] = area_i*ratio/perimeter_i,
// matching Clipper's polygon offset distance. This closes the F1 gap to CPU.
__global__ void jfa_expand_labels_kernel(uint32_t *expanded_labels,
                                          const int2 *seeds,
                                          const int32_t *compact_ids,
                                          const float *expand_per_comp,
                                          int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    int2 s = seeds[idx];
    if (s.x < 0) { expanded_labels[idx] = 0; return; }
    int cid = compact_ids[s.y * w + s.x];
    if (cid < 0) { expanded_labels[idx] = 0; return; }
    float e = expand_per_comp[cid]; // 0 for filtered-out / empty slots
    if (e <= 0.0f) { expanded_labels[idx] = 0; return; }
    long long d2 = (long long)(x - s.x) * (x - s.x) +
                    (long long)(y - s.y) * (y - s.y);
    double e2 = (double)e * (double)e;
    if ((double)d2 > e2) { expanded_labels[idx] = 0; return; }
    expanded_labels[idx] = (uint32_t)(cid + 1); // +1 so 0=bg
}

// Compute per-component expand distance for ALL pre-filter compact_ids.
// expand_i = area_i * ratio / perimeter_i (Clipper-equivalent for solid axis-
// aligned components). Indexed by PRE-filter compact_id so JFA's
// compact_ids[seed] lookup is consistent. Returns 0 for empty / score-rejected
// slots (those become bg in JFA expand).
__global__ void compute_expand_per_comp_kernel(
    const GpuDetBox *bboxes, int num_slots,
    float unclip_ratio, float min_expand, float max_expand,
    float box_thresh, float *expand_per_comp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_slots) return;
    const GpuDetBox &b = bboxes[i];
    int pc = b.pixel_count;
    if (pc < 3) { expand_per_comp[i] = 0.0f; return; }
    int bw = b.xmax - b.xmin + 1;
    int bh = b.ymax - b.ymin + 1;
    if (bw < 3 || bh < 3) { expand_per_comp[i] = 0.0f; return; }
    float score = b.score / (float)pc;
    if (score < box_thresh) { expand_per_comp[i] = 0.0f; return; }
    float perim = 2.0f * (float)(bw + bh);
    float area = (float)pc;
    float e = area * unclip_ratio / perim;
    if (e < min_expand) e = min_expand;
    if (e > max_expand) e = max_expand;
    expand_per_comp[i] = e;
}

// Step 4a: init bbox slots with sentinels so atomic scatter min/max works.
__global__ void jfa_init_bboxes_kernel(GpuDetBox *bboxes, int num_slots) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_slots) return;
    GpuDetBox &b = bboxes[i];
    b.xmin = INT_MAX; b.ymin = INT_MAX;
    b.xmax = INT_MIN; b.ymax = INT_MIN;
    b.pixel_count = 0;
    b.score = 0.0f;
}

// Step 4b: image-wide single pass, atomic scatter. One thread per pixel
// atomic-updates the bbox slot for its label. Replaces the previous
// per-component image scan (~50× less memory traffic at 2048 slots × 960²).
__global__ void jfa_extract_bboxes_kernel(
    const uint32_t *expanded_labels,
    int w, int h,
    GpuDetBox *bboxes, int num_slots) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = w * h;
    if (idx >= total) return;
    uint32_t label = expanded_labels[idx];
    if (label == 0) return;
    int cid = (int)label - 1;
    if (cid >= num_slots) return;
    int x = idx % w, y = idx / w;
    GpuDetBox &b = bboxes[cid];
    atomicMin(&b.xmin, x); atomicMax(&b.xmax, x);
    atomicMin(&b.ymin, y); atomicMax(&b.ymax, y);
    atomicAdd(&b.pixel_count, 1);
}

void cuda_jfa_expand_labels(const uint8_t *d_bitmap,
                            const int32_t *d_compact_ids,
                            const float *d_expand_per_comp,
                            uint32_t *d_expanded_labels,
                            int w, int h,
                            int2 *d_seeds, int2 *d_seeds_alt,
                            cudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((w + 31) / 32, (h + 7) / 8);

    jfa_init_kernel<<<grid, block, 0, stream>>>(d_seeds, d_bitmap, w, h);
    CUDA_CHECK(cudaGetLastError());

    int2 *in_seeds  = d_seeds;
    int2 *out_seeds = d_seeds_alt;
    int max_dim = max(w, h);
    for (int s = 1 << (31 - __builtin_clz(max_dim)); s > 0; s >>= 1) {
        jfa_step_kernel<<<grid, block, 0, stream>>>(in_seeds, out_seeds, w, h, s);
        CUDA_CHECK(cudaGetLastError());
        int2 *tmp = in_seeds; in_seeds = out_seeds; out_seeds = tmp;
    }
    // Final result must land in d_seeds (downstream consumer).
    if (in_seeds != d_seeds) {
        CUDA_CHECK(cudaMemcpyAsync(d_seeds, in_seeds,
                                   (size_t)w * h * sizeof(int2),
                                   cudaMemcpyDeviceToDevice, stream));
    }

    jfa_expand_labels_kernel<<<grid, block, 0, stream>>>(
        d_expanded_labels, d_seeds, d_compact_ids, d_expand_per_comp, w, h);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_compute_expand_per_comp(
    const GpuDetBox *d_bboxes, int num_slots,
    float unclip_ratio, float min_expand, float max_expand,
    float box_thresh, float *d_expand_per_comp, cudaStream_t stream) {
    if (num_slots <= 0) return;
    int block = 128;
    int grid = (num_slots + block - 1) / block;
    compute_expand_per_comp_kernel<<<grid, block, 0, stream>>>(
        d_bboxes, num_slots, unclip_ratio, min_expand, max_expand,
        box_thresh, d_expand_per_comp);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_jfa_extract_bboxes(const uint32_t *d_expanded_labels,
                             const float * /*d_pred_map*/,
                             const float * /*d_expand_per_comp*/,
                             int w, int h,
                             GpuDetBox *d_bboxes, int num_slots,
                             cudaStream_t stream) {
    if (num_slots <= 0) return;
    int total = w * h;
    int block = 256;
    int init_grid = (num_slots + block - 1) / block;
    jfa_init_bboxes_kernel<<<init_grid, block, 0, stream>>>(d_bboxes, num_slots);
    CUDA_CHECK(cudaGetLastError());
    int grid = (total + block - 1) / block;
    jfa_extract_bboxes_kernel<<<grid, block, 0, stream>>>(
        d_expanded_labels, w, h, d_bboxes, num_slots);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace turbo_ocr::kernels
