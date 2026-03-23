#include "common.cuh"

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

// TQ2_0 dequantize: 2-bit ternary {0,1,2} -> float {-1,0,+1} * scale
// The dequantize_block framework calls this with even iqs values (0, 2, 4, ..., 254).
// Elements iqs and iqs+1 always share the same byte because:
//   byte index = iqs / 4, and with iqs even: iqs%4 is 0 or 2, so (iqs+1)%4 is 1 or 3 (same byte).
static __device__ __forceinline__ void dequantize_tq2_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_tq2_0 * x = (const block_tq2_0 *) vx;

    const float d = __half2float(x[ib].d);

    // iqs/4 gives the byte containing elements iqs..iqs+3 (4 elements per byte).
    // (iqs & 3) is 0 or 2 (iqs always even), so shift is 0 or 4, placing
    // element iqs at bits [shift..shift+1] and element iqs+1 at bits [shift+2..shift+3].
    const uint8_t byte = x[ib].qs[iqs / 4];
    const int shift = (iqs & 3) * 2;  // 0 or 4, since iqs%4 is 0 or 2 for even iqs

    const int val0 = (byte >> shift)       & 3;
    const int val1 = (byte >> (shift + 2)) & 3;

    // Stored encoding: -1->0, 0->1, +1->2; decode by subtracting 1
    v.x = d * (float)(val0 - 1);
    v.y = d * (float)(val1 - 1);
}
