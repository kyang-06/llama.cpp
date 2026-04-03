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

// TQ1_0 dequantize: 1.6875 bpw ternary, elements stored in mixed qs/qh arrays.
// Layout: qs[48] holds 240 elements (5 trits/byte), qh[4] holds 16 elements (4 trits/byte).
//
// Element e maps to:
//   e in [0,   160): byte = qs[e%32],         trit index = e/32      (pow3[0..4])
//   e in [160, 240): byte = qs[32+(e-160)%16], trit index = (e-160)/16  (pow3[0..4])
//   e in [240, 256): byte = qh[(e-240)%4],     trit index = (e-240)/4   (pow3[0..3])
//
// Trit extraction: q = (uint8_t)(byte * pow3[n]); trit = (q*3u >> 8); weight = trit - 1.
//
// iqs is always even (0,2,...,254); both iqs and iqs+1 fall in the same region
// and share the same pow3 multiplier since all region boundaries are even.
static __device__ __forceinline__ void dequantize_tq1_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_tq1_0 * x = (const block_tq1_0 *) vx;

    const float d = __half2float(x[ib].d);

    // pow3[n] = 3^n for n=0..5, all fit in uint8_t
    constexpr uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

    uint8_t b0, b1, p;

    if (iqs < 160) {
        // qs[0..31]: each byte holds 5 trits for elements spaced 32 apart.
        // Element e -> qs[e%32], trit e/32.
        b0 = x[ib].qs[iqs % 32];
        b1 = x[ib].qs[(iqs + 1) % 32];
        p  = pow3[iqs / 32];
    } else if (iqs < 240) {
        // qs[32..47]: each byte holds 5 trits for elements spaced 16 apart.
        // Element e -> qs[32 + (e-160)%16], trit (e-160)/16.
        const int em = iqs - 160;
        b0 = x[ib].qs[32 + em % 16];
        b1 = x[ib].qs[32 + (em + 1) % 16];
        p  = pow3[em / 16];
    } else {
        // qh[0..3]: each byte holds 4 trits for elements spaced 4 apart.
        // Element e -> qh[(e-240)%4], trit (e-240)/4.
        const int em = iqs - 240;
        b0 = x[ib].qh[em % 4];
        b1 = x[ib].qh[(em + 1) % 4];
        p  = pow3[em / 4];
    }

    // Decode: multiply truncates to uint8, extract trit via fixed-point, subtract 1.
    v.x = d * ((int)(((uint8_t)(b0 * p) * 3u) >> 8) - 1);
    v.y = d * ((int)(((uint8_t)(b1 * p) * 3u) >> 8) - 1);
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
