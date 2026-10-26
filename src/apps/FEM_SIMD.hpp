//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//
// Thin AVX-512 (double, FEM_VL = 8 lanes) layer for hand-vectorized CPU SoA
// variants. RAJAPERF_FEM_AVX512 is defined when it is usable; compile with
// -DRAJAPERF_FEM_NO_INTRINSICS to fall back to the compiler-vectorized code.
//
#ifndef RAJAPerf_FEM_SIMD_HPP
#define RAJAPerf_FEM_SIMD_HPP

#include "FEM_MACROS.hpp"

// full unroll of short fixed-trip loops so array indices fold to constants
#if defined(__clang__)
#define RAJAPERF_FEM_UNROLL _Pragma("clang loop unroll(full)")
#else
#define RAJAPERF_FEM_UNROLL _Pragma("GCC unroll 16")
#endif

// large per-batch kernels: one out-of-line copy shared by all variants
#define RAJAPERF_FEM_NOINLINE inline __attribute__((noinline))

// for lambda bodies, e.g. the ones passed to static_for: [&](auto I) ATTR {...}
#define RAJAPERF_FEM_LAMBDA_INLINE __attribute__((always_inline))

// Keeps an alignas(64) local in memory, which makes clang realign the frame;
// otherwise zmm spills use caller-dependent (possibly line-splitting) slots.
#define RAJAPERF_FEM_STACK_ALIGN(a) asm volatile("" : : "r"(a))

#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(RP_USE_DOUBLE) && \
    !defined(RAJAPERF_FEM_NO_INTRINSICS)
#define RAJAPERF_FEM_AVX512

#include <immintrin.h>

namespace rajaperf {
namespace fem_simd {

using vd = __m512d;
using vmask = __mmask8;

static_assert(FEM_VL == 8, "AVX-512 path assumes 8 double lanes");

RAJA_INLINE vd vload(const Real_type *p) { return _mm512_loadu_pd(p); }
RAJA_INLINE vd vload(vmask m, const Real_type *p) {
  return _mm512_maskz_loadu_pd(m, p);
}
RAJA_INLINE void vstore(Real_type *p, vd a) { _mm512_storeu_pd(p, a); }
RAJA_INLINE void vstore(Real_type *p, vmask m, vd a) {
  _mm512_mask_storeu_pd(p, m, a);
}
RAJA_INLINE vd vset1(Real_type a) { return _mm512_set1_pd(a); }
RAJA_INLINE vd vzero() { return _mm512_setzero_pd(); }
RAJA_INLINE vd vadd(vd a, vd b) { return _mm512_add_pd(a, b); }
RAJA_INLINE vd vsub(vd a, vd b) { return _mm512_sub_pd(a, b); }
RAJA_INLINE vd vmul(vd a, vd b) { return _mm512_mul_pd(a, b); }
RAJA_INLINE vd vdiv(vd a, vd b) { return _mm512_div_pd(a, b); }
// a*b + c
RAJA_INLINE vd vfma(vd a, vd b, vd c) { return _mm512_fmadd_pd(a, b, c); }
// a*b - c
RAJA_INLINE vd vfms(vd a, vd b, vd c) { return _mm512_fmsub_pd(a, b, c); }
// c - a*b
RAJA_INLINE vd vfnma(vd a, vd b, vd c) { return _mm512_fnmadd_pd(a, b, c); }
RAJA_INLINE vd vabs(vd a) { return _mm512_abs_pd(a); }
RAJA_INLINE Real_type vhsum(vd a) { return _mm512_reduce_add_pd(a); }

RAJA_INLINE vmask lane_mask(Index_type n) {
  return static_cast<vmask>((n >= 8) ? 0xFF : ((1u << n) - 1u));
}

// (a < b) ? x : y per lane
RAJA_INLINE vd vsel_lt(vd a, vd b, vd x, vd y) {
  return _mm512_mask_blend_pd(_mm512_cmp_pd_mask(a, b, _CMP_LT_OQ), y, x);
}

// NV independent 8-lane vectors in lockstep (8*NV elements): every operation
// issues NV independent instructions, i.e. NV-way ILP per dependency chain.
template <int NV>
struct vdx {
  vd r[NV];
};

template <int NV>
RAJA_INLINE vdx<NV> vsplat(vd a) {
  vdx<NV> o;
  RAJAPERF_FEM_UNROLL
  for (int k = 0; k < NV; ++k) {
    o.r[k] = a;
  }
  return o;
}

#define RAJAPERF_FEM_VDX_OP(op, PARAMS, ARGS)                                \
  template <int NV>                                                          \
  RAJA_INLINE vdx<NV> op PARAMS {                                            \
    vdx<NV> o;                                                               \
    RAJAPERF_FEM_UNROLL                                                      \
    for (int k = 0; k < NV; ++k) {                                           \
      o.r[k] = op ARGS;                                                      \
    }                                                                        \
    return o;                                                                \
  }

RAJAPERF_FEM_VDX_OP(vabs, (vdx<NV> a), (a.r[k]))
RAJAPERF_FEM_VDX_OP(vadd, (vdx<NV> a, vdx<NV> b), (a.r[k], b.r[k]))
RAJAPERF_FEM_VDX_OP(vsub, (vdx<NV> a, vdx<NV> b), (a.r[k], b.r[k]))
RAJAPERF_FEM_VDX_OP(vmul, (vdx<NV> a, vdx<NV> b), (a.r[k], b.r[k]))
RAJAPERF_FEM_VDX_OP(vdiv, (vdx<NV> a, vdx<NV> b), (a.r[k], b.r[k]))
RAJAPERF_FEM_VDX_OP(vfma, (vdx<NV> a, vdx<NV> b, vdx<NV> c),
                    (a.r[k], b.r[k], c.r[k]))
RAJAPERF_FEM_VDX_OP(vfms, (vdx<NV> a, vdx<NV> b, vdx<NV> c),
                    (a.r[k], b.r[k], c.r[k]))
RAJAPERF_FEM_VDX_OP(vsel_lt, (vdx<NV> a, vdx<NV> b, vdx<NV> x, vdx<NV> y),
                    (a.r[k], b.r[k], x.r[k], y.r[k]))
#undef RAJAPERF_FEM_VDX_OP

// In-register 8x8 transpose: r[i][j] -> r[j][i].
RAJA_INLINE void transpose8(vd (&r)[8]) {
  const vd t0 = _mm512_unpacklo_pd(r[0], r[1]);
  const vd t1 = _mm512_unpackhi_pd(r[0], r[1]);
  const vd t2 = _mm512_unpacklo_pd(r[2], r[3]);
  const vd t3 = _mm512_unpackhi_pd(r[2], r[3]);
  const vd t4 = _mm512_unpacklo_pd(r[4], r[5]);
  const vd t5 = _mm512_unpackhi_pd(r[4], r[5]);
  const vd t6 = _mm512_unpacklo_pd(r[6], r[7]);
  const vd t7 = _mm512_unpackhi_pd(r[6], r[7]);
  const vd u0 = _mm512_shuffle_f64x2(t0, t2, 0x88);
  const vd u1 = _mm512_shuffle_f64x2(t0, t2, 0xDD);
  const vd u2 = _mm512_shuffle_f64x2(t1, t3, 0x88);
  const vd u3 = _mm512_shuffle_f64x2(t1, t3, 0xDD);
  const vd u4 = _mm512_shuffle_f64x2(t4, t6, 0x88);
  const vd u5 = _mm512_shuffle_f64x2(t4, t6, 0xDD);
  const vd u6 = _mm512_shuffle_f64x2(t5, t7, 0x88);
  const vd u7 = _mm512_shuffle_f64x2(t5, t7, 0xDD);
  r[0] = _mm512_shuffle_f64x2(u0, u4, 0x88);
  r[4] = _mm512_shuffle_f64x2(u0, u4, 0xDD);
  r[2] = _mm512_shuffle_f64x2(u1, u5, 0x88);
  r[6] = _mm512_shuffle_f64x2(u1, u5, 0xDD);
  r[1] = _mm512_shuffle_f64x2(u2, u6, 0x88);
  r[5] = _mm512_shuffle_f64x2(u2, u6, 0xDD);
  r[3] = _mm512_shuffle_f64x2(u3, u7, 0x88);
  r[7] = _mm512_shuffle_f64x2(u3, u7, 0xDD);
}

// AoS -> SoA: out[k] = {row[0][k], ..., row[7][k]} for k < N, where row[l]
// points at N contiguous values of lane l.
template <Index_type N>
RAJA_INLINE void load_transpose(const Real_type *const (&row)[8],
                                vd *RAJA_RESTRICT out) {
  for (Index_type c = 0; c < N; c += 8) {
    const Index_type n = (N - c < 8) ? N - c : 8;
    const vmask m = lane_mask(n);
    vd r[8];
    for (Index_type l = 0; l < 8; ++l) {
      r[l] = vload(m, row[l] + c);
    }
    transpose8(r);
    for (Index_type k = 0; k < n; ++k) {
      out[c + k] = r[k];
    }
  }
}

// SoA -> AoS: row[l][k] = in[k][l] (or += when ACCUMULATE) for k < N and
// lanes l < nel.
template <Index_type N, bool ACCUMULATE = false>
RAJA_INLINE void transpose_store(Real_type *const (&row)[8], Index_type nel,
                                 const vd *RAJA_RESTRICT in) {
  for (Index_type c = 0; c < N; c += 8) {
    const Index_type n = (N - c < 8) ? N - c : 8;
    const vmask m = lane_mask(n);
    vd r[8];
    for (Index_type k = 0; k < 8; ++k) {
      r[k] = (k < n) ? in[c + k] : vzero();
    }
    transpose8(r);
    for (Index_type l = 0; l < nel; ++l) {
      if constexpr (ACCUMULATE) {
        vstore(row[l] + c, m, vadd(vload(m, row[l] + c), r[l]));
      } else {
        vstore(row[l] + c, m, r[l]);
      }
    }
  }
}

// Sum-factorized 3D mass operator for 8 elements (lanes):
//   y = B^T (d .* (B x)) per dimension, x, y as [dz][dy][dx], d as [qz][qy][qx],
//   b[q][d] = B(q, d). Same operation order as the MASS*PA stage macros.
template <Index_type D1, Index_type Q1>
RAJA_INLINE void mass_pa_apply(const Real_type (&b)[Q1][D1],
                               const vd *RAJA_RESTRICT x,
                               const vd *RAJA_RESTRICT d,
                               vd *RAJA_RESTRICT y) {
  vd t1[D1][D1][Q1]; // [dz][dy][qx]
  vd t2[D1][Q1][Q1]; // [dz][qy][qx]
  vd t3[Q1][Q1][Q1]; // [qz][qy][qx]
  vd t4[Q1][Q1][D1]; // [qz][qy][dx]
  vd t5[Q1][D1][D1]; // [qz][dy][dx]

  RAJAPERF_FEM_UNROLL
  for (Index_type dz = 0; dz < D1; ++dz) {
    RAJAPERF_FEM_UNROLL
    for (Index_type dy = 0; dy < D1; ++dy) {
      RAJAPERF_FEM_UNROLL
      for (Index_type qx = 0; qx < Q1; ++qx) {
        vd u = vmul(x[D1 * (dy + D1 * dz)], vset1(b[qx][0]));
        RAJAPERF_FEM_UNROLL
        for (Index_type dx = 1; dx < D1; ++dx) {
          u = vfma(x[dx + D1 * (dy + D1 * dz)], vset1(b[qx][dx]), u);
        }
        t1[dz][dy][qx] = u;
      }
    }
  }
  RAJAPERF_FEM_UNROLL
  for (Index_type dz = 0; dz < D1; ++dz) {
    RAJAPERF_FEM_UNROLL
    for (Index_type qy = 0; qy < Q1; ++qy) {
      RAJAPERF_FEM_UNROLL
      for (Index_type qx = 0; qx < Q1; ++qx) {
        vd u = vmul(t1[dz][0][qx], vset1(b[qy][0]));
        RAJAPERF_FEM_UNROLL
        for (Index_type dy = 1; dy < D1; ++dy) {
          u = vfma(t1[dz][dy][qx], vset1(b[qy][dy]), u);
        }
        t2[dz][qy][qx] = u;
      }
    }
  }
  RAJAPERF_FEM_UNROLL
  for (Index_type qz = 0; qz < Q1; ++qz) {
    RAJAPERF_FEM_UNROLL
    for (Index_type qy = 0; qy < Q1; ++qy) {
      RAJAPERF_FEM_UNROLL
      for (Index_type qx = 0; qx < Q1; ++qx) {
        vd u = vmul(t2[0][qy][qx], vset1(b[qz][0]));
        RAJAPERF_FEM_UNROLL
        for (Index_type dz = 1; dz < D1; ++dz) {
          u = vfma(t2[dz][qy][qx], vset1(b[qz][dz]), u);
        }
        t3[qz][qy][qx] = vmul(u, d[qx + Q1 * (qy + Q1 * qz)]);
      }
    }
  }
  RAJAPERF_FEM_UNROLL
  for (Index_type qz = 0; qz < Q1; ++qz) {
    RAJAPERF_FEM_UNROLL
    for (Index_type qy = 0; qy < Q1; ++qy) {
      RAJAPERF_FEM_UNROLL
      for (Index_type dx = 0; dx < D1; ++dx) {
        vd u = vmul(t3[qz][qy][0], vset1(b[0][dx]));
        RAJAPERF_FEM_UNROLL
        for (Index_type qx = 1; qx < Q1; ++qx) {
          u = vfma(t3[qz][qy][qx], vset1(b[qx][dx]), u);
        }
        t4[qz][qy][dx] = u;
      }
    }
  }
  RAJAPERF_FEM_UNROLL
  for (Index_type qz = 0; qz < Q1; ++qz) {
    RAJAPERF_FEM_UNROLL
    for (Index_type dy = 0; dy < D1; ++dy) {
      RAJAPERF_FEM_UNROLL
      for (Index_type dx = 0; dx < D1; ++dx) {
        vd u = vmul(t4[qz][0][dx], vset1(b[0][dy]));
        RAJAPERF_FEM_UNROLL
        for (Index_type qy = 1; qy < Q1; ++qy) {
          u = vfma(t4[qz][qy][dx], vset1(b[qy][dy]), u);
        }
        t5[qz][dy][dx] = u;
      }
    }
  }
  RAJAPERF_FEM_UNROLL
  for (Index_type dz = 0; dz < D1; ++dz) {
    RAJAPERF_FEM_UNROLL
    for (Index_type dy = 0; dy < D1; ++dy) {
      RAJAPERF_FEM_UNROLL
      for (Index_type dx = 0; dx < D1; ++dx) {
        vd u = vmul(t5[0][dy][dx], vset1(b[0][dz]));
        RAJAPERF_FEM_UNROLL
        for (Index_type qz = 1; qz < Q1; ++qz) {
          u = vfma(t5[qz][dy][dx], vset1(b[qz][dz]), u);
        }
        y[dx + D1 * (dy + D1 * dz)] = u;
      }
    }
  }
}

// 8 overlapping 3-node segments of a contiguous 17-node row p[0..16]
// (segment k = p[2k..2k+2], e.g. the nodes of 8 consecutive quadratic
// elements along x): v[dx] lane k = p[2k + dx]. No gathers.
RAJA_INLINE void load_seg3_x8(const Real_type *p, vd (&v)[3]) {
  const __m512i even = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
  const __m512i odd = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
  const vd a0 = vload(p);
  const vd a8 = vload(p + 8);
  const vd b1 = vload(p + 1);
  const vd b9 = vload(p + 9);
  v[0] = _mm512_permutex2var_pd(a0, even, a8);
  v[1] = _mm512_permutex2var_pd(a0, odd, a8);
  v[2] = _mm512_permutex2var_pd(b1, odd, b9);
}

// Transpose of load_seg3_x8: p[2k + dx] += v[dx] lane k. The shared node
// p[2k+2] (k < 7) receives v[2]_k + v[0]_{k+1}, so every node is updated by
// exactly one plain read-modify-write (no conflicts, no scatters).
RAJA_INLINE void add_seg3_x8(Real_type *p, const vd (&v)[3]) {
  const __m512i ilo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
  const __m512i ihi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
  // [0, v2_0, ..., v2_6]
  const vd v2s = _mm512_castsi512_pd(_mm512_alignr_epi64(
      _mm512_castpd_si512(v[2]), _mm512_setzero_si512(), 7));
  const vd ev = vadd(v[0], v2s);
  vstore(p, vadd(vload(p), _mm512_permutex2var_pd(ev, ilo, v[1])));
  vstore(p + 8, vadd(vload(p + 8), _mm512_permutex2var_pd(ev, ihi, v[1])));
  // p[16] += v2_7 (lane 7 of a vector at p + 9)
  vstore(p + 9, vmask(0x80), vadd(vload(vmask(0x80), p + 9), v[2]));
}

} // namespace fem_simd
} // namespace rajaperf

#endif

#include <cstdint>
#include <type_traits>
#include <utility>

namespace rajaperf {
namespace fem_simd {

// f(std::integral_constant<int, I>{}) for I = 0..N-1: compile-time loop indices
template <class F, int... I>
RAJA_INLINE void static_for_impl(F &&f, std::integer_sequence<int, I...>) {
  (f(std::integral_constant<int, I>{}), ...);
}
template <int N, class F>
RAJA_INLINE void static_for(F &&f) {
  static_for_impl(f, std::make_integer_sequence<int, N>{});
}

RAJA_INLINE bool is_aligned64(const void *p) {
  return (reinterpret_cast<std::uintptr_t>(p) & 63u) == 0;
}

// dst[0:n) = src[0:n); non-temporal (no write-allocate) stores when AVX-512 is
// available. dst and src must be 64B aligned and n a multiple of 8.
RAJA_INLINE void stream_copy(Real_type *RAJA_RESTRICT dst,
                             const Real_type *RAJA_RESTRICT src, Index_type n) {
#if defined(RAJAPERF_FEM_AVX512)
  for (Index_type k = 0; k < n; k += 8) {
    _mm512_stream_pd(dst + k, _mm512_load_pd(src + k));
  }
#else
  for (Index_type k = 0; k < n; ++k) {
    dst[k] = src[k];
  }
#endif
}

// Software prefetch of every cache line of [p, p + bytes).
template <bool WRITE = false>
RAJA_INLINE void prefetch_range(const void *p, Index_type bytes) {
  const char *const c = static_cast<const char *>(p);
  for (Index_type o = 0; o < bytes; o += 64) {
    __builtin_prefetch(c + o, WRITE ? 1 : 0, 3);
  }
}

// Orders a thread's non-temporal stores before later synchronization.
RAJA_INLINE void stream_fence() {
#if defined(RAJAPERF_FEM_AVX512)
  _mm_sfence();
#endif
}

} // namespace fem_simd
} // namespace rajaperf

#endif // closing endif for header file include guard
