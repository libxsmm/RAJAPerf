//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Action of a 3D finite element mass matrix on elements with shared DOFs
/// via partial assembly and sum factorization
///
/// for (Index_type e = 0; e < NE; ++e) {
///
///   constexpr Index_type MQ1 = mpa_at::Q1D;
///   constexpr Index_type MD1 = mpa_at::D1D;
///
///   constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;
///
///   Real_type sm_B[MQ1][MD1];
///   Real_type sm_Bt[MD1][MQ1];
///
///   Real_type sm0[MDQ * MDQ * MDQ];
///   Real_type sm1[MDQ * MDQ * MDQ];
///   Real_type(*sm_X)[MD1][MD1]   = (Real_type(*)[MD1][MD1])sm0;
///   Real_type(*DDQ)[MD1][MQ1]    = (Real_type(*)[MD1][MQ1])sm1;
///   Real_type(*DQQ)[MQ1][MQ1]    = (Real_type(*)[MQ1][MQ1])sm0;
///   Real_type(*QQQ)[MQ1][MQ1]    = (Real_type(*)[MQ1][MQ1])sm1;
///   Real_type(*QQD)[MQ1][MD1]    = (Real_type(*)[MQ1][MD1])sm0;
///   Real_type(*QDD)[MD1][MD1]    = (Real_type(*)[MD1][MD1])sm1;
///
///   Index_type thread_dofs[MD1 * MD1 * MD1];
///
///   for(Index_type dz=0; dz<mpa_at::D1D; ++dz) {
///     for(Index_type dy=0; dy<mpa_at::D1D; ++dy) {
///       for(Index_type dx=0; dx<mpa_at::D1D; ++dx) {
///         Index_type j = dx + mpa_at::D1D * (dy + dz * mpa_at::D1D);
///         //missing dof_map for lexicographical ordering
///         thread_dofs[j] = ElemToDoF[j + mpa_at::D1D * mpa_at::D1D * mpa_at::D1D * e];
///         sm_X[dz][dy][dx]  = X[thread_dofs[j]];
///       }
///     }
///   }
///
///   for(Index_type d=0; d<mpa_at::D1D; ++d) {
///     for(Index_type q=0; q<mpa_at::Q1D; ++q) {
///       sm_B[q][d]  = MPAT_B(q, d);
///       sm_Bt[d][q] = sm_B[q][d];
///     }
///   }
///
///
///   for(Index_type dz=0; dz<mpa_at::D1D; ++dz) {
///     for(Index_type dy=0; dy<mpa_at::D1D; ++dy) {
///       for(Index_type qx=0; qx<mpa_at::Q1D; ++qx) {
///         Real_type u = 0.0;
///         for (Index_type dx = 0; dx < mpa_at::D1D; ++dx)
///         {
///           u += sm_X[dz][dy][dx] * sm_B[qx][dx];
///         }
///         DDQ[dz][dy][qx] = u;
///       }
///     }
///   }
///
///   for(Index_type dz=0; dz<mpa_at::D1D; ++dz) {
///     for(Index_type qy=0; qy<mpa_at::Q1D; ++qy) {
///       for(Index_type qx=0; qx<mpa_at::Q1D; ++qx) {
///
///         Real_type u = 0.0;
///         for (Index_type dy = 0; dy < mpa_at::D1D; ++dy)
///         {
///           u += DDQ[dz][dy][qx] * sm_B[qy][dy];
///         }
///         DQQ[dz][qy][qx] = u;
///       }
///     }
///   }
///
///   for(Index_type qz=0; qz<mpa_at::Q1D; ++qz) {
///     for(Index_type qy=0; qy<mpa_at::Q1D; ++qy) {
///       for(Index_type qx=0; qx<mpa_at::Q1D; ++qx) {
///         Real_type u = 0.0;
///         for (Index_type dz = 0; dz < mpa_at::D1D; ++dz)
///         {
///           u += DQQ[dz][qy][qx] * sm_B[qz][dz];
///         }
///         QQQ[qz][qy][qx] = u * MPAT_D(qx, qy, qz, e);
///       }
///     }
///   }
///
///   for(Index_type qz=0; qz<mpa_at::Q1D; ++qz) {
///     for(Index_type qy=0; qy<mpa_at::Q1D; ++qy) {
///       for(Index_type dx=0; dx<mpa_at::D1D; ++dx) {
///         Real_type u = 0.0;
///         for (Index_type qx = 0; qx < mpa_at::Q1D; ++qx)
///         {
///           u += QQQ[qz][qy][qx] * sm_Bt[dx][qx];
///         }
///         QQD[qz][qy][dx] = u;
///       }
///     }
///   }
///
///   for(Index_type qz=0; qz<mpa_at::Q1D; ++qz) {
///     for(Index_type dy=0; dy<mpa_at::D1D; ++dy) {
///       for(Index_type dx=0; dx<mpa_at::D1D; ++dx) {
///         Real_type u = 0.0;
///         for (Index_type qy = 0; qy<mpa_at::Q1D; ++qy)
///         {
///         u += QQD[qz][qy][dx] * sm_Bt[dy][qy];
///         }
///         QDD[qz][dy][dx] = u;
///       }
///     }
///   }
///
///   for(Index_type dz=0; dz<mpa_at::D1D; ++dz) {
///     for(Index_type dy=0; dy<mpa_at::D1D; ++dy) {
///       for(Index_type dx=0; dx<mpa_at::D1D; ++dx) {
///         Real_type u = 0.0;
///         for (Index_type qz = 0; qz < mpa_at::Q1D; ++qz)
///         {
///            u += QDD[qz][dy][dx] * sm_Bt[dz][qz];
///         }
///         const Index_type j = dx + mpa_at::D1D * (dy + dz * mpa_at::D1D);
///         Y[thread_dofs[j]] += u; //atomic add
///       }
///     }
///   }
///
/// } // element loop
///

#ifndef RAJAPerf_Apps_MASS3DPA_ATOMIC_HPP
#define RAJAPerf_Apps_MASS3DPA_ATOMIC_HPP

#define MASS3DPA_ATOMIC_DATA_SETUP                                             \
  Real_ptr B = m_B;                                                            \
  Real_ptr D = m_D;                                                            \
  Real_ptr X = m_X;                                                            \
  Real_ptr Y = m_Y;                                                            \
  Index_ptr ElemToDoF = m_ElemToDoF;                                           \
  Index_type NE = m_NE;

#include "FEM_MACROS.hpp"
#include "FEM_SIMD.hpp"
#include "common/KernelBase.hpp"

#include "RAJA/RAJA.hpp"

// Number of Dofs/Qpts in 1D
namespace mpa_at {
constexpr RAJA::Index_type D1D = 3;
constexpr RAJA::Index_type Q1D = 4;
} // namespace mpa_at

#define MPAT_B(x, y) B[x + mpa_at::Q1D * y]
#define MPAT_D(qx, qy, qz, e)                                                  \
  D[qx + mpa_at::Q1D * qy + mpa_at::Q1D * mpa_at::Q1D * qz +                   \
    mpa_at::Q1D * mpa_at::Q1D * mpa_at::Q1D * e]

#define MASS3DPA_ATOMIC_0_CPU                                                  \
  constexpr Index_type MQ1 = mpa_at::Q1D;                                      \
  constexpr Index_type MD1 = mpa_at::D1D;                                      \
  constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;                          \
  Real_type sm_B[MQ1][MD1];                                                    \
  Real_type sm_Bt[MD1][MQ1];                                                   \
  Real_type sm0[MDQ * MDQ * MDQ];                                              \
  Real_type sm1[MDQ * MDQ * MDQ];                                              \
  Real_type(*sm_X)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm0;                    \
  Real_type(*DDQ)[MD1][MQ1] = (Real_type(*)[MD1][MQ1])sm1;                     \
  Real_type(*DQQ)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])sm0;                     \
  Real_type(*QQQ)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])sm1;                     \
  Real_type(*QQD)[MQ1][MD1] = (Real_type(*)[MQ1][MD1])sm0;                     \
  Real_type(*QDD)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm1;                     \
  Index_type thread_dofs[MD1 * MD1 * MD1];

#define MASS3DPA_ATOMIC_0_GPU                                                  \
  constexpr Index_type MQ1 = mpa_at::Q1D;                                      \
  constexpr Index_type MD1 = mpa_at::D1D;                                      \
  constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;                          \
  RAJA_TEAM_SHARED Real_type sm_B[MQ1][MD1];                                   \
  RAJA_TEAM_SHARED Real_type sm_Bt[MD1][MQ1];                                  \
  RAJA_TEAM_SHARED Real_type sm0[MDQ * MDQ * MDQ];                             \
  RAJA_TEAM_SHARED Real_type sm1[MDQ * MDQ * MDQ];                             \
  Real_type(*sm_X)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm0;                    \
  Real_type(*DDQ)[MD1][MQ1] = (Real_type(*)[MD1][MQ1])sm1;                     \
  Real_type(*DQQ)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])sm0;                     \
  Real_type(*QQQ)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])sm1;                     \
  Real_type(*QQD)[MQ1][MD1] = (Real_type(*)[MQ1][MD1])sm0;                     \
  Real_type(*QDD)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm1;                     \
  RAJA_TEAM_SHARED Index_type thread_dofs[MD1 * MD1 * MD1];

#define MASS3DPA_ATOMIC_1                                                      \
  Index_type j = dx + mpa_at::D1D * (dy + dz * mpa_at::D1D);                   \
  thread_dofs[j] = ElemToDoF[j + mpa_at::D1D * mpa_at::D1D * mpa_at::D1D * e]; \
  sm_X[dz][dy][dx] =                                                           \
      X[thread_dofs[j]]; // missing dof_map for lexicographical ordering

#define MASS3DPA_ATOMIC_2                                                      \
  sm_B[q][d] = MPAT_B(q, d);                                                   \
  sm_Bt[d][q] = sm_B[q][d];

// flop counts
// 2 * D1D
#define MASS3DPA_ATOMIC_3                                                      \
  Real_type u = 0.0;                                                           \
  for (Index_type dx = 0; dx < mpa_at::D1D; ++dx) {                            \
    u += sm_X[dz][dy][dx] * sm_B[qx][dx];                                      \
  }                                                                            \
  DDQ[dz][dy][qx] = u;

// 2 * D1D
#define MASS3DPA_ATOMIC_4                                                      \
  Real_type u = 0.0;                                                           \
  for (Index_type dy = 0; dy < mpa_at::D1D; ++dy) {                            \
    u += DDQ[dz][dy][qx] * sm_B[qy][dy];                                       \
  }                                                                            \
  DQQ[dz][qy][qx] = u;

// 2 * D1D + 1
#define MASS3DPA_ATOMIC_5                                                      \
  Real_type u = 0.0;                                                           \
  for (Index_type dz = 0; dz < mpa_at::D1D; ++dz) {                            \
    u += DQQ[dz][qy][qx] * sm_B[qz][dz];                                       \
  }                                                                            \
  QQQ[qz][qy][qx] = u * MPAT_D(qx, qy, qz, e);

// 2 * Q1D
#define MASS3DPA_ATOMIC_6                                                      \
  Real_type u = 0.0;                                                           \
  for (Index_type qx = 0; qx < mpa_at::Q1D; ++qx) {                            \
    u += QQQ[qz][qy][qx] * sm_Bt[dx][qx];                                      \
  }                                                                            \
  QQD[qz][qy][dx] = u;

// 2 * Q1D
#define MASS3DPA_ATOMIC_7                                                      \
  Real_type u = 0.0;                                                           \
  for (Index_type qy = 0; qy < mpa_at::Q1D; ++qy) {                            \
    u += QQD[qz][qy][dx] * sm_Bt[dy][qy];                                      \
  }                                                                            \
  QDD[qz][dy][dx] = u;

// 2 * Q1D + 1
#define MASS3DPA_ATOMIC_8                                                      \
  Real_type u = 0.0;                                                           \
  for (Index_type qz = 0; qz < mpa_at::Q1D; ++qz) {                            \
    u += QDD[qz][dy][dx] * sm_Bt[dz][qz];                                      \
  }                                                                            \
  const Index_type j = dx + mpa_at::D1D * (dy + dz * mpa_at::D1D);

#define MASS3DPA_ATOMIC_9(atomicAdd)                                           \
  atomicAdd(Y[thread_dofs[j]], u); // atomic add

//
// CPU SoA variant: batch of FEM_VL consecutive elements [e0, eend) of one
// x-row. Rows are grouped in MPAT_ROW_BLOCK^2 blocks (y,z) that are 4-colored
// like a checkerboard, so threads working on one color share no DOFs.
// Stage 2 (basis load) is reused as is.
//
constexpr RAJA::Index_type MPAT_ROW_BLOCK = 4;

#define MASS3DPA_ATOMIC_0_CPU_SIMD(e0, eend)                                   \
  constexpr Index_type MQ1 = mpa_at::Q1D;                                      \
  constexpr Index_type MD1 = mpa_at::D1D;                                      \
  constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;                          \
  constexpr Index_type VL = FEM_VL;                                            \
  FEM_BATCH_SETUP(e0, eend);                                                   \
  Real_type sm_B[MQ1][MD1];                                                    \
  Real_type sm_Bt[MD1][MQ1];                                                   \
  Real_type sm0[MDQ * MDQ * MDQ][VL];                                          \
  Real_type sm1[MDQ * MDQ * MDQ][VL];                                          \
  Real_type(*sm_X)[MD1][MD1][VL] = (Real_type(*)[MD1][MD1][VL])sm0;            \
  Real_type(*DDQ)[MD1][MQ1][VL] = (Real_type(*)[MD1][MQ1][VL])sm1;             \
  Real_type(*DQQ)[MQ1][MQ1][VL] = (Real_type(*)[MQ1][MQ1][VL])sm0;             \
  Real_type(*QQQ)[MQ1][MQ1][VL] = (Real_type(*)[MQ1][MQ1][VL])sm1;             \
  Real_type(*QQD)[MQ1][MD1][VL] = (Real_type(*)[MQ1][MD1][VL])sm0;             \
  Real_type(*QDD)[MD1][MD1][VL] = (Real_type(*)[MD1][MD1][VL])sm1;             \
  Index_type thread_dofs[MD1 * MD1 * MD1][VL];

// gather AoS -> SoA through ElemToDoF
#define MASS3DPA_ATOMIC_SIMD_1                                                 \
  const Index_type j = dx + mpa_at::D1D * (dy + dz * mpa_at::D1D);             \
  CPU_FOREACH_LANE(l) {                                                        \
    thread_dofs[j][l] =                                                        \
        ElemToDoF[j + mpa_at::D1D * mpa_at::D1D * mpa_at::D1D * le[l]];        \
    sm_X[dz][dy][dx][l] = X[thread_dofs[j][l]];                                \
  }

#define MASS3DPA_ATOMIC_SIMD_3                                                 \
  CPU_FOREACH_LANE(l) {                                                        \
    Real_type u = 0.0;                                                         \
    for (Index_type dx = 0; dx < mpa_at::D1D; ++dx) {                          \
      u += sm_X[dz][dy][dx][l] * sm_B[qx][dx];                                 \
    }                                                                          \
    DDQ[dz][dy][qx][l] = u;                                                    \
  }

#define MASS3DPA_ATOMIC_SIMD_4                                                 \
  CPU_FOREACH_LANE(l) {                                                        \
    Real_type u = 0.0;                                                         \
    for (Index_type dy = 0; dy < mpa_at::D1D; ++dy) {                          \
      u += DDQ[dz][dy][qx][l] * sm_B[qy][dy];                                  \
    }                                                                          \
    DQQ[dz][qy][qx][l] = u;                                                    \
  }

#define MASS3DPA_ATOMIC_SIMD_5                                                 \
  CPU_FOREACH_LANE(l) {                                                        \
    Real_type u = 0.0;                                                         \
    for (Index_type dz = 0; dz < mpa_at::D1D; ++dz) {                          \
      u += DQQ[dz][qy][qx][l] * sm_B[qz][dz];                                  \
    }                                                                          \
    QQQ[qz][qy][qx][l] = u * MPAT_D(qx, qy, qz, le[l]);                        \
  }

#define MASS3DPA_ATOMIC_SIMD_6                                                 \
  CPU_FOREACH_LANE(l) {                                                        \
    Real_type u = 0.0;                                                         \
    for (Index_type qx = 0; qx < mpa_at::Q1D; ++qx) {                          \
      u += QQQ[qz][qy][qx][l] * sm_Bt[dx][qx];                                 \
    }                                                                          \
    QQD[qz][qy][dx][l] = u;                                                    \
  }

#define MASS3DPA_ATOMIC_SIMD_7                                                 \
  CPU_FOREACH_LANE(l) {                                                        \
    Real_type u = 0.0;                                                         \
    for (Index_type qy = 0; qy < mpa_at::Q1D; ++qy) {                          \
      u += QQD[qz][qy][dx][l] * sm_Bt[dy][qy];                                 \
    }                                                                          \
    QDD[qz][dy][dx][l] = u;                                                    \
  }

#define MASS3DPA_ATOMIC_SIMD_8                                                 \
  Real_type r_u[VL];                                                           \
  CPU_FOREACH_LANE(l) {                                                        \
    Real_type u = 0.0;                                                         \
    for (Index_type qz = 0; qz < mpa_at::Q1D; ++qz) {                          \
      u += QDD[qz][dy][dx][l] * sm_Bt[dz][qz];                                 \
    }                                                                          \
    r_u[l] = u;                                                                \
  }                                                                            \
  const Index_type j = dx + mpa_at::D1D * (dy + dz * mpa_at::D1D);

// scatter SoA -> AoS without atomics: neighboring lanes (x-adjacent elements)
// share DOFs, so even and odd lanes are scattered in two conflict-free passes
#define MASS3DPA_ATOMIC_SIMD_9                                                 \
  for (Index_type parity = 0; parity < 2; ++parity) {                          \
    CPU_FOREACH_LANE(l) {                                                      \
      if (l < nel && (l & 1) == parity) {                                      \
        Y[thread_dofs[j][l]] += r_u[l];                                        \
      }                                                                        \
    }                                                                          \
  }

// Row block (by, bz) of the b-th block with color c = (by%2) + 2*(bz%2).
#define MPAT_COLOR_BLOCK(c, b, nby, nbz)                                       \
  const Index_type nby_c = ((nby) - ((c) & 1) + 1) / 2;                        \
  const Index_type by = 2 * ((b) % nby_c) + ((c) & 1);                         \
  const Index_type bz = 2 * ((b) / nby_c) + ((c) >> 1);                        \
  const Index_type ey_end = RAJA_MIN((by + 1) * MPAT_ROW_BLOCK, Ny);           \
  const Index_type ez_end = RAJA_MIN((bz + 1) * MPAT_ROW_BLOCK, Nz);

#if defined(RAJAPERF_FEM_AVX512)
//
// AVX-512 batch of consecutive elements [e0, eend) (at most FEM_VL) of one
// x-row: X gathered through ElemToDoF, D transposed from its contiguous
// per-element blocks, stages 3-8 via fem_simd::mass_pa_apply, and Y updated
// with one masked gather-add-scatter per local dof j. For fixed j the dofs
// of distinct elements are distinct, so no scatter has conflicting lanes;
// tail lanes (replicated elements) are masked off.
//
RAJAPERF_FEM_NOINLINE void mass3dpa_atomic_batch_avx512(
    rajaperf::Real_const_ptr B, rajaperf::Real_const_ptr D,
    rajaperf::Real_const_ptr X, rajaperf::Real_ptr Y,
    const rajaperf::Index_type *ElemToDoF,
    const rajaperf::Index_type e0, const rajaperf::Index_type eend)
{
  using namespace rajaperf::fem_simd;
  using rajaperf::Real_type;
  using rajaperf::Index_type;
  constexpr Index_type D1 = mpa_at::D1D;
  constexpr Index_type Q1 = mpa_at::Q1D;
  constexpr Index_type ND = D1 * D1 * D1;
  static_assert(sizeof(Index_type) == 8, "64-bit dof indices expected");

  const Index_type nel = (eend - e0 < FEM_VL) ? eend - e0 : FEM_VL;
  const vmask lm = lane_mask(nel);

  Real_type b[Q1][D1];
  for (Index_type d = 0; d < D1; ++d) {
    for (Index_type q = 0; q < Q1; ++q) {
      b[q][d] = MPAT_B(q, d);
    }
  }

  Index_type el[8];
  for (Index_type l = 0; l < 8; ++l) {
    el[l] = e0 + ((l < nel) ? l : nel - 1);
  }

  alignas(64) vd dv[Q1 * Q1 * Q1];
  RAJAPERF_FEM_STACK_ALIGN(dv);
  {
    const Real_type *drow[8];
    for (Index_type l = 0; l < 8; ++l) {
      const Index_type e = el[l];
      drow[l] = &MPAT_D(0, 0, 0, e);
    }
    load_transpose<Q1 * Q1 * Q1>(drow, dv);
  }

  const __m512i ebase = _mm512_mullo_epi64(
      _mm512_loadu_si512(el), _mm512_set1_epi64(ND));
  __m512i dof[ND];
  alignas(64) vd x[ND];
  alignas(64) vd y[ND];
  RAJAPERF_FEM_UNROLL
  for (Index_type j = 0; j < ND; ++j) {
    dof[j] = _mm512_i64gather_epi64(
        _mm512_add_epi64(ebase, _mm512_set1_epi64(j)), ElemToDoF, 8);
    x[j] = _mm512_i64gather_pd(dof[j], X, 8);
  }

  mass_pa_apply<D1, Q1>(b, x, dv, y);

  RAJAPERF_FEM_UNROLL
  for (Index_type j = 0; j < ND; ++j) {
    const vd yo = _mm512_mask_i64gather_pd(vzero(), lm, dof[j], Y, 8);
    _mm512_mask_i64scatter_pd(Y, lm, dof[j], vadd(yo, y[j]), 8);
  }
}
#endif

namespace rajaperf {
class RunParams;

namespace apps {

class MASS3DPA_ATOMIC : public KernelBase {
public:
  MASS3DPA_ATOMIC(const RunParams &params);

  ~MASS3DPA_ATOMIC();

  void setSize(Index_type target_size, Index_type target_reps);
  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void defineSeqVariantTunings();
  void defineOpenMPVariantTunings();
  void defineCudaVariantTunings();
  void defineHipVariantTunings();
  void defineSyclVariantTunings();

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);

  template <size_t block_size> void runCudaVariantImpl(VariantID vid);
  template <size_t block_size> void runHipVariantImpl(VariantID vid);
  template <size_t work_group_size> void runSyclVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size =
      mpa_at::Q1D * mpa_at::Q1D * mpa_at::Q1D;
  using gpu_block_sizes_type = integer::list_type<default_gpu_block_size>;

  Real_ptr m_B;
  Real_ptr m_D;
  Real_ptr m_X;
  Real_ptr m_Y;

  Index_type m_Nx;       // zones in x dimension
  Index_type m_Ny;       // zones in y dimension
  Index_type m_Nz;       // zones in z dimension
  Index_type m_P;        // polynomial order
  Index_type m_Tot_Dofs; // total number of dofs

  Index_ptr m_ElemToDoF;

  Index_type m_NE;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
