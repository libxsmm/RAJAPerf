//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Element-wise action of a 3D finite element mass matrix via partial
/// assembly and sum factorization on a block vector
///
/// for (Index_type e = 0; e < NE; ++e) {
///
///  Real_type B[MQ1][MD1];
///
///  Real_type sm0[MDQ * MDQ * MDQ];
///  Real_type sm1[MDQ * MDQ * MDQ];
///  Real_type(*X)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm0;
///  Real_type(*DDQ)[MD1][MQ1] = (Real_type(*)[MD1][MQ1])sm1;
///  Real_type(*DQQ)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])sm0;
///  Real_type(*QQQ)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])sm1;
///  Real_type(*QQD)[MQ1][MD1] = (Real_type(*)[MQ1][MD1])sm0;
///  Real_type(*QDD)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm1;
///
///  for (Index_type d = 0; d < mvpa::D1D; ++d) {
///    for (Index_type q = 0; q < mvpa::Q1D; ++q) {
///      Real_type basis = b(q, d);
///      B[q][d] = basis;
///      Bt[d][q] = basis;
///    }
///  }
///
///  for (Index_type c = 0; c < 3; ++c) {
///
///    for (Index_type dz = 0; dz < mvpa::D1D; ++dz) {
///      for (Index_type dy = 0; dy < mvpa::D1D; ++dy) {
///        for (Index_type dx = 0; dx < mvpa::D1D; ++dx) {
///
///          smX[dz][dy][dx] = mvpaX_(dx, dy, dz, c, e);
///        }
///      }
///    }
///
///    for (Index_type dz = 0; dz < mvpa::D1D; ++dz) {
///      for (Index_type dy = 0; dy < mvpa::D1D; ++dy) {
///        for (Index_type qx = 0; qx < mvpa::Q1D; ++qx) {
///
///          Real_type u = 0.0;
///          for (Index_type dx = 0; dx < mvpa::D1D; ++dx) {
///            u += X[dz][dy][dx] * B[qx][dx];
///          }
///          DDQ[dz][dy][qx] = u;
///        }
///      }
///    }
///
///    for (Index_type dz = 0; dz < mvpa::D1D; ++dz) {
///      for (Index_type qy = 0; qy < mvpa::Q1D; ++qy) {
///        for (Index_type qx = 0; qx < mvpa::Q1D; ++qx) {
///
///          Real_type u = 0.0;
///          for (Index_type dy = 0; dy < mvpa::D1D; ++dy) {
///            u += DDQ[dz][dy][qx] * B[qy][dy];
///          }
///          DQQ[dz][qy][qx] = u;
///        }
///      }
///    }
///
///    for (Index_type qz = 0; qz < mvpa::Q1D; ++qz) {
///      for (Index_type qy = 0; qy < mvpa::Q1D; ++qy) {
///        for (Index_type qx = 0; qx < mvpa::Q1D; ++qx) {
///
///          Real_type u = 0.0;
///          for (Index_type dz = 0; dz < mvpa::D1D; ++dz) {
///            u += DQQ[dz][qy][qx] * B[qz][dz];
///          }
///          QQQ[qz][qy][qx] = u * D(qx, qy, qz, e);
///        }
///      }
///    }
///
///    for (Index_type qz = 0; qz < mvpa::Q1D; ++qz) {
///      for (Index_type qy = 0; qy < mvpa::Q1D; ++qy) {
///        for (Index_type dx = 0; dx < mvpa::D1D; ++dx) {
///
///          Real_type u = 0.0;
///          for (Index_type qx = 0; qx < mvpa::Q1D; ++qx) {
///            u += QQQ[qz][qy][qx] * Bt[dx][qx];
///          }
///          QQD[qz][qy][dx] = u;
///        }
///      }
///    }
///
///    for (Index_type qz = 0; qz < mvpa::Q1D; ++qz) {
///      for (Index_type dy = 0; dy < mvpa::D1D; ++dy) {
///        for (Index_type dx = 0; dx < mvpa::D1D; ++dx) {
///
///          Real_type u = 0.0;
///          for (Index_type qy = 0; qy < mvpa::Q1D; ++qy) {
///            u += QQD[qz][qy][dx] * Bt[dy][qy];
///          }
///          QDD[qz][dy][dx] = u;
///        }
///      }
///    }
///
///    for (Index_type dz = 0; dz < mvpa::D1D; ++dz) {
///      for (Index_type dy = 0; dy < mvpa::D1D; ++dy) {
///        for (Index_type dx = 0; dx < mvpa::D1D; ++dx) {
///
///          Real_type u = 0.0;
///          for (Index_type qz = 0; qz < mvpa::Q1D; ++qz) {
///            u += QDD[qz][dy][dx] * Bt[dz][qz];
///          }
///          mvpaY_(dx, dy, dz, c, e) = u;
///        }
///      }
///    }
///
///  } // element loop
///

#ifndef RAJAPerf_Apps_MASSVEC3DPA_HPP
#define RAJAPerf_Apps_MASSVEC3DPA_HPP

#define MASSVEC3DPA_DATA_SETUP                                                 \
  Real_ptr B = m_B;                                                            \
  Real_ptr D = m_D;                                                            \
  Real_ptr X = m_X;                                                            \
  Real_ptr Y = m_Y;                                                            \
  Index_type NE = m_NE;

#include "FEM_MACROS.hpp"
#include "FEM_SIMD.hpp"
#include "common/KernelBase.hpp"

#include "RAJA/RAJA.hpp"

// Number of Dofs/Qpts in 1D
namespace mvpa {
constexpr RAJA::Index_type D1D = 3;
constexpr RAJA::Index_type Q1D = 4;
constexpr RAJA::Index_type DIM = 3;
} // namespace mvpa
#define MVPA_B(x, y) B[x + mvpa::Q1D * y]
#define MVPA_X(dx, dy, dz, c, e)                                               \
  X[dx + mvpa::D1D * dy + mvpa::D1D * mvpa::D1D * dz +                         \
    mvpa::D1D * mvpa::D1D * mvpa::D1D * c +                                    \
    mvpa::D1D * mvpa::D1D * mvpa::D1D * mvpa::DIM * e]
#define MVPA_Y(dx, dy, dz, c, e)                                               \
  Y[dx + mvpa::D1D * dy + mvpa::D1D * mvpa::D1D * dz +                         \
    mvpa::D1D * mvpa::D1D * mvpa::D1D * c +                                    \
    mvpa::D1D * mvpa::D1D * mvpa::D1D * mvpa::DIM * e]
#define MVPA_D(qx, qy, qz, e)                                                  \
  D[qx + mvpa::Q1D * qy + mvpa::Q1D * mvpa::Q1D * qz +                         \
    mvpa::Q1D * mvpa::Q1D * mvpa::Q1D * e]

#define MASSVEC3DPA_0_CPU                                                      \
  constexpr Index_type MQ1 = mvpa::Q1D;                                        \
  constexpr Index_type MD1 = mvpa::D1D;                                        \
  constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;                          \
  /*RAJA_TEAM_SHARED*/ Real_type smB[MQ1][MD1];                                \
  /*RAJA_TEAM_SHARED*/ Real_type smBt[MD1][MQ1];                               \
  /*RAJA_TEAM_SHARED*/ Real_type sm0[MDQ * MDQ * MDQ];                         \
  /*RAJA_TEAM_SHARED*/ Real_type sm1[MDQ * MDQ * MDQ];                         \
  Real_type(*smX)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm0;                     \
  Real_type(*DDQ)[MD1][MQ1] = (Real_type(*)[MD1][MQ1])sm1;                     \
  Real_type(*DQQ)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])sm0;                     \
  Real_type(*QQQ)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])sm1;                     \
  Real_type(*QQD)[MQ1][MD1] = (Real_type(*)[MQ1][MD1])sm0;                     \
  Real_type(*QDD)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm1;

#define MASSVEC3DPA_0_GPU                                                      \
  constexpr Index_type MQ1 = mvpa::Q1D;                                        \
  constexpr Index_type MD1 = mvpa::D1D;                                        \
  constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;                          \
  RAJA_TEAM_SHARED Real_type smB[MQ1][MD1];                                    \
  RAJA_TEAM_SHARED Real_type smBt[MD1][MQ1];                                   \
  RAJA_TEAM_SHARED Real_type sm0[MDQ * MDQ * MDQ];                             \
  RAJA_TEAM_SHARED Real_type sm1[MDQ * MDQ * MDQ];                             \
  Real_type(*smX)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm0;                     \
  Real_type(*DDQ)[MD1][MQ1] = (Real_type(*)[MD1][MQ1])sm1;                     \
  Real_type(*DQQ)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])sm0;                     \
  Real_type(*QQQ)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])sm1;                     \
  Real_type(*QQD)[MQ1][MD1] = (Real_type(*)[MQ1][MD1])sm0;                     \
  Real_type(*QDD)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm1;

#define MASSVEC3DPA_1                                                          \
  Real_type r_smB = MVPA_B(q, d);                                              \
  smB[q][d] = r_smB;                                                           \
  smBt[d][q] = r_smB;

#define MASSVEC3DPA_2 smX[dz][dy][dx] = MVPA_X(dx, dy, dz, c, e);

// 2 * mvpa::D1D * mvpa::Q1D * mvpa::D1D * mvpa::D1D
#define MASSVEC3DPA_3                                                          \
  Real_type u = 0.0;                                                           \
  for (Index_type dx = 0; dx < mvpa::D1D; ++dx) {                              \
    u += smX[dz][dy][dx] * smB[qx][dx];                                        \
  }                                                                            \
  DDQ[dz][dy][qx] = u;

// 2 * mvpa::D1D * mvpa::Q1D * mvpa::Q1D * mvpa::D1D
#define MASSVEC3DPA_4                                                          \
  Real_type u = 0.0;                                                           \
  for (Index_type dy = 0; dy < mvpa::D1D; ++dy) {                              \
    u += DDQ[dz][dy][qx] * smB[qy][dy];                                        \
  }                                                                            \
  DQQ[dz][qy][qx] = u;

// 2 * mvpa::D1D * mvpa::Q1D * mvpa::Q1D * mvpa::Q1D + mvpa::Q1D * mvpa::Q1D *
// mvpa::Q1D
#define MASSVEC3DPA_5                                                          \
  Real_type u = 0.0;                                                           \
  for (Index_type dz = 0; dz < mvpa::D1D; ++dz) {                              \
    u += DQQ[dz][qy][qx] * smB[qz][dz];                                        \
  }                                                                            \
  QQQ[qz][qy][qx] = u * MVPA_D(qx, qy, qz, e);

// 2 * mvpa::Q1D * mvpa::D1D * mvpa::Q1D * mvpa::Q1D
#define MASSVEC3DPA_6                                                          \
  Real_type u = 0.0;                                                           \
  for (Index_type qx = 0; qx < mvpa::Q1D; ++qx) {                              \
    u += QQQ[qz][qy][qx] * smBt[dx][qx];                                       \
  }                                                                            \
  QQD[qz][qy][dx] = u;

// 2 * mvpa::Q1D * mvpa::D1D * mvpa::D1D * mvpa::Q1D
#define MASSVEC3DPA_7                                                          \
  Real_type u = 0.0;                                                           \
  for (Index_type qy = 0; qy < mvpa::Q1D; ++qy) {                              \
    u += QQD[qz][qy][dx] * smBt[dy][qy];                                       \
  }                                                                            \
  QDD[qz][dy][dx] = u;

// 2 * mvpa::Q1D * mvpa::D1D * mvpa::D1D * mvpa::D1D
#define MASSVEC3DPA_8                                                          \
  Real_type u = 0.0;                                                           \
  for (Index_type qz = 0; qz < mvpa::Q1D; ++qz) {                              \
    u += QDD[qz][dy][dx] * smBt[dz][qz];                                       \
  }                                                                            \
  MVPA_Y(dx, dy, dz, c, e) = u;

//
// CPU SoA variant: batch of FEM_VL elements starting at e0, element innermost.
// Stage 1 (basis load) is element independent and reused as is.
//
#define MASSVEC3DPA_0_CPU_SIMD                                                 \
  constexpr Index_type MQ1 = mvpa::Q1D;                                        \
  constexpr Index_type MD1 = mvpa::D1D;                                        \
  constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;                          \
  constexpr Index_type VL = FEM_VL;                                            \
  FEM_BATCH_SETUP(e0, NE);                                                     \
  Real_type smB[MQ1][MD1];                                                     \
  Real_type smBt[MD1][MQ1];                                                    \
  Real_type sm0[MDQ * MDQ * MDQ][VL];                                          \
  Real_type sm1[MDQ * MDQ * MDQ][VL];                                          \
  Real_type(*smX)[MD1][MD1][VL] = (Real_type(*)[MD1][MD1][VL])sm0;             \
  Real_type(*DDQ)[MD1][MQ1][VL] = (Real_type(*)[MD1][MQ1][VL])sm1;             \
  Real_type(*DQQ)[MQ1][MQ1][VL] = (Real_type(*)[MQ1][MQ1][VL])sm0;             \
  Real_type(*QQQ)[MQ1][MQ1][VL] = (Real_type(*)[MQ1][MQ1][VL])sm1;             \
  Real_type(*QQD)[MQ1][MD1][VL] = (Real_type(*)[MQ1][MD1][VL])sm0;             \
  Real_type(*QDD)[MD1][MD1][VL] = (Real_type(*)[MD1][MD1][VL])sm1;

// gather AoS -> SoA
#define MASSVEC3DPA_SIMD_2                                                     \
  CPU_FOREACH_LANE(l) {                                                        \
    smX[dz][dy][dx][l] = MVPA_X(dx, dy, dz, c, le[l]);                         \
  }

#define MASSVEC3DPA_SIMD_3                                                     \
  CPU_FOREACH_LANE(l) {                                                        \
    Real_type u = 0.0;                                                         \
    for (Index_type dx = 0; dx < mvpa::D1D; ++dx) {                            \
      u += smX[dz][dy][dx][l] * smB[qx][dx];                                   \
    }                                                                          \
    DDQ[dz][dy][qx][l] = u;                                                    \
  }

#define MASSVEC3DPA_SIMD_4                                                     \
  CPU_FOREACH_LANE(l) {                                                        \
    Real_type u = 0.0;                                                         \
    for (Index_type dy = 0; dy < mvpa::D1D; ++dy) {                            \
      u += DDQ[dz][dy][qx][l] * smB[qy][dy];                                   \
    }                                                                          \
    DQQ[dz][qy][qx][l] = u;                                                    \
  }

#define MASSVEC3DPA_SIMD_5                                                     \
  CPU_FOREACH_LANE(l) {                                                        \
    Real_type u = 0.0;                                                         \
    for (Index_type dz = 0; dz < mvpa::D1D; ++dz) {                            \
      u += DQQ[dz][qy][qx][l] * smB[qz][dz];                                   \
    }                                                                          \
    QQQ[qz][qy][qx][l] = u * MVPA_D(qx, qy, qz, le[l]);                        \
  }

#define MASSVEC3DPA_SIMD_6                                                     \
  CPU_FOREACH_LANE(l) {                                                        \
    Real_type u = 0.0;                                                         \
    for (Index_type qx = 0; qx < mvpa::Q1D; ++qx) {                            \
      u += QQQ[qz][qy][qx][l] * smBt[dx][qx];                                  \
    }                                                                          \
    QQD[qz][qy][dx][l] = u;                                                    \
  }

#define MASSVEC3DPA_SIMD_7                                                     \
  CPU_FOREACH_LANE(l) {                                                        \
    Real_type u = 0.0;                                                         \
    for (Index_type qy = 0; qy < mvpa::Q1D; ++qy) {                            \
      u += QQD[qz][qy][dx][l] * smBt[dy][qy];                                  \
    }                                                                          \
    QDD[qz][dy][dx][l] = u;                                                    \
  }

// scatter SoA -> AoS (valid lanes only)
#define MASSVEC3DPA_SIMD_8                                                     \
  CPU_FOREACH_LANE(l) {                                                        \
    Real_type u = 0.0;                                                         \
    for (Index_type qz = 0; qz < mvpa::Q1D; ++qz) {                            \
      u += QDD[qz][dy][dx][l] * smBt[dz][qz];                                  \
    }                                                                          \
    if (l < nel) {                                                             \
      const Index_type e = e0 + l;                                             \
      MVPA_Y(dx, dy, dz, c, e) = u;                                            \
    }                                                                          \
  }

#if defined(RAJAPERF_FEM_AVX512)
//
// AVX-512 batch of FEM_VL elements [e0, e0+nel): AoS <-> SoA through 8x8
// in-register transposes of each element's contiguous D, X and Y blocks
// instead of gathers/scatters; stages 3-8 via fem_simd::mass_pa_apply.
//
template <bool STREAM = false>
RAJAPERF_FEM_NOINLINE void massvec3dpa_batch_avx512(
    rajaperf::Real_const_ptr B, rajaperf::Real_const_ptr D,
    rajaperf::Real_const_ptr X, rajaperf::Real_ptr Y,
    const rajaperf::Index_type e0, const rajaperf::Index_type nel)
{
  using namespace rajaperf::fem_simd;
  using rajaperf::Real_type;
  using rajaperf::Index_type;
  constexpr Index_type D1 = mvpa::D1D;
  constexpr Index_type Q1 = mvpa::Q1D;

  Real_type b[Q1][D1];
  for (Index_type d = 0; d < D1; ++d) {
    for (Index_type q = 0; q < Q1; ++q) {
      b[q][d] = MVPA_B(q, d);
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
      drow[l] = &MVPA_D(0, 0, 0, e);
    }
    load_transpose<Q1 * Q1 * Q1>(drow, dv);
  }

  // a full batch's Y is one contiguous block of 8*81 doubles (64B aligned when
  // Y is): stage it in L1 and write it with streaming stores
  constexpr Index_type NY = mvpa::DIM * D1 * D1 * D1;
  Real_type *const ybatch = &Y[NY * e0];
  const bool nt = STREAM && nel == 8 && is_aligned64(ybatch);
  alignas(64) Real_type ybuf[8 * NY];

  for (Index_type c = 0; c < mvpa::DIM; ++c) {
    const Real_type *xrow[8];
    Real_type *yrow[8];
    for (Index_type l = 0; l < 8; ++l) {
      const Index_type e = el[l];
      xrow[l] = &MVPA_X(0, 0, 0, c, e);
      yrow[l] = nt ? ybuf + NY * l + D1 * D1 * D1 * c : &MVPA_Y(0, 0, 0, c, e);
    }
    alignas(64) vd x[D1 * D1 * D1];
    alignas(64) vd y[D1 * D1 * D1];
    load_transpose<D1 * D1 * D1>(xrow, x);
    mass_pa_apply<D1, Q1>(b, x, dv, y);
    transpose_store<D1 * D1 * D1>(yrow, nel, y);
  }
  if (nt) {
    stream_copy(ybatch, ybuf, 8 * NY);
  }
}

#define MASSVEC3DPA_CPU_SIMD_BATCH                                             \
  massvec3dpa_batch_avx512(B, D, X, Y, e0, (NE - e0 < FEM_VL) ? NE - e0 : FEM_VL);
#endif

namespace rajaperf {
class RunParams;

namespace apps {

class MASSVEC3DPA : public KernelBase {
public:
  MASSVEC3DPA(const RunParams &params);

  ~MASSVEC3DPA();

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
  template <size_t tune_idx>
  void runOpenMPVariant(VariantID vid);

  template <size_t block_size, size_t tune_idx>
  void runCudaVariantImpl(VariantID vid);
  template <size_t block_size, size_t tune_idx>
  void runHipVariantImpl(VariantID vid);
  template <size_t work_group_size> void runSyclVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size =
      mvpa::Q1D * mvpa::Q1D * mvpa::Q1D;
  using gpu_block_sizes_type = integer::list_type<default_gpu_block_size>;

  Real_ptr m_B;
  Real_ptr m_D;
  Real_ptr m_X;
  Real_ptr m_Y;

  Index_type m_NE;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
