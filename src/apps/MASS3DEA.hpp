//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Element assembly of a 3D finite element mass matrix
///
/// Based on MFEM's/CEED algorithms.
/// Reference implementation - MFEM-v4.9
/// https://github.com/mfem/mfem/blob/v4.9/fem/integ/bilininteg_mass_kernels.hpp#L1268
/// Kernel uses shared memory which is optimal for orders higher than 2
///
/// for (Index_type e = 0; e < NE; ++e)
///   {
///
///     Real_type s_B[MQ1s][MD1s];
///     Real_type r_B[MQ1r][MD1r];
///
///     Real_type (*l_B)[MD1] = nullptr;
///
///     for(Index_type d=0; d<D1D; ++d) {
///       for(Index_type q=0; q<Q1D; ++q) {
///         s_B[q][d] = B(q,d);
///       }
///     }
///
///     l_B = (Real_type (*)[MD1])s_B;
///
///     Real_type s_D[MQ1][MQ1][MQ1];
///
///     for(Index_type k1=0; k1<Q1D; ++k1) {
///       for(Index_type k2=0; k2<Q1D; ++k2) {
///         for(Index_type k3=0; k3<Q1D; ++k3) {
///           s_D[k1][k2][k3] = D(k1,k2,k3,e);
///         }
///       }
///     }
///
///     for(Index_type i1=0; i1<D1D; ++i1) {
///       for(Index_type i2=0; i2<D1D; ++i2) {
///         for(Index_type i3=0; i3<D1D; ++i3) {
///
///           for (Index_type j1 = 0; j1 < D1D; ++j1) {
///             for (Index_type j2 = 0; j2 < D1D; ++j2) {
///               for (Index_type j3 = 0; j3 < D1D; ++j3) {
///
///                 Real_type val = 0.0;
///                 for (Index_type k1 = 0; k1 < Q1D; ++k1) {
///                   for (Index_type k2 = 0; k2 < Q1D; ++k2) {
///                     for (Index_type k3 = 0; k3 < Q1D; ++k3) {
///
///                       val += l_B[k1][i1] * l_B[k1][j1]
///                         * l_B[k2][i2] * l_B[k2][j2]
///                         * l_B[k3][i3] * l_B[k3][j3]
///                         * s_D[k1][k2][k3];
///                     }
///                   }
///                 }
///
///                 M(i1, i2, i3, j1, j2, j3, e) = val;
///               }
///             }
///           }
///
///         }
///       }
///     }
///
///   } // element loop
///

#ifndef RAJAPerf_Apps_MASS3DEA_HPP
#define RAJAPerf_Apps_MASS3DEA_HPP

#define MASS3DEA_DATA_SETUP                                                    \
  Real_ptr B = m_B;                                                            \
  Real_ptr D = m_D;                                                            \
  Real_ptr M = m_M;                                                            \
  Index_type NE = m_NE;

#include "FEM_MACROS.hpp"
#include "FEM_SIMD.hpp"
#include "common/KernelBase.hpp"

#include "RAJA/RAJA.hpp"

// Number of Dofs/Qpts in 1D
namespace mea {
constexpr RAJA::Index_type D1D = 4;
constexpr RAJA::Index_type Q1D = 5;
} // namespace mea
#define MEA_B(x, y) B[x + mea::Q1D * y]
#define MEA_M(i1, i2, i3, j1, j2, j3, e)                                       \
  M[i1 + mea::D1D *                                                            \
             (i2 + mea::D1D *                                                  \
                       (i3 + mea::D1D *                                        \
                                 (j1 + mea::D1D *                              \
                                           (j2 + mea::D1D *                    \
                                                     (j3 + mea::D1D * e)))))]

#define MEA_D(qx, qy, qz, e)                                                   \
  D[qx + mea::Q1D * qy + mea::Q1D * mea::Q1D * qz +                            \
    mea::Q1D * mea::Q1D * mea::Q1D * e]

#define MASS3DEA_0 RAJA_TEAM_SHARED Real_type s_B[mea::Q1D][mea::D1D];

#define MASS3DEA_0_CPU Real_type s_B[mea::Q1D][mea::D1D];

#define MASS3DEA_1 s_B[q][d] = MEA_B(q, d);

#define MASS3DEA_2 RAJA_TEAM_SHARED Real_type s_D[mea::Q1D][mea::Q1D][mea::Q1D];

#define MASS3DEA_2_CPU Real_type s_D[mea::Q1D][mea::Q1D][mea::Q1D];

#define MASS3DEA_3 s_D[k1][k2][k3] = MEA_D(k1, k2, k3, e);

#define MASS3DEA_4                                                             \
  for (Index_type j1 = 0; j1 < mea::D1D; ++j1) {                               \
    for (Index_type j2 = 0; j2 < mea::D1D; ++j2) {                             \
      for (Index_type j3 = 0; j3 < mea::D1D; ++j3) {                           \
                                                                               \
        Real_type val = 0.0;                                                   \
        for (Index_type k1 = 0; k1 < mea::Q1D; ++k1) {                         \
          for (Index_type k2 = 0; k2 < mea::Q1D; ++k2) {                       \
            for (Index_type k3 = 0; k3 < mea::Q1D; ++k3) {                     \
                                                                               \
              val += s_B[k1][i1] * s_B[k1][j1] * s_B[k2][i2] * s_B[k2][j2] *   \
                     s_B[k3][i3] * s_B[k3][j3] * s_D[k1][k2][k3];              \
            }                                                                  \
          }                                                                    \
        }                                                                      \
        MEA_M(i1, i2, i3, j1, j2, j3, e) = val;                                \
      }                                                                        \
    }                                                                          \
  }

//
// CPU SoA variant: batch of FEM_VL elements starting at e0, element innermost.
// Output is produced one contiguous 64-entry i-block per (j1,j2,j3); i1 gives
// D1D independent accumulators. Stage 1 (basis load) is reused as is.
//
#define MASS3DEA_0_CPU_SIMD                                                    \
  FEM_BATCH_SETUP(e0, NE);                                                     \
  Real_type s_B[mea::Q1D][mea::D1D];

#define MASS3DEA_2_CPU_SIMD Real_type s_D[mea::Q1D][mea::Q1D][mea::Q1D][FEM_VL];

// gather AoS -> SoA
#define MASS3DEA_SIMD_3                                                        \
  CPU_FOREACH_LANE(l) { s_D[k1][k2][k3][l] = MEA_D(k1, k2, k3, le[l]); }

#define MASS3DEA_SIMD_4                                                        \
  Real_type s_M[mea::D1D * mea::D1D * mea::D1D][FEM_VL];                       \
  for (Index_type i3 = 0; i3 < mea::D1D; ++i3) {                               \
    for (Index_type i2 = 0; i2 < mea::D1D; ++i2) {                             \
      Real_type val[mea::D1D][FEM_VL];                                         \
      for (Index_type i1 = 0; i1 < mea::D1D; ++i1) {                           \
        CPU_FOREACH_LANE(l) { val[i1][l] = 0.0; }                              \
      }                                                                        \
      for (Index_type k1 = 0; k1 < mea::Q1D; ++k1) {                           \
        Real_type b1[mea::D1D];                                                \
        for (Index_type i1 = 0; i1 < mea::D1D; ++i1) {                         \
          b1[i1] = s_B[k1][i1] * s_B[k1][j1];                                  \
        }                                                                      \
        for (Index_type k2 = 0; k2 < mea::Q1D; ++k2) {                         \
          Real_type b12[mea::D1D];                                             \
          for (Index_type i1 = 0; i1 < mea::D1D; ++i1) {                       \
            b12[i1] = b1[i1] * s_B[k2][i2] * s_B[k2][j2];                      \
          }                                                                    \
          for (Index_type k3 = 0; k3 < mea::Q1D; ++k3) {                       \
            for (Index_type i1 = 0; i1 < mea::D1D; ++i1) {                     \
              const Real_type b = b12[i1] * s_B[k3][i3] * s_B[k3][j3];         \
              CPU_FOREACH_LANE(l) { val[i1][l] += b * s_D[k1][k2][k3][l]; }    \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
      for (Index_type i1 = 0; i1 < mea::D1D; ++i1) {                           \
        CPU_FOREACH_LANE(l) {                                                  \
          s_M[i1 + mea::D1D * (i2 + mea::D1D * i3)][l] = val[i1][l];           \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  /* scatter SoA -> AoS: one contiguous i-block per valid element */           \
  for (Index_type l = 0; l < nel; ++l) {                                       \
    const Index_type e = e0 + l;                                               \
    Real_ptr Mblk = &MEA_M(0, 0, 0, j1, j2, j3, e);                            \
    FEM_OMP_SIMD                                                               \
    for (Index_type i = 0; i < mea::D1D * mea::D1D * mea::D1D; ++i) {          \
      Mblk[i] = s_M[i][l];                                                     \
    }                                                                          \
  }

#if defined(RAJAPERF_FEM_AVX512)
//
// AVX-512 version of MASS3DEA_SIMD_4 for one (batch, j1, j2, j3) block:
//   M[i][e] = sum_k (b1[k1][i1] b2[k2][i2]) (b3[k3][i3] D[k][e]),
//   bn[k][i] = B[k][i] B[k][jn].
// t = b3 D is formed once per (i3, k); the i1 x (2 i2) = 8 register
// accumulators each take one FMA with a broadcast coefficient, and
// correspond to one contiguous 8-entry chunk of M, written back with an 8x8
// transpose.
//
RAJAPERF_FEM_NOINLINE void mass3dea_block_avx512(
    const rajaperf::Real_type (&s_B)[mea::Q1D][mea::D1D],
    const rajaperf::Real_type (&s_D)[mea::Q1D][mea::Q1D][mea::Q1D][FEM_VL],
    const rajaperf::Index_type j1, const rajaperf::Index_type j2,
    const rajaperf::Index_type j3, rajaperf::Real_ptr M,
    const rajaperf::Index_type e0, const rajaperf::Index_type nel)
{
  using namespace rajaperf::fem_simd;
  using rajaperf::Real_type;
  using rajaperf::Index_type;
  constexpr Index_type D1 = mea::D1D;
  constexpr Index_type Q1 = mea::Q1D;
  static_assert(D1 == 4, "i-chunk mapping assumes D1D == 4");

  // c12[ip][k1][k2][i1 + 4*ii] = b1[k1][i1] * b2[k2][2*ip + ii]
  alignas(64) Real_type c12[2][Q1][Q1][8];
  for (Index_type k1 = 0; k1 < Q1; ++k1) {
    for (Index_type k2 = 0; k2 < Q1; ++k2) {
      for (Index_type i2 = 0; i2 < D1; ++i2) {
        const Real_type b2 = s_B[k2][i2] * s_B[k2][j2];
        for (Index_type i1 = 0; i1 < D1; ++i1) {
          c12[i2 / 2][k1][k2][i1 + D1 * (i2 % 2)] =
              s_B[k1][i1] * s_B[k1][j1] * b2;
        }
      }
    }
  }

  Real_type *row[8];
  for (Index_type l = 0; l < 8; ++l) {
    const Index_type e = e0 + ((l < nel) ? l : nel - 1);
    row[l] = &MEA_M(0, 0, 0, j1, j2, j3, e);
  }

  alignas(64) vd T[Q1][Q1][Q1];
  RAJAPERF_FEM_STACK_ALIGN(T);
  for (Index_type i3 = 0; i3 < D1; ++i3) {
    RAJAPERF_FEM_UNROLL
    for (Index_type k3 = 0; k3 < Q1; ++k3) {
      const vd b3 = vset1(s_B[k3][i3] * s_B[k3][j3]);
      for (Index_type k1 = 0; k1 < Q1; ++k1) {
        RAJAPERF_FEM_UNROLL
        for (Index_type k2 = 0; k2 < Q1; ++k2) {
          T[k1][k2][k3] = vmul(b3, vload(s_D[k1][k2][k3]));
        }
      }
    }
    for (Index_type ip = 0; ip < 2; ++ip) {
      vd acc[8];
      RAJAPERF_FEM_UNROLL
      for (Index_type r = 0; r < 8; ++r) {
        acc[r] = vzero();
      }
      for (Index_type k1 = 0; k1 < Q1; ++k1) {
        for (Index_type k2 = 0; k2 < Q1; ++k2) {
          vd c[8];
          RAJAPERF_FEM_UNROLL
          for (Index_type r = 0; r < 8; ++r) {
            c[r] = vset1(c12[ip][k1][k2][r]);
          }
          RAJAPERF_FEM_UNROLL
          for (Index_type k3 = 0; k3 < Q1; ++k3) {
            const vd t = T[k1][k2][k3];
            RAJAPERF_FEM_UNROLL
            for (Index_type r = 0; r < 8; ++r) {
              acc[r] = vfma(c[r], t, acc[r]);
            }
          }
        }
      }
      // acc[i1 + 4*ii] holds i = i1 + 4*(2*ip + ii) + 16*i3 for 8 lanes
      transpose8(acc);
      const Index_type off = D1 * D1 * i3 + 8 * ip;
      for (Index_type l = 0; l < nel; ++l) {
        vstore(row[l] + off, acc[l]);
      }
    }
  }
}

#define MASS3DEA_CPU_SIMD_4 mass3dea_block_avx512(s_B, s_D, j1, j2, j3, M, e0, nel);
#else
#define MASS3DEA_CPU_SIMD_4 MASS3DEA_SIMD_4
#endif

//
// Sum-factorized element assembly for the batch [e0, e0+FEM_VL) and one j3:
//   M(i,j) = sum_k1 b(k1,i1,j1) sum_k2 b(k2,i2,j2) sum_k3 b(k3,i3,j3) D(k),
//   b(k,i,j) = B(k,i) B(k,j): O(D^2Q^3 + D^4Q^2 + D^6Q) instead of O(D^6Q^3).
// Each (j1,j2) block of 64 entries per element is written with streaming
// stores when STREAM (and AVX-512) is available and M is 64B aligned.
// SEQ (with STREAM): the batch's j3 slices (VL x 1024 entries) are staged
// and each element's contiguous 8 KB slice is streamed in one go, so a thread
// has one sequential write stream instead of VL interleaved ones.
//
template <bool STREAM, bool SEQ = false>
RAJAPERF_FEM_NOINLINE void mass3dea_sumfact_block(
    rajaperf::Real_const_ptr B, rajaperf::Real_const_ptr D, rajaperf::Real_ptr M,
    const rajaperf::Index_type NE, const rajaperf::Index_type e0,
    const rajaperf::Index_type j3, const bool nt_ok)
{
  using rajaperf::Real_type;
  using rajaperf::Index_type;
  constexpr Index_type D1 = mea::D1D;
  constexpr Index_type Q1 = mea::Q1D;
  constexpr Index_type VL = FEM_VL;
  constexpr Index_type NI = D1 * D1 * D1;

  FEM_BATCH_SETUP(e0, NE);

  Real_type bb[Q1][D1][D1];
  for (Index_type k = 0; k < Q1; ++k) {
    for (Index_type i = 0; i < D1; ++i) {
      for (Index_type j = 0; j < D1; ++j) {
        bb[k][i][j] = MEA_B(k, i) * MEA_B(k, j);
      }
    }
  }

  alignas(64) Real_type s_D[Q1][Q1][Q1][VL];
  for (Index_type k1 = 0; k1 < Q1; ++k1) {
    for (Index_type k2 = 0; k2 < Q1; ++k2) {
      for (Index_type k3 = 0; k3 < Q1; ++k3) {
        CPU_FOREACH_LANE(l) { s_D[k1][k2][k3][l] = MEA_D(k1, k2, k3, le[l]); }
      }
    }
  }

  // T3[i3][k1][k2] = sum_k3 b(k3,i3,j3) D[k1][k2][k3]
  alignas(64) Real_type T3[D1][Q1][Q1][VL];
  for (Index_type i3 = 0; i3 < D1; ++i3) {
    for (Index_type k1 = 0; k1 < Q1; ++k1) {
      for (Index_type k2 = 0; k2 < Q1; ++k2) {
        CPU_FOREACH_LANE(l) {
          Real_type s = 0.0;
          for (Index_type k3 = 0; k3 < Q1; ++k3) {
            s += bb[k3][i3][j3] * s_D[k1][k2][k3][l];
          }
          T3[i3][k1][k2][l] = s;
        }
      }
    }
  }

  // T2[j2][i2][i3][k1] = sum_k2 b(k2,i2,j2) T3[i3][k1][k2]
  alignas(64) Real_type T2[D1][D1][D1][Q1][VL];
  for (Index_type j2 = 0; j2 < D1; ++j2) {
    for (Index_type i2 = 0; i2 < D1; ++i2) {
      for (Index_type i3 = 0; i3 < D1; ++i3) {
        for (Index_type k1 = 0; k1 < Q1; ++k1) {
          CPU_FOREACH_LANE(l) {
            Real_type s = 0.0;
            for (Index_type k2 = 0; k2 < Q1; ++k2) {
              s += bb[k2][i2][j2] * T3[i3][k1][k2][l];
            }
            T2[j2][i2][i3][k1][l] = s;
          }
        }
      }
    }
  }

  Real_type *row[VL];
  for (Index_type l = 0; l < VL; ++l) {
    const Index_type e = le[l];
    row[l] = &MEA_M(0, 0, 0, 0, 0, j3, e);
  }

  alignas(64) Real_type blk[NI][VL];
  RAJAPERF_FEM_STACK_ALIGN(blk);
#if defined(RAJAPERF_FEM_AVX512)
  constexpr bool stage = STREAM && SEQ;
  alignas(64) Real_type stg[stage ? VL : 1][stage ? NI * D1 * D1 : 1];
  if constexpr (stage) { RAJAPERF_FEM_STACK_ALIGN(stg); }
#endif
  for (Index_type j2 = 0; j2 < D1; ++j2) {
    for (Index_type j1 = 0; j1 < D1; ++j1) {
      // blk[i1 + D1*(i2 + D1*i3)] = sum_k1 b(k1,i1,j1) T2[j2][i2][i3][k1]
      for (Index_type i3 = 0; i3 < D1; ++i3) {
        for (Index_type i2 = 0; i2 < D1; ++i2) {
          for (Index_type i1 = 0; i1 < D1; ++i1) {
            CPU_FOREACH_LANE(l) {
              Real_type s = 0.0;
              for (Index_type k1 = 0; k1 < Q1; ++k1) {
                s += bb[k1][i1][j1] * T2[j2][i2][i3][k1][l];
              }
              blk[i1 + D1 * (i2 + D1 * i3)][l] = s;
            }
          }
        }
      }
      const Index_type off = NI * (j1 + D1 * j2);
#if defined(RAJAPERF_FEM_AVX512)
      using namespace rajaperf::fem_simd;
      const bool nt = STREAM && nt_ok && nel == VL;
      for (Index_type c = 0; c < NI; c += 8) {
        vd r[8];
        for (Index_type k = 0; k < 8; ++k) {
          r[k] = _mm512_load_pd(blk[c + k]);
        }
        transpose8(r);
        if constexpr (stage) {
          if (nt) {
            for (Index_type l = 0; l < VL; ++l) {
              _mm512_store_pd(&stg[l][off + c], r[l]);
            }
            continue;
          }
        }
        if (nt) {
          for (Index_type l = 0; l < VL; ++l) {
            _mm512_stream_pd(row[l] + off + c, r[l]);
          }
        } else {
          for (Index_type l = 0; l < nel; ++l) {
            vstore(row[l] + off + c, r[l]);
          }
        }
      }
#else
      RAJA_UNUSED_VAR(nt_ok);
      for (Index_type l = 0; l < nel; ++l) {
        for (Index_type i = 0; i < NI; ++i) {
          row[l][off + i] = blk[i][l];
        }
      }
#endif
    }
  }
#if defined(RAJAPERF_FEM_AVX512)
  if constexpr (stage) {
    if (nt_ok && nel == VL) {
      for (Index_type l = 0; l < VL; ++l) {
        rajaperf::fem_simd::stream_copy(row[l], stg[l], NI * D1 * D1);
      }
    }
  }
#endif
}

namespace rajaperf {
class RunParams;

namespace apps {

class MASS3DEA : public KernelBase {
public:
  MASS3DEA(const RunParams &params);

  ~MASS3DEA();

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
  static const size_t default_gpu_block_size = mea::D1D * mea::D1D * mea::D1D;
  using gpu_block_sizes_type = integer::list_type<default_gpu_block_size>;

  Real_ptr m_B;
  Real_ptr m_D;
  Real_ptr m_M;

  Index_type m_NE;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
