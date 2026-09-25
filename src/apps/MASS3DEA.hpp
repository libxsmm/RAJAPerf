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
