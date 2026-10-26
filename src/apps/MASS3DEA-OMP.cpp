//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// Uncomment to add compiler directives loop unrolling
//#define USE_RAJAPERF_UNROLL

#include "MASS3DEA.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {


// tune_idx 0: reference algorithm; 1: sum factorization; 2: sum factorization
// with streaming stores of M; 3: 2 with each element's slice of M staged and
// streamed contiguously
template <size_t tune_idx>
void MASS3DEA::runOpenMPVariant(VariantID vid) {

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  MASS3DEA_DATA_SETUP;

  if constexpr (tune_idx > 0) {

  constexpr bool stream = (tune_idx == 2 || tune_idx == 3);
  constexpr bool seq = (tune_idx == 3);
  const bool nt_ok = fem_simd::is_aligned64(M);
  const Index_type nbe = (NE + FEM_VL - 1) / FEM_VL;

  switch (vid) {

  case Base_OpenMP: {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      #pragma omp parallel
      {
        #pragma omp for collapse(2) nowait
        for (Index_type b = 0; b < nbe; ++b) {
          for (Index_type j3 = 0; j3 < mea::D1D; ++j3) {
            mass3dea_sumfact_block<stream, seq>(B, D, M, NE, b * FEM_VL, j3, nt_ok);
          }
        }
        if (stream) { fem_simd::stream_fence(); }
      }

    }
    stopTimer();

    break;
  }

  case RAJA_OpenMP: {

    auto res{getHostResource()};

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RAJA::region<RAJA::omp_parallel_region>( [=]() {
        RAJA::forall< RAJA::omp_for_nowait_static_exec< > >( res,
          RAJA::RangeSegment(0, nbe * mea::D1D), [=](Index_type ej) {
            mass3dea_sumfact_block<stream, seq>(B, D, M, NE, (ej / mea::D1D) * FEM_VL,
                                           ej % mea::D1D, nt_ok);
          });
        if (stream) { fem_simd::stream_fence(); }
      });

    }
    stopTimer();

    break;
  }

  default:
    getCout() << "\n MASS3DEA : Unknown OpenMP variant id = " << vid
              << std::endl;
  }

  } else {

  switch (vid) {

  case Base_OpenMP: {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

RP_CALI_SUBKERNEL_BEGIN("MASS3DEA_1");
      // (batch, j-block) pairs keep all threads busy for small NE
#pragma omp parallel for collapse(4)
      for (Index_type e0 = 0; e0 < NE; e0 += FEM_VL) {
      for (Index_type j3 = 0; j3 < mea::D1D; ++j3) {
      for (Index_type j2 = 0; j2 < mea::D1D; ++j2) {
      for (Index_type j1 = 0; j1 < mea::D1D; ++j1) {

        MASS3DEA_0_CPU_SIMD

        CPU_FOREACH(d, x, mea::D1D) {
          CPU_FOREACH(q, y, mea::Q1D) {
            MASS3DEA_1
          }
        }

        MASS3DEA_2_CPU_SIMD

        CPU_FOREACH(k1, x, mea::Q1D) {
          CPU_FOREACH(k2, y, mea::Q1D) {
            CPU_FOREACH(k3, z, mea::Q1D) {
              MASS3DEA_SIMD_3
            }
          }
        }

        MASS3DEA_CPU_SIMD_4

      }
      }
      }
      } // element batch loop
RP_CALI_SUBKERNEL_END("MASS3DEA_1");
    }
    stopTimer();

    break;
  }

  case RAJA_OpenMP: {

    auto res{getHostResource()};

    //Currently Teams requires two policies if compiled with a device
    using launch_policy = RAJA::LaunchPolicy<RAJA::omp_launch_t>;

    using outer_x = RAJA::LoopPolicy<RAJA::omp_for_exec>;

    using inner_x = RAJA::LoopPolicy<RAJA::seq_exec>;

    using inner_y = RAJA::LoopPolicy<RAJA::seq_exec>;

    using inner_z = RAJA::LoopPolicy<RAJA::seq_exec>;

    constexpr Index_type njb = mea::D1D * mea::D1D * mea::D1D;
    const Index_type nbe = (NE + FEM_VL - 1) / FEM_VL;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MASS3DEA_1");
      //Grid is empty as the host does not need a compute grid to be specified
      //clang-format off
      RAJA::launch<launch_policy>( res,
        RAJA::LaunchParams(),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          // (batch, j-block) pairs keep all threads busy for small NE
          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, nbe * njb),
            [&](Index_type ej) {

              const Index_type e0 = (ej / njb) * FEM_VL;
              const Index_type j1 = ej % mea::D1D;
              const Index_type j2 = (ej / mea::D1D) % mea::D1D;
              const Index_type j3 = (ej / (mea::D1D * mea::D1D)) % mea::D1D;

              MASS3DEA_0_CPU_SIMD

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
                [&](Index_type ) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mea::D1D),
                    [&](Index_type d) {
                      RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mea::Q1D),
                        [&](Index_type q) {
                          MASS3DEA_1
                        }
                      ); // RAJA::loop<inner_y>
                    }
                  ); // RAJA::loop<inner_x>
                }
              ); // RAJA::loop<inner_z>

              MASS3DEA_2_CPU_SIMD

              RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mea::Q1D),
                [&](Index_type k1) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mea::Q1D),
                    [&](Index_type k2) {
                      RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mea::Q1D),
                        [&](Index_type k3) {
                          MASS3DEA_SIMD_3
                        }
                      ); // RAJA::loop<inner_x>
                    }
                  ); // RAJA::loop<inner_y>
                }
              ); // RAJA::loop<inner_z>

              ctx.teamSync();

              MASS3DEA_CPU_SIMD_4

            } // lambda (ej)
          );    // RAJA::loop<outer_x>

        }  // outer lambda (ctx)
      );  // // RAJA::launch
      //clang-format on
      RP_CALI_SUBKERNEL_END("MASS3DEA_1");

    }  // loop over kernel reps
    stopTimer();

    return;
  }

  default:
    getCout() << "\n MASS3DEA : Unknown OpenMP variant id = " << vid
              << std::endl;
  }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

void MASS3DEA::defineOpenMPVariantTunings()
{
  for (VariantID vid : {Base_OpenMP, RAJA_OpenMP}) {
    addVariantTuning<&MASS3DEA::runOpenMPVariant<0>>(vid, getDefaultTuningName());
    // same result, far fewer FLOPs than setFLOPsPerRep() counts
    addVariantTuning<&MASS3DEA::runOpenMPVariant<1>>(vid, "sumfact");
#if defined(RAJAPERF_FEM_AVX512)
    addVariantTuning<&MASS3DEA::runOpenMPVariant<2>>(vid, "sumfact_nt");
    addVariantTuning<&MASS3DEA::runOpenMPVariant<3>>(vid, "sumfact_nt_seq");
#endif
  }
}

} // end namespace apps
} // end namespace rajaperf
