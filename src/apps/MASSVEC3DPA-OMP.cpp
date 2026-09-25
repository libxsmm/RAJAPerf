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

#include "MASSVEC3DPA.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {


// tune_idx 0: default; 1: AVX-512 batches with streaming stores of Y
template <size_t tune_idx>
void MASSVEC3DPA::runOpenMPVariant(VariantID vid)
{

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  MASSVEC3DPA_DATA_SETUP;

#if defined(RAJAPERF_FEM_AVX512)
  if constexpr (tune_idx == 1) {

    switch (vid) {

    case Base_OpenMP: {
      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {
        #pragma omp parallel
        {
          #pragma omp for nowait
          for (Index_type e0 = 0; e0 < NE; e0 += FEM_VL) {
            massvec3dpa_batch_avx512<true>(B, D, X, Y, e0,
                                           (NE - e0 < FEM_VL) ? NE - e0 : FEM_VL);
          }
          fem_simd::stream_fence();
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
            RAJA::RangeStrideSegment(0, NE, FEM_VL), [=](Index_type e0) {
              massvec3dpa_batch_avx512<true>(B, D, X, Y, e0,
                                             (NE - e0 < FEM_VL) ? NE - e0 : FEM_VL);
            });
          fem_simd::stream_fence();
        });
      }
      stopTimer();
      break;
    }

    default:
      getCout() << "\n MASSVEC3DPA : Unknown OpenMP variant id = " << vid
                << std::endl;
    }
    return;
  }
#endif

  switch (vid) {

  case Base_OpenMP: {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

RP_CALI_SUBKERNEL_BEGIN("MASSVEC3DPA_1");
#pragma omp parallel for
      for (Index_type e0 = 0; e0 < NE; e0 += FEM_VL) {

#if defined(RAJAPERF_FEM_AVX512)
        MASSVEC3DPA_CPU_SIMD_BATCH
#else
        MASSVEC3DPA_0_CPU_SIMD;

        SHARED_LOOP_2D(q, d, mvpa::Q1D, mvpa::D1D) {
          MASSVEC3DPA_1;
        }

        for (Index_type c = 0; c < 3; ++c) {
          SHARED_LOOP_3D(dx, dy, dz, mvpa::D1D, mvpa::D1D, mvpa::D1D) {

            MASSVEC3DPA_SIMD_2;
          }

          SHARED_LOOP_3D(qx, dy, dz, mvpa::Q1D, mvpa::D1D, mvpa::D1D) {

            MASSVEC3DPA_SIMD_3;
          }

          SHARED_LOOP_3D(qx, qy, dz, mvpa::Q1D, mvpa::Q1D, mvpa::D1D) {
            MASSVEC3DPA_SIMD_4;
          }

          SHARED_LOOP_3D(qx, qy, qz, mvpa::Q1D, mvpa::Q1D, mvpa::Q1D) {
            MASSVEC3DPA_SIMD_5;
          }

          SHARED_LOOP_3D(dx, qy, qz, mvpa::D1D, mvpa::Q1D, mvpa::Q1D) {
            MASSVEC3DPA_SIMD_6;
          }

          SHARED_LOOP_3D(dx, dy, qz, mvpa::D1D, mvpa::D1D, mvpa::Q1D) {
            MASSVEC3DPA_SIMD_7;
          }

          SHARED_LOOP_3D(dx, dy, dz, mvpa::D1D, mvpa::D1D, mvpa::D1D) {
            MASSVEC3DPA_SIMD_8;
          }

        } //(c) dimension loop
#endif

      } // element loop
RP_CALI_SUBKERNEL_END("MASSVEC3DPA_1");
    }
    stopTimer();

    break;
  }

  case RAJA_OpenMP: {

    auto res{getHostResource()};

    using launch_policy = RAJA::LaunchPolicy<RAJA::omp_launch_t>;

    using outer_x = RAJA::LoopPolicy<RAJA::omp_for_exec>;

#if !defined(RAJAPERF_FEM_AVX512)
    using inner_x = RAJA::LoopPolicy<RAJA::seq_exec>;

    using inner_y = RAJA::LoopPolicy<RAJA::seq_exec>;

    using inner_z = RAJA::LoopPolicy<RAJA::seq_exec>;
#endif

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MASSVEC3DPA_1");
      //Grid is empty as the host does not need a compute grid to be specified
      //clang-format off
      RAJA::launch<launch_policy>( res,
        RAJA::LaunchParams(),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeStrideSegment(0, NE, FEM_VL),
            [&](Index_type e0) {

#if defined(RAJAPERF_FEM_AVX512)
            MASSVEC3DPA_CPU_SIMD_BATCH
#else
            MASSVEC3DPA_0_CPU_SIMD;

            //3 loops to remain consistent with the GPU versions
            //Masking out of the z-dimension thread is done with GPU versions
            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
              [&](Index_type) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::D1D),
                  [&](Index_type d) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::Q1D),
                      [&](Index_type q) {
                        MASSVEC3DPA_1;
                      } // lambda (q)
                    ); // RAJA::loop<inner_x>
                  } // lambda (d)
                ); // RAJA::loop<inner_y>
              } // lambda ()
            ); // RAJA::loop<inner_z>

            for (Index_type c = 0; c < 3; ++c) {

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mvpa::D1D),
              [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::D1D),
                      [&](Index_type dx) {
                        MASSVEC3DPA_SIMD_2;
                      } // lambda (dx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (dy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>

            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mvpa::D1D),
              [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::Q1D),
                      [&](Index_type qx) {
                        MASSVEC3DPA_SIMD_3;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (dy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>

            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mvpa::D1D),
              [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::Q1D),
                  [&](Index_type qy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::Q1D),
                      [&](Index_type qx) {
                        MASSVEC3DPA_SIMD_4;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (qy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>

            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mvpa::Q1D),
              [&](Index_type qz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::Q1D),
                  [&](Index_type qy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::Q1D),
                      [&](Index_type qx) {
                        MASSVEC3DPA_SIMD_5;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (qy)
                ); // RAJA::loop<inner_y>
              } // lambda (qz)
            ); // RAJA::loop<inner_z>

            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mvpa::Q1D),
              [&](Index_type qz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::Q1D),
                  [&](Index_type qy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::D1D),
                      [&](Index_type dx) {
                        MASSVEC3DPA_SIMD_6;
                      } // lambda (dx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (qy)
                ); // RAJA::loop<inner_y>
              } // lambda (qz)
            ); // RAJA::loop<inner_z>

            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mvpa::Q1D),
              [&](Index_type qz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::D1D),
                      [&](Index_type dx) {
                        MASSVEC3DPA_SIMD_7;
                      } // lambda (dx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (dy)
                ); // RAJA::loop<inner_y>
              } // lambda (qz)
            ); // RAJA::loop<inner_z>

            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mvpa::D1D),
              [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::D1D),
                      [&](Index_type dx) {
                      MASSVEC3DPA_SIMD_8;
                      } // lambda (dx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (dy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>

            ctx.teamSync();

            } // c - dim loop
#endif
          }  // lambda (e)
         );  // RAJA::loop<outer_x>
        }  // outer lambda (ctx)
      );  // // RAJA::launch
      //clang-format on
      RP_CALI_SUBKERNEL_END("MASSVEC3DPA_1");

    }  // loop over kernel reps
    stopTimer();

    return;
  }

  default:
    getCout() << "\n MASSVEC3DPA : Unknown OpenMP variant id = " << vid
              << std::endl;
  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

void MASSVEC3DPA::defineOpenMPVariantTunings()
{
  for (VariantID vid : {Base_OpenMP, RAJA_OpenMP}) {
    addVariantTuning<&MASSVEC3DPA::runOpenMPVariant<0>>(vid, getDefaultTuningName());
#if defined(RAJAPERF_FEM_AVX512)
    addVariantTuning<&MASSVEC3DPA::runOpenMPVariant<1>>(vid, "nt");
#endif
  }
}

} // end namespace apps
} // end namespace rajaperf
