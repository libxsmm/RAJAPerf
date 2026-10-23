//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// Uncomment to add compiler directives for loop unrolling
//#define USE_RAJAPERF_UNROLL

#include "DIFFUSION3DPA.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

// tune_idx 0: default; 1: AVX-512 batches with in-register transposes
template <size_t tune_idx>
void DIFFUSION3DPA::runOpenMPVariant(VariantID vid) {

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  DIFFUSION3DPA_DATA_SETUP;

#if defined(RAJAPERF_FEM_AVX512)
  if constexpr (tune_idx == 1) {

    switch (vid) {

    case Base_OpenMP: {
      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {
        #pragma omp parallel for
        for (Index_type e0 = 0; e0 < NE; e0 += FEM_VL) {
          diffusion3dpa_batch_avx512(Basis, dBasis, D, X, Y, e0,
                                     (NE - e0 < FEM_VL) ? NE - e0 : FEM_VL, NE);
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
        RAJA::forall<RAJA::omp_parallel_for_exec>( res,
          RAJA::RangeStrideSegment(0, NE, FEM_VL), [=](Index_type e0) {
            diffusion3dpa_batch_avx512(Basis, dBasis, D, X, Y, e0,
                                       (NE - e0 < FEM_VL) ? NE - e0 : FEM_VL, NE);
          });
      }
      stopTimer();
      break;
    }

    default:
      getCout() << "\n DIFFUSION3DPA : Unknown OpenMP variant id = " << vid
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

RP_CALI_SUBKERNEL_BEGIN("DIFFUSION3DPA_1");
#pragma omp parallel for
      for (Index_type e0 = 0; e0 < NE; e0 += FEM_VL) {

        DIFFUSION3DPA_0_CPU_SIMD;

        CPU_FOREACH(dz,z,diff::D1D) {
          CPU_FOREACH(dy,y,diff::D1D) {
            CPU_FOREACH(dx,x,diff::D1D) {
              DIFFUSION3DPA_SIMD_1
            }
          }
        }

        CPU_FOREACH(dy,y,diff::D1D) {
          CPU_FOREACH(qx,x,diff::Q1D) {
            DIFFUSION3DPA_2
          }
        }

        CPU_FOREACH(dz,z,diff::D1D) {
          CPU_FOREACH(dy,y,diff::D1D) {
            CPU_FOREACH(qx,x,diff::Q1D) {

              DIFFUSION3DPA_SIMD_3
            }
          }
        }

        CPU_FOREACH(dz,z,diff::D1D) {
          CPU_FOREACH(qy,y,diff::Q1D) {
            CPU_FOREACH(qx,x,diff::Q1D) {
              DIFFUSION3DPA_SIMD_4
            }
          }
        }

        CPU_FOREACH(qz,z,diff::Q1D) {
          CPU_FOREACH(qy,y,diff::Q1D) {
            CPU_FOREACH(qx,x,diff::Q1D) {
              DIFFUSION3DPA_SIMD_5
            }
          }
        }

        CPU_FOREACH(dy,y,diff::D1D) {
          CPU_FOREACH(qx,x,diff::Q1D) {
            DIFFUSION3DPA_6
          }
        }

        CPU_FOREACH(qz,z,diff::Q1D) {
          CPU_FOREACH(qy,y,diff::Q1D) {
            CPU_FOREACH(dx,x,diff::D1D) {
              DIFFUSION3DPA_SIMD_7
            }
          }
        }

        CPU_FOREACH(qz,z,diff::Q1D) {
          CPU_FOREACH(dy,y,diff::D1D) {
            CPU_FOREACH(dx,x,diff::D1D) {
              DIFFUSION3DPA_SIMD_8
            }
          }
        }

        CPU_FOREACH(dz,z,diff::D1D) {
          CPU_FOREACH(dy,y,diff::D1D) {
            CPU_FOREACH(dx,x,diff::D1D) {
              DIFFUSION3DPA_SIMD_9
            }
          }
        }

      } // element loop
RP_CALI_SUBKERNEL_END("DIFFUSION3DPA_1");
    }
    stopTimer();

    break;
  }

  case RAJA_OpenMP: {

    auto res{getHostResource()};

    // Currently Teams requires two policies if compiled with a device
    using launch_policy = RAJA::LaunchPolicy<RAJA::omp_launch_t>;

    using outer_x = RAJA::LoopPolicy<RAJA::omp_for_exec>;

    using inner_x = RAJA::LoopPolicy<RAJA::seq_exec>;

    using inner_y = RAJA::LoopPolicy<RAJA::seq_exec>;

    using inner_z = RAJA::LoopPolicy<RAJA::seq_exec>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("DIFFUSION3DPA_1");
      // Grid is empty as the host does not need a compute grid to be specified
      //clang-format off
      RAJA::launch<launch_policy>( res,
          RAJA::LaunchParams(),
          [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeStrideSegment(0, NE, FEM_VL),
            [&](Index_type e0) {

              DIFFUSION3DPA_0_CPU_SIMD;

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::D1D),
                        [&](Index_type dx) {

                          DIFFUSION3DPA_SIMD_1;

                        } // lambda (dx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
                [&](Index_type RAJA_UNUSED_ARG(dz)) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                        [&](Index_type qx) {

                          DIFFUSION3DPA_2;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                        [&](Index_type qx) {

                          DIFFUSION3DPA_SIMD_3;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                    [&](Index_type qy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                        [&](Index_type qx) {

                          DIFFUSION3DPA_SIMD_4;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::Q1D),
               [&](Index_type qz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                   [&](Index_type qy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                       [&](Index_type qx) {

                        DIFFUSION3DPA_SIMD_5;

                       } // lambda (qx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (qy)
                 );  //RAJA::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
               [&](Index_type RAJA_UNUSED_ARG(dz)) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                   [&](Index_type dy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                       [&](Index_type qx) {

                        DIFFUSION3DPA_6;

                       } // lambda (q)
                     ); // RAJA::loop<inner_x>
                   } // lambda (d)
                 );  //RAJA::loop<inner_y>
               } // lambda (dz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::Q1D),
               [&](Index_type qz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                   [&](Index_type qy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::D1D),
                       [&](Index_type dx) {

                        DIFFUSION3DPA_SIMD_7;

                       } // lambda (dx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (qy)
                 );  //RAJA::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::Q1D),
               [&](Index_type qz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                   [&](Index_type dy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::D1D),
                       [&](Index_type dx) {

                         DIFFUSION3DPA_SIMD_8;

                       } // lambda (dx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (dy)
                 );  //RAJA::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::D1D),
               [&](Index_type dz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                   [&](Index_type dy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::D1D),
                       [&](Index_type dx) {

                         DIFFUSION3DPA_SIMD_9;

                       } // lambda (dx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (dy)
                 );  //RAJA::loop<inner_y>
               } // lambda (dz)
             );  //RAJA::loop<inner_z>

            } // lambda (e)
          ); // RAJA::loop<outer_x>

        }  // outer lambda (ctx)
      );  // RAJA::launch
      //clang-format on
      RP_CALI_SUBKERNEL_END("DIFFUSION3DPA_1");

    }  // loop over kernel reps
    stopTimer();

    return;
  }

  default:
    getCout() << "\n DIFFUSION3DPA : Unknown OpenMP variant id = " << vid
              << std::endl;
  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

void DIFFUSION3DPA::defineOpenMPVariantTunings()
{
  for (VariantID vid : {Base_OpenMP, RAJA_OpenMP}) {
    addVariantTuning<&DIFFUSION3DPA::runOpenMPVariant<0>>(vid, getDefaultTuningName());
#if defined(RAJAPERF_FEM_AVX512)
    addVariantTuning<&DIFFUSION3DPA::runOpenMPVariant<1>>(vid, "transpose");
#endif
  }
}

} // end namespace apps
} // end namespace rajaperf
