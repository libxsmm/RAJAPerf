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

#include "MASS3DPA_ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>
#include <vector>

namespace rajaperf {
namespace apps {


void MASS3DPA_ATOMIC::runOpenMPVariant(VariantID vid) {

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  MASS3DPA_ATOMIC_DATA_SETUP;
  RAJA_UNUSED_VAR(NE);

  const Index_type Nx = m_Nx;
  const Index_type Ny = m_Ny;
  const Index_type Nz = m_Nz;
  const Index_type nby = (Ny + MPAT_ROW_BLOCK - 1) / MPAT_ROW_BLOCK;
  const Index_type nbz = (Nz + MPAT_ROW_BLOCK - 1) / MPAT_ROW_BLOCK;

  switch (vid) {

  case Base_OpenMP: {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

RP_CALI_SUBKERNEL_BEGIN("MASS3DPA_ATOMIC_1");
#pragma omp parallel
      for (Index_type c = 0; c < 4; ++c) {
        const Index_type nblk = ((nby - (c & 1) + 1) / 2) * ((nbz - (c >> 1) + 1) / 2);
#pragma omp for schedule(dynamic, 1)
      for (Index_type b = 0; b < nblk; ++b) {
      MPAT_COLOR_BLOCK(c, b, nby, nbz);
      for (Index_type ez = bz * MPAT_ROW_BLOCK; ez < ez_end; ++ez) {
      for (Index_type ey = by * MPAT_ROW_BLOCK; ey < ey_end; ++ey) {
      const Index_type rend = Nx * (ey + Ny * ez) + Nx;
      for (Index_type e0 = rend - Nx; e0 < rend; e0 += FEM_VL) {

#if defined(RAJAPERF_FEM_AVX512)
        mass3dpa_atomic_batch_avx512(B, D, X, Y, ElemToDoF, e0, rend);
#else
        MASS3DPA_ATOMIC_0_CPU_SIMD(e0, rend);

        SHARED_LOOP_3D(dx, dy, dz, mpa_at::D1D, mpa_at::D1D, mpa_at::D1D) {
          MASS3DPA_ATOMIC_SIMD_1;
        }

        SHARED_LOOP_2D(q, d, mpa_at::Q1D, mpa_at::D1D) {
          MASS3DPA_ATOMIC_2;
        }

        SHARED_LOOP_3D(qx, dy, dz, mpa_at::Q1D, mpa_at::D1D, mpa_at::D1D) {
          MASS3DPA_ATOMIC_SIMD_3;
        }

        SHARED_LOOP_3D(qx, qy, dz, mpa_at::Q1D, mpa_at::Q1D, mpa_at::D1D) {
          MASS3DPA_ATOMIC_SIMD_4;
        }

        SHARED_LOOP_3D(qx, qy, qz, mpa_at::Q1D, mpa_at::Q1D, mpa_at::Q1D) {
          MASS3DPA_ATOMIC_SIMD_5;
        }

        SHARED_LOOP_3D(dx, qy, qz, mpa_at::D1D, mpa_at::Q1D, mpa_at::Q1D) {
          MASS3DPA_ATOMIC_SIMD_6;
        }

        SHARED_LOOP_3D(dx, dy, qz, mpa_at::D1D, mpa_at::D1D, mpa_at::Q1D) {
          MASS3DPA_ATOMIC_SIMD_7;
        }

        SHARED_LOOP_3D(dx, dy, dz, mpa_at::D1D, mpa_at::D1D, mpa_at::D1D) {
          MASS3DPA_ATOMIC_SIMD_8;
          MASS3DPA_ATOMIC_SIMD_9;
        }
#endif

      } // element batch loop
      } // ey
      } // ez
      } // block loop
      } // color loop
RP_CALI_SUBKERNEL_END("MASS3DPA_ATOMIC_1");
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

      RP_CALI_SUBKERNEL_BEGIN("MASS3DPA_ATOMIC_1");
      //Grid is empty as the host does not need a compute grid to be specified
      //clang-format off
      RAJA::launch<launch_policy>( res,
        RAJA::LaunchParams(),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

         for (Index_type c = 0; c < 4; ++c) {
          const Index_type nblk = ((nby - (c & 1) + 1) / 2) * ((nbz - (c >> 1) + 1) / 2);

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, nblk),
            [&](Index_type b) {

           MPAT_COLOR_BLOCK(c, b, nby, nbz);
           for (Index_type ez = bz * MPAT_ROW_BLOCK; ez < ez_end; ++ez) {
           for (Index_type ey = by * MPAT_ROW_BLOCK; ey < ey_end; ++ey) {
           const Index_type rend = Nx * (ey + Ny * ez) + Nx;
           for (Index_type e0 = rend - Nx; e0 < rend; e0 += FEM_VL) {

#if defined(RAJAPERF_FEM_AVX512)
            mass3dpa_atomic_batch_avx512(B, D, X, Y, ElemToDoF, e0, rend);
#else
            MASS3DPA_ATOMIC_0_CPU_SIMD(e0, rend);

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
              [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                      [&](Index_type dx) {
                      MASS3DPA_ATOMIC_SIMD_1;
                      } // lambda (dx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (dy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
              [&](Index_type ) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type d) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                      [&](Index_type q) {
                      MASS3DPA_ATOMIC_2;
                      } // lambda (q)
                    ); // RAJA::loop<inner_x>
                  } // lambda (d)
                ); // RAJA::loop<inner_y>
              } // lambda ()
            ); // RAJA::loop<inner_z>
            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
              [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                      [&](Index_type qx) {
                      MASS3DPA_ATOMIC_SIMD_3;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (dy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>
            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
              [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                  [&](Index_type qy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                      [&](Index_type qx) {
                      MASS3DPA_ATOMIC_SIMD_4;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (qy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>
            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
              [&](Index_type qz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                  [&](Index_type qy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                      [&](Index_type qx) {
                      MASS3DPA_ATOMIC_SIMD_5;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (qy)
                ); // RAJA::loop<inner_y>
              } // lambda (qz)
            ); // RAJA::loop<inner_z>
            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
              [&](Index_type qz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                  [&](Index_type qy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                      [&](Index_type dx) {
                      MASS3DPA_ATOMIC_SIMD_6;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (qy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>
            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
              [&](Index_type qz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                      [&](Index_type dx) {
                      MASS3DPA_ATOMIC_SIMD_7;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (dy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>
            ctx.teamSync();


            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
              [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                      [&](Index_type dx) {
                      MASS3DPA_ATOMIC_SIMD_8;
                      MASS3DPA_ATOMIC_SIMD_9;
                      } // lambda (dx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (dy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>
#endif

           } // e0
           } // ey
           } // ez
            }  // lambda (b)
          );  // RAJA::loop<outer_x>, implicit barrier between colors
         }  // color loop

        }  // outer lambda (ctx)
      );  // // RAJA::launch
      //clang-format on      
      RP_CALI_SUBKERNEL_END("MASS3DPA_ATOMIC_1");

    }  // loop over kernel reps
    stopTimer();

    return;
  }

  default:
    getCout() << "\n MASS3DPA_ATOMIC : Unknown OpenMP variant id = " << vid
              << std::endl;
  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(MASS3DPA_ATOMIC, OpenMP, Base_OpenMP, RAJA_OpenMP)

} // end namespace apps
} // end namespace rajaperf
