//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ENERGY.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{


void ENERGY::runOpenMPVariantReference(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  ENERGY_DATA_SETUP;

  auto energy_lam1 = [=](Index_type i) {
                       ENERGY_BODY1;
                     };
  auto energy_lam2 = [=](Index_type i) {
                       ENERGY_BODY2;
                     };
  auto energy_lam3 = [=](Index_type i) {
                       ENERGY_BODY3;
                     };
  auto energy_lam4 = [=](Index_type i) {
                       ENERGY_BODY4;
                     };
  auto energy_lam5 = [=](Index_type i) {
                       ENERGY_BODY5;
                     };
  auto energy_lam6 = [=](Index_type i) {
                       ENERGY_BODY6;
                     };

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        #pragma omp parallel
        {
          RP_CALI_SUBKERNEL_BEGIN("ENERGY_1");
          #pragma omp for schedule(static) nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            ENERGY_BODY1;
          }
          RP_CALI_SUBKERNEL_END("ENERGY_1");

          RP_CALI_SUBKERNEL_BEGIN("ENERGY_2");
          #pragma omp for schedule(static) nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            ENERGY_BODY2;
          }
          RP_CALI_SUBKERNEL_END("ENERGY_2");

          RP_CALI_SUBKERNEL_BEGIN("ENERGY_3");
          #pragma omp for schedule(static) nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            ENERGY_BODY3;
          }
          RP_CALI_SUBKERNEL_END("ENERGY_3");

          RP_CALI_SUBKERNEL_BEGIN("ENERGY_4");
          #pragma omp for schedule(static) nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            ENERGY_BODY4;
          }
          RP_CALI_SUBKERNEL_END("ENERGY_4");

          RP_CALI_SUBKERNEL_BEGIN("ENERGY_5");
          #pragma omp for schedule(static) nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            ENERGY_BODY5;
          }
          RP_CALI_SUBKERNEL_END("ENERGY_5");

          RP_CALI_SUBKERNEL_BEGIN("ENERGY_6");
          #pragma omp for schedule(static) nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            ENERGY_BODY6;
          }
          RP_CALI_SUBKERNEL_END("ENERGY_6");

        } // end omp parallel region

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        #pragma omp parallel
        {
          RP_CALI_SUBKERNEL_BEGIN("ENERGY_1");
          #pragma omp for schedule(static) nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            energy_lam1(i);
          }
          RP_CALI_SUBKERNEL_END("ENERGY_1");

          RP_CALI_SUBKERNEL_BEGIN("ENERGY_2");
          #pragma omp for schedule(static) nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            energy_lam2(i);
          }
          RP_CALI_SUBKERNEL_END("ENERGY_2");

          RP_CALI_SUBKERNEL_BEGIN("ENERGY_3");
          #pragma omp for schedule(static) nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            energy_lam3(i);
          }
          RP_CALI_SUBKERNEL_END("ENERGY_3");

          RP_CALI_SUBKERNEL_BEGIN("ENERGY_4");
          #pragma omp for schedule(static) nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            energy_lam4(i);
          }
          RP_CALI_SUBKERNEL_END("ENERGY_4");

          RP_CALI_SUBKERNEL_BEGIN("ENERGY_5");
          #pragma omp for schedule(static) nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            energy_lam5(i);
          }
          RP_CALI_SUBKERNEL_END("ENERGY_5");

          RP_CALI_SUBKERNEL_BEGIN("ENERGY_6");
          #pragma omp for schedule(static) nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            energy_lam6(i);
          }
          RP_CALI_SUBKERNEL_END("ENERGY_6");

        } // end omp parallel region

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RAJA::region<RAJA::omp_parallel_region>( [=]() {

          RP_CALI_SUBKERNEL_BEGIN("ENERGY_1");
          RAJA::forall< RAJA::omp_for_nowait_static_exec< > >( res,
            RAJA::RangeSegment(ibegin, iend), energy_lam1);
          RP_CALI_SUBKERNEL_END("ENERGY_1");

          RP_CALI_SUBKERNEL_BEGIN("ENERGY_2");
          RAJA::forall< RAJA::omp_for_nowait_static_exec< > >( res,
            RAJA::RangeSegment(ibegin, iend), energy_lam2);
          RP_CALI_SUBKERNEL_END("ENERGY_2");

          RP_CALI_SUBKERNEL_BEGIN("ENERGY_3");
          RAJA::forall< RAJA::omp_for_nowait_static_exec< > >( res,
            RAJA::RangeSegment(ibegin, iend), energy_lam3);
          RP_CALI_SUBKERNEL_END("ENERGY_3");

          RP_CALI_SUBKERNEL_BEGIN("ENERGY_4");
          RAJA::forall< RAJA::omp_for_nowait_static_exec< > >( res,
            RAJA::RangeSegment(ibegin, iend), energy_lam4);
          RP_CALI_SUBKERNEL_END("ENERGY_4");

          RP_CALI_SUBKERNEL_BEGIN("ENERGY_5");
          RAJA::forall< RAJA::omp_for_nowait_static_exec< > >( res,
            RAJA::RangeSegment(ibegin, iend), energy_lam5);
          RP_CALI_SUBKERNEL_END("ENERGY_5");

          RP_CALI_SUBKERNEL_BEGIN("ENERGY_6");
          RAJA::forall< RAJA::omp_for_nowait_static_exec< > >( res,
            RAJA::RangeSegment(ibegin, iend), energy_lam6);
          RP_CALI_SUBKERNEL_END("ENERGY_6");

        }); // end omp parallel region

      }
      stopTimer();
      break;
    }

    default : {
      getCout() << "\n  ENERGY : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

// tune_idx 0: reference (six loops); 1: fused single pass; 2: fused with
// streaming (non-temporal) stores of e_new and q_new
template < size_t tune_idx >
void ENERGY::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  if constexpr (tune_idx == 0) {
    runOpenMPVariantReference(vid);
  } else {

  constexpr bool stream = (tune_idx == 2);
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  ENERGY_DATA_SETUP;

  const bool nt_ok = fem_simd::is_aligned64(e_new) && fem_simd::is_aligned64(q_new);

  auto energy_tile_lam = [=](Index_type t) {
                           ENERGY_FUSED_TILE(t, stream);
                         };

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        #pragma omp parallel
        {
          RP_CALI_SUBKERNEL_BEGIN("ENERGY_FUSED");
          #pragma omp for schedule(static) nowait
          for (Index_type t = ibegin; t < iend; t += ENERGY_TILE ) {
            ENERGY_FUSED_TILE(t, stream);
          }
          if (stream) { fem_simd::stream_fence(); }
          RP_CALI_SUBKERNEL_END("ENERGY_FUSED");
        }

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        #pragma omp parallel
        {
          RP_CALI_SUBKERNEL_BEGIN("ENERGY_FUSED");
          #pragma omp for schedule(static) nowait
          for (Index_type t = ibegin; t < iend; t += ENERGY_TILE ) {
            energy_tile_lam(t);
          }
          if (stream) { fem_simd::stream_fence(); }
          RP_CALI_SUBKERNEL_END("ENERGY_FUSED");
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RAJA::region<RAJA::omp_parallel_region>( [=]() {
          RP_CALI_SUBKERNEL_BEGIN("ENERGY_FUSED");
          RAJA::forall< RAJA::omp_for_nowait_static_exec< > >( res,
            RAJA::RangeStrideSegment(ibegin, iend, ENERGY_TILE), energy_tile_lam);
          if (stream) { fem_simd::stream_fence(); }
          RP_CALI_SUBKERNEL_END("ENERGY_FUSED");
        });

      }
      stopTimer();
      break;
    }

    default : {
      getCout() << "\n  ENERGY : Unknown variant id = " << vid << std::endl;
    }

  }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

void ENERGY::defineOpenMPVariantTunings()
{
  for (VariantID vid : {Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP}) {
    addVariantTuning<&ENERGY::runOpenMPVariant<0>>(vid, getDefaultTuningName());
    // same result; fewer bytes moved than setBytes*PerRep() count
    addVariantTuning<&ENERGY::runOpenMPVariant<1>>(vid, "fused");
#if defined(RAJAPERF_FEM_AVX512)
    addVariantTuning<&ENERGY::runOpenMPVariant<2>>(vid, "fused_nt");
#endif
  }
}
} // end namespace apps
} // end namespace rajaperf
