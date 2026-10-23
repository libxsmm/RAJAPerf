//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INTSC_HEXRECT.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

// tune_idx 0: one intersection per iteration; 1: one donor zone (its eight
// intersections) per iteration with plane splitting; 2: as 1 without polygon
// staging copies (split_lean)
template < size_t tune_idx >
void INTSC_HEXRECT::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  //  We compute each standard intersection within a single thread
  //  to avoid collisions in vv_out hence only distribute different
  //  standard intersections among threads, iend is getActualProblemSize.
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0 ;
  const Index_type iend = ( tune_idx == 0 ) ? m_nrecords
                                            : m_nrecords / intsc_per_zone ;

  INTSC_HEXRECT_DATA_SETUP ;

  auto intsc_hexrect_lam = [=] ( Index_type i ) {
      if constexpr ( tune_idx == 0 ) {
        INTSC_HEXRECT_OMP ( i ) ;
      } else {
        INTSC_HEXRECT_SPLIT_BODY_T ( i, tune_idx == 2 ) ;
      }
  } ;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXRECT_1");
        #pragma omp parallel for
        for (Index_type i = ibegin ; i < iend ; ++i ) {
          if constexpr ( tune_idx == 0 ) {
            INTSC_HEXRECT_OMP( i ) ;
          } else {
            INTSC_HEXRECT_SPLIT_BODY_T( i, tune_idx == 2 ) ;
          }
        }
        RP_CALI_SUBKERNEL_END("INTSC_HEXRECT_1");

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXRECT_1");
        #pragma omp parallel for
        for (Index_type i = ibegin ; i < iend ; ++i ) {
          intsc_hexrect_lam(i);
        }
        RP_CALI_SUBKERNEL_END("INTSC_HEXRECT_1");

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXRECT_1");
        RAJA::forall<RAJA::omp_parallel_for_exec>( res,
          RAJA::RangeSegment(ibegin, iend), intsc_hexrect_lam);
        RP_CALI_SUBKERNEL_END("INTSC_HEXRECT_1");

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  INTSC_HEXRECT : Unknown OpenMP variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

void INTSC_HEXRECT::defineOpenMPVariantTunings()
{
  for (VariantID vid : {Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP}) {
    addVariantTuning<&INTSC_HEXRECT::runOpenMPVariant<0>>(vid, getDefaultTuningName());
    // same result; less work than setFLOPsPerRep() counts
    addVariantTuning<&INTSC_HEXRECT::runOpenMPVariant<1>>(vid, "split");
    addVariantTuning<&INTSC_HEXRECT::runOpenMPVariant<2>>(vid, "split_lean");
  }
}

} // end namespace apps
} // end namespace rajaperf
