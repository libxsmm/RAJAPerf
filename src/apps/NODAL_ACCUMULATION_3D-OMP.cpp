//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NODAL_ACCUMULATION_3D.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{


void NODAL_ACCUMULATION_3D::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();

  Real_ptr x = m_x;
  Real_ptr vol = m_vol;
  NODAL_ACCUMULATION_3D_GATHER_SETUP;


  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("NODAL_ACCUMULATION_3D_1");
        #pragma omp parallel for
        for (Index_type r = 0 ; r < nrows ; ++r ) {
          NODAL_ACCUMULATION_3D_GATHER_ROW(r);
        }
        RP_CALI_SUBKERNEL_END("NODAL_ACCUMULATION_3D_1");

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto nodal_accumulation_3d_lam = [=](Index_type r) {
            NODAL_ACCUMULATION_3D_GATHER_ROW(r);
          };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("NODAL_ACCUMULATION_3D_1");
        #pragma omp parallel for
        for (Index_type r = 0 ; r < nrows ; ++r ) {
          nodal_accumulation_3d_lam(r);
        }
        RP_CALI_SUBKERNEL_END("NODAL_ACCUMULATION_3D_1");

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      auto nodal_accumulation_3d_lam = [=](Index_type r) {
                                         NODAL_ACCUMULATION_3D_GATHER_ROW(r);
                                       };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("NODAL_ACCUMULATION_3D_1");
        RAJA::forall<RAJA::omp_parallel_for_exec>( res,
          RAJA::RangeSegment(0, nrows), nodal_accumulation_3d_lam);
        RP_CALI_SUBKERNEL_END("NODAL_ACCUMULATION_3D_1");

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  NODAL_ACCUMULATION_3D : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(NODAL_ACCUMULATION_3D, OpenMP, Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP)

} // end namespace apps
} // end namespace rajaperf
