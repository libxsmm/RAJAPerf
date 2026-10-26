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

#if defined(RAJA_ENABLE_OPENMP)
#include <omp.h>
#endif

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

//
// 'blocked' tuning: the gather of the default tuning, cache blocked. Work
// units are (block of <= JB node rows in j) x (slab of k-planes) and walk the
// planes of a slab in order, so the vol rows of plane k-1 (and the rows j-1)
// are still in L2 instead of being read again from memory.
//
void NODAL_ACCUMULATION_3D::runOpenMPVariantBlocked(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();

  Real_ptr x = m_x;
  Real_ptr vol = m_vol;
  NODAL_ACCUMULATION_3D_GATHER_SETUP;

  RAJA_UNUSED_VAR(nrows);
  // about one unit per thread: each block re-reads one halo row (j-1) and
  // each slab one halo plane (k-1), so keep both blocks large
  constexpr Index_type JB = 64;
  const Index_type nk = kmax - kmin + 1;
  const Index_type ntj = (nj + JB - 1) / JB;
  const Index_type nks = RAJA_MIN(nk, RAJA_MAX(Index_type(1),
      (static_cast<Index_type>(omp_get_max_threads()) + ntj - 1) / ntj));
  const Index_type nunits = ntj * nks;

  auto nodal_block = [=](Index_type u) {
    const Index_type j0 = (nj * (u % ntj)) / ntj;
    const Index_type j1 = (nj * (u % ntj + 1)) / ntj;
    const Index_type ks = u / ntj;
    const Index_type ka = (nk * ks) / nks;
    const Index_type kb = (nk * (ks + 1)) / nks;
    for (Index_type kk = ka; kk < kb; ++kk) {
      for (Index_type jj = j0; jj < j1; ++jj) {
        NODAL_ACCUMULATION_3D_GATHER_ROW(kk * nj + jj);
      }
    }
  };

  switch ( vid ) {

    case Base_OpenMP :
    case Lambda_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("NODAL_ACCUMULATION_3D_1");
        #pragma omp parallel for
        for (Index_type u = 0 ; u < nunits ; ++u ) {
          nodal_block(u);
        }
        RP_CALI_SUBKERNEL_END("NODAL_ACCUMULATION_3D_1");

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("NODAL_ACCUMULATION_3D_1");
        RAJA::forall<RAJA::omp_parallel_for_exec>( res,
          RAJA::RangeSegment(0, nunits), nodal_block);
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

void NODAL_ACCUMULATION_3D::defineOpenMPVariantTunings()
{
  for (VariantID vid : {Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP}) {
    addVariantTuning<&NODAL_ACCUMULATION_3D::runOpenMPVariant>(vid, getDefaultTuningName());
    addVariantTuning<&NODAL_ACCUMULATION_3D::runOpenMPVariantBlocked>(vid, "blocked");
  }
}

} // end namespace apps
} // end namespace rajaperf
