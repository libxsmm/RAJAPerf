//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "EDGE3D.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

// tune_idx 0: reference algorithm; 1: metric-tensor reformulation;
// 2/3 (AVX-512 only): metric with 2/4 zmm per value (16/32 elements per batch);
// 4: register-blocked metric (metric_rb), 5/6 (AVX-512 only): with 2/4 zmm
#if defined(RAJAPERF_FEM_AVX512)
#define EDGE3D_OMP_BODY \
  if constexpr (tune_idx == 0) { EDGE3D_SIMD_BODY; } \
  else if constexpr (tune_idx == 1) { EDGE3D_METRIC_BODY; } \
  else if constexpr (tune_idx <= 3) { EDGE3D_METRIC_BODY_NV(nv); } \
  else { EDGE3D_METRIC_RB_BODY(nv); }
#else
#define EDGE3D_OMP_BODY \
  if constexpr (tune_idx == 0) { EDGE3D_SIMD_BODY; } \
  else if constexpr (tune_idx == 1) { EDGE3D_METRIC_BODY; } \
  else { EDGE3D_METRIC_RB_BODY(nv); }
#endif


template < size_t tune_idx >
void EDGE3D::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = m_domain->fpz;
  const Index_type iend = m_domain->lpz+1;

  EDGE3D_DATA_SETUP;

  constexpr int nv = (tune_idx == 3 || tune_idx == 6) ? 4
                   : (tune_idx == 2 || tune_idx == 5) ? 2 : 1;
  constexpr Index_type bl = EDGE3D_VL * nv;
  const Edge3DQpTable<NQ_1D> edge3d_qpt(1);

  auto edge3d_lam =
    [=](Index_type i0) {
      EDGE3D_OMP_BODY
    };

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("EDGE3D_1");
        #pragma omp parallel for
        for (Index_type i0 = ibegin ; i0 < iend ; i0 += bl ) {
          EDGE3D_OMP_BODY
        }
        RP_CALI_SUBKERNEL_END("EDGE3D_1");

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("EDGE3D_1");
        #pragma omp parallel for
        for (Index_type i0 = ibegin ; i0 < iend ; i0 += bl ) {
          edge3d_lam(i0);
        }
        RP_CALI_SUBKERNEL_END("EDGE3D_1");

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("EDGE3D_1");
        RAJA::forall<RAJA::omp_parallel_for_exec>( res,
          RAJA::RangeStrideSegment(ibegin, iend, bl), edge3d_lam);
        RP_CALI_SUBKERNEL_END("EDGE3D_1");

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  EDGE3D : Unknown OpenMP variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

void EDGE3D::defineOpenMPVariantTunings()
{
  for (VariantID vid : {Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP}) {
    addVariantTuning<&EDGE3D::runOpenMPVariant<0>>(vid, getDefaultTuningName());
    // same result, fewer FLOPs than setFLOPsPerRep() counts (see EDGE3D_METRIC_BODY)
    addVariantTuning<&EDGE3D::runOpenMPVariant<1>>(vid, "metric");
    // metric with the matrix accumulated in registers (no Mu load/store per qp)
    addVariantTuning<&EDGE3D::runOpenMPVariant<4>>(vid, "metric_rb");
#if defined(RAJAPERF_FEM_AVX512)
    // metric with 16/32 elements per batch (2/4 zmm per value)
    addVariantTuning<&EDGE3D::runOpenMPVariant<2>>(vid, "metric_x2");
    addVariantTuning<&EDGE3D::runOpenMPVariant<3>>(vid, "metric_x4");
    addVariantTuning<&EDGE3D::runOpenMPVariant<5>>(vid, "metric_rb_x2");
    addVariantTuning<&EDGE3D::runOpenMPVariant<6>>(vid, "metric_rb_x4");
#endif
  }
}

} // end namespace apps
} // end namespace rajaperf
