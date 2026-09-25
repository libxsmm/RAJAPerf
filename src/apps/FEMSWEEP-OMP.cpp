//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FEMSWEEP.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>
#include <vector>

#if defined(RAJA_ENABLE_OPENMP)
#include <omp.h>
#endif


namespace rajaperf
{
namespace apps
{


void FEMSWEEP::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();

  FEMSWEEP_DATA_SETUP;

  const Index_type nbg = (ng + FEM_VL - 1) / FEM_VL;

  const int nthreads = omp_get_max_threads();
  int team_size = 1;
  int nteams = nthreads;
  femsweep_team_shape(na * nbg, nthreads, team_size, nteams);
  std::vector<FemsweepTeamBarrier> barrier_storage(nteams);
  FemsweepTeamBarrier* team_barriers = barrier_storage.data();

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("FEMSWEEP_1");
        #pragma omp parallel num_threads(nthreads)
        {
          FEMSWEEP_TEAM_SWEEP(omp_get_thread_num());
        }
        RP_CALI_SUBKERNEL_END("FEMSWEEP_1");

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      using launch_policy =
          RAJA::LaunchPolicy<RAJA::omp_launch_t>;

      // TODO: add omp_parallel_collapse_exec version

      using outer_xy =
          RAJA::LoopPolicy<RAJA::omp_for_exec>;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("FEMSWEEP_1");
        RAJA::launch<launch_policy>( res,
            RAJA::LaunchParams(),
            [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          // one slot per thread (static schedule); team members run concurrently
          RAJA::loop<outer_xy>(ctx, RAJA::RangeSegment(0, nthreads),
              [&](Index_type slot) {
            FEMSWEEP_TEAM_SWEEP(static_cast<int>(slot));
          });  // slot loop
        });
        RP_CALI_SUBKERNEL_END("FEMSWEEP_1");

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n FEMSWEEP : Unknown OpenMP variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(FEMSWEEP, OpenMP, Base_OpenMP, RAJA_OpenMP)

} // end namespace apps
} // end namespace rajaperf
