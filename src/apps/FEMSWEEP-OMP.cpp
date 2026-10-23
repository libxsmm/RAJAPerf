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


//
// tune_idx 0: reference SoA sweep; 1: femsweep_element with prefetching
// FEMSWEEP_PF elements ahead; 2, 3: as 1, prefetching 1 or 4 elements ahead;
// 4: as 1, but the last, partial round of tasks is run by larger teams so
// all threads stay busy.
//
template < size_t tune_idx >
void FEMSWEEP::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();

  FEMSWEEP_DATA_SETUP;

  const FemsweepData fd{Bdat, Adat, Fdat, Sgdat, M0dat, Xdat,
                        AngleElem2FaceType, elem_to_faces, F_g2l, idx1, idx2,
                        ne, ng, sharedinteriorfaces};
  RAJA_UNUSED_VAR(fd);

  const Index_type nbg = (ng + FEM_VL - 1) / FEM_VL;

  const int nthreads = omp_get_max_threads();
  int team_size = 1;
  int nteams = nthreads;
  constexpr Index_type pfd = tune_idx == 2 ? 1 : (tune_idx == 3 ? 4 : FEMSWEEP_PF);
  RAJA_UNUSED_VAR(pfd);
  if constexpr (tune_idx == 0) {
    femsweep_team_shape(na * nbg, nthreads, team_size, nteams);
  } else {
    // bandwidth bound: teams (and their barriers) only to keep all threads busy
    while (team_size < 8 && nthreads / team_size > na * nbg) { team_size *= 2; }
    nteams = nthreads / team_size;
  }
  // tune_idx 4: tasks [0, t_split) in full rounds of single-thread sweeps,
  // the remaining ones by nteams2 teams of team_size2 threads
  const Index_type t_split = (tune_idx == 4) ? (na * nbg / nthreads) * nthreads : na * nbg;
  int team_size2 = 1;
  while (team_size2 < 8 && (na * nbg - t_split) * team_size2 * 2 <= nthreads) {
    team_size2 *= 2;
  }
  const int nteams2 = nthreads / team_size2;
  std::vector<FemsweepTeamBarrier> barrier_storage(nthreads);
  FemsweepTeamBarrier* team_barriers = barrier_storage.data();
  std::vector<FemsweepTeamBarrier> barrier_storage2(nteams2);
  FemsweepTeamBarrier* team_barriers2 = barrier_storage2.data();
  RAJA_UNUSED_VAR(team_barriers2);

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("FEMSWEEP_1");
        #pragma omp parallel num_threads(nthreads)
        {
          const int slot = omp_get_thread_num();
          if constexpr (tune_idx == 0) {
            FEMSWEEP_TEAM_SWEEP(slot);
          } else if constexpr (tune_idx == 4) {
            FEMSWEEP_TEAM_SWEEP_RANGE(slot, 0, t_split, 1, nthreads, team_barriers, pfd);
            FEMSWEEP_TEAM_SWEEP_RANGE(slot, t_split, na * nbg, team_size2, nteams2,
                                      team_barriers2, pfd);
          } else {
            FEMSWEEP_TEAM_SWEEP_RANGE(slot, 0, na * nbg, team_size, nteams,
                                      team_barriers, pfd);
          }
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
            const int islot = static_cast<int>(slot);
            if constexpr (tune_idx == 0) {
              FEMSWEEP_TEAM_SWEEP(islot);
            } else if constexpr (tune_idx == 4) {
              FEMSWEEP_TEAM_SWEEP_RANGE(islot, 0, t_split, 1, nthreads, team_barriers, pfd);
              FEMSWEEP_TEAM_SWEEP_RANGE(islot, t_split, na * nbg, team_size2, nteams2,
                                        team_barriers2, pfd);
            } else {
              FEMSWEEP_TEAM_SWEEP_RANGE(islot, 0, na * nbg, team_size, nteams,
                                        team_barriers, pfd);
            }
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

void FEMSWEEP::defineOpenMPVariantTunings()
{
  for (VariantID vid : {Base_OpenMP, RAJA_OpenMP}) {
    addVariantTuning<&FEMSWEEP::runOpenMPVariant<0>>(vid, getDefaultTuningName());
    addVariantTuning<&FEMSWEEP::runOpenMPVariant<1>>(vid, "prefetch");
    addVariantTuning<&FEMSWEEP::runOpenMPVariant<2>>(vid, "prefetch_pf1");
    addVariantTuning<&FEMSWEEP::runOpenMPVariant<3>>(vid, "prefetch_pf4");
    addVariantTuning<&FEMSWEEP::runOpenMPVariant<4>>(vid, "prefetch_mixed");
  }
}

} // end namespace apps
} // end namespace rajaperf
