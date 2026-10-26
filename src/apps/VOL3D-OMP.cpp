//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "VOL3D.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"
#include "FEM_SIMD.hpp"

#include <cstdint>
#include <iostream>

#if defined(RAJA_ENABLE_OPENMP)
#include <omp.h>
#endif

namespace rajaperf
{
namespace apps
{


void VOL3D::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = m_domain->fpz;
  const Index_type iend = m_domain->lpz+1;

  VOL3D_DATA_SETUP;

  auto vol3d_lam = [=](Index_type i) {
                     VOL3D_BODY;
                   };

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("VOL3D_1");
        #pragma omp parallel for
        for (Index_type i = ibegin ; i < iend ; ++i ) {
          VOL3D_BODY;
        }
        RP_CALI_SUBKERNEL_END("VOL3D_1");

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("VOL3D_1");
        #pragma omp parallel for
        for (Index_type i = ibegin ; i < iend ; ++i ) {
          vol3d_lam(i);
        }
        RP_CALI_SUBKERNEL_END("VOL3D_1");

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("VOL3D_1");
        RAJA::forall<RAJA::omp_parallel_for_exec>( res,
          RAJA::RangeSegment(ibegin, iend), vol3d_lam);
        RP_CALI_SUBKERNEL_END("VOL3D_1");

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  VOL3D : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

// vol[i] of one tile, staged in an L1 buffer: VOL3D_BODY writes through this.
struct Vol3dTile
{
  Real_type* buf;
  Index_type off;
  RAJA_INLINE Real_type& operator[](Index_type i) const { return buf[i - off]; }
};

//
// Non-default tunings (the default is runOpenMPVariant):
// tune_idx 1: vol written with non-temporal stores. Each thread
// gets one contiguous chunk aligned to 64B lines of vol, computed in L1 tiles
// that are stream-copied to vol. Pays off only when vol is not reused from
// the LLC.
// tune_idx 2: cache blocked. The index range is split into rows of length
// jp; work units are (tile of JT rows) x (slab of k-planes) and walk the
// planes of a slab innermost, so the k+1 plane rows of x, y, z are reused
// from L2 instead of being read again from memory. Every index in
// [ibegin, iend) is computed exactly once with VOL3D_BODY.
// tune_idx 3: plane strips. Thread s owns the offsets [s*chunk, (s+1)*chunk)
// of every k-plane and walks the planes in order, so x, y, z of plane k+1 are
// still in L2 when it computes plane k+1 (one long contiguous run per plane).
// tune_idx 4: tune_idx 3 with vol written by streaming stores as in 1.
//
template < size_t tune_idx >
void VOL3D::runOpenMPVariantTuned(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = m_domain->fpz;
  const Index_type iend = m_domain->lpz+1;

  VOL3D_DATA_SETUP;

  if constexpr (tune_idx == 2) {

    constexpr Index_type JT = 16;
    const Index_type jp = m_domain->jp;
    const Index_type nrow = m_domain->kp / jp;
    const Index_type k_lo = (ibegin / jp) / nrow;
    const Index_type nk = ((iend - 1) / jp) / nrow + 1 - k_lo;
    const Index_type ntj = (nrow + JT - 1) / JT;
    const Index_type nks = RAJA_MIN(nk, RAJA_MAX(Index_type(1),
        (4 * static_cast<Index_type>(omp_get_max_threads()) + ntj - 1) / ntj));
    const Index_type nunits = ntj * nks;

    auto vol3d_block = [=](Index_type u) {
      const Index_type j0 = (u % ntj) * JT;
      const Index_type j1 = RAJA_MIN(j0 + JT, nrow);
      const Index_type ks = u / ntj;
      const Index_type ka = k_lo + (nk * ks) / nks;
      const Index_type kb = k_lo + (nk * (ks + 1)) / nks;
      for (Index_type k = ka; k < kb; ++k) {
        for (Index_type j = j0; j < j1; ++j) {
          const Index_type r0 = (k * nrow + j) * jp;
          const Index_type lo = RAJA_MAX(r0, ibegin);
          const Index_type hi = RAJA_MIN(r0 + jp, iend);
          for (Index_type i = lo; i < hi; ++i) {
            VOL3D_BODY;
          }
        }
      }
    };

    switch ( vid ) {

      case Base_OpenMP :
      case Lambda_OpenMP : {

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("VOL3D_1");
          #pragma omp parallel for
          for (Index_type u = 0; u < nunits; ++u) {
            vol3d_block(u);
          }
          RP_CALI_SUBKERNEL_END("VOL3D_1");

        }
        stopTimer();

        break;
      }

      case RAJA_OpenMP : {

        auto res{getHostResource()};

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("VOL3D_1");
          RAJA::forall<RAJA::omp_parallel_for_exec>( res,
            RAJA::RangeSegment(0, nunits), vol3d_block);
          RP_CALI_SUBKERNEL_END("VOL3D_1");

        }
        stopTimer();

        break;
      }

      default : {
        getCout() << "\n  VOL3D : Unknown variant id = " << vid << std::endl;
      }

    }
    return;
  }

  if constexpr (tune_idx == 3 || tune_idx == 4) {

    constexpr bool stream = (tune_idx == 4);
    constexpr Index_type TILE = 256;
    const Index_type kp = m_domain->kp;
    const Index_type kfirst = ibegin / kp;
    const Index_type klast = (iend - 1) / kp;
    const Index_type nstrips = omp_get_max_threads();
    const Index_type chunk = 8 * (((kp + nstrips - 1) / nstrips + 7) / 8);

    // [lo, hi) with full aligned 64B lines of vol streamed from an L1 tile
    auto vol3d_range_nt = [=](Index_type lo, Index_type hi) {
      const Index_type a = RAJA_MIN(hi, lo + static_cast<Index_type>(
          ((64 - (reinterpret_cast<std::uintptr_t>(vol + lo) & 63)) & 63) /
          sizeof(Real_type)));
      for (Index_type i = lo; i < a; ++i) {
        VOL3D_BODY;
      }
      alignas(64) Real_type buf[TILE];
      for (Index_type t0 = a; t0 < hi; t0 += TILE) {
        const Index_type n = RAJA_MIN(TILE, hi - t0);
        {
          const Vol3dTile vol{buf, t0};
          for (Index_type i = t0; i < t0 + n; ++i) {
            VOL3D_BODY;
          }
        }
        const Index_type n8 = n & ~Index_type(7);
        fem_simd::stream_copy(vol + t0, buf, n8);
        for (Index_type k = n8; k < n; ++k) {
          vol[t0 + k] = buf[k];
        }
      }
    };

    // strip s: offsets [s*chunk, (s+1)*chunk) of every k-plane, planes in
    // order, so the k+1 plane of x, y, z is reused from L2 at the next k
    auto vol3d_strip = [=](Index_type s) {
      const Index_type lo = s * chunk;
      const Index_type hi = RAJA_MIN(lo + chunk, kp);
      for (Index_type k = kfirst; k <= klast && lo < hi; ++k) {
        const Index_type sb = RAJA_MAX(k * kp + lo, ibegin);
        const Index_type se = RAJA_MIN(k * kp + hi, iend);
        if (sb >= se) { continue; }
        if constexpr (stream) {
          vol3d_range_nt(sb, se);
        } else {
          for (Index_type i = sb; i < se; ++i) {
            VOL3D_BODY;
          }
        }
      }
      if constexpr (stream) { fem_simd::stream_fence(); }
    };
    RAJA_UNUSED_VAR(vol3d_range_nt);

    switch ( vid ) {

      case Base_OpenMP :
      case Lambda_OpenMP : {

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("VOL3D_1");
          #pragma omp parallel for schedule(static, 1)
          for (Index_type s = 0; s < nstrips; ++s) {
            vol3d_strip(s);
          }
          RP_CALI_SUBKERNEL_END("VOL3D_1");

        }
        stopTimer();

        break;
      }

      case RAJA_OpenMP : {

        auto res{getHostResource()};

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("VOL3D_1");
          RAJA::forall<RAJA::omp_parallel_for_static_exec<1>>( res,
            RAJA::RangeSegment(0, nstrips), vol3d_strip);
          RP_CALI_SUBKERNEL_END("VOL3D_1");

        }
        stopTimer();

        break;
      }

      default : {
        getCout() << "\n  VOL3D : Unknown variant id = " << vid << std::endl;
      }

    }
    return;
  }

  static_assert(tune_idx >= 1 && tune_idx <= 4, "unknown VOL3D tuning");
  if constexpr (tune_idx == 1) {

    constexpr Index_type TILE = 256;
    // first index >= ibegin whose vol entry starts a 64B line
    const Index_type a0 = RAJA_MIN(iend, ibegin + static_cast<Index_type>(
        ((64 - (reinterpret_cast<std::uintptr_t>(vol + ibegin) & 63)) & 63) /
        sizeof(Real_type)));
    const Index_type nchunks = omp_get_max_threads();
    const Index_type nlines = (iend - a0 + 7) / 8;
    const Index_type per = 8 * ((nlines + nchunks - 1) / nchunks);

    auto vol3d_nt_chunk = [=](Index_type c) {
      if (c == 0) {
        for (Index_type i = ibegin; i < a0; ++i) {
          VOL3D_BODY;
        }
      }
      const Index_type lo = RAJA_MIN(a0 + c * per, iend);
      const Index_type hi = RAJA_MIN(lo + per, iend);
      alignas(64) Real_type buf[TILE];
      for (Index_type t0 = lo; t0 < hi; t0 += TILE) {
        const Index_type n = RAJA_MIN(TILE, hi - t0);
        {
          const Vol3dTile vol{buf, t0};
          for (Index_type i = t0; i < t0 + n; ++i) {
            VOL3D_BODY;
          }
        }
        const Index_type n8 = n & ~Index_type(7);
        fem_simd::stream_copy(vol + t0, buf, n8);
        for (Index_type k = n8; k < n; ++k) {
          vol[t0 + k] = buf[k];
        }
      }
      fem_simd::stream_fence();
    };

    switch ( vid ) {

      case Base_OpenMP :
      case Lambda_OpenMP : {

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("VOL3D_1");
          #pragma omp parallel for schedule(static, 1)
          for (Index_type c = 0; c < nchunks; ++c) {
            vol3d_nt_chunk(c);
          }
          RP_CALI_SUBKERNEL_END("VOL3D_1");

        }
        stopTimer();

        break;
      }

      case RAJA_OpenMP : {

        auto res{getHostResource()};

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("VOL3D_1");
          RAJA::forall<RAJA::omp_parallel_for_static_exec<1>>( res,
            RAJA::RangeSegment(0, nchunks), vol3d_nt_chunk);
          RP_CALI_SUBKERNEL_END("VOL3D_1");

        }
        stopTimer();

        break;
      }

      default : {
        getCout() << "\n  VOL3D : Unknown variant id = " << vid << std::endl;
      }

    }
    return;
  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

void VOL3D::defineOpenMPVariantTunings()
{
  for (VariantID vid : {Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP}) {
    addVariantTuning<&VOL3D::runOpenMPVariant>(vid, getDefaultTuningName());
    addVariantTuning<&VOL3D::runOpenMPVariantTuned<2>>(vid, "blocked");
    addVariantTuning<&VOL3D::runOpenMPVariantTuned<3>>(vid, "strip");
#if defined(RAJAPERF_FEM_AVX512)
    addVariantTuning<&VOL3D::runOpenMPVariantTuned<1>>(vid, "nt");
    addVariantTuning<&VOL3D::runOpenMPVariantTuned<4>>(vid, "strip_nt");
#endif
  }
}

} // end namespace apps
} // end namespace rajaperf
