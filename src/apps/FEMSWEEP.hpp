//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// FEMSWEEP kernel reference implementation:
///
/// for (Index_type ag = 0; ag < na * ng; ++ag)
/// {
///   const Index_type a = ag / ng;
///   const Index_type g = ag % ng;
///   // number and offset of hyperplanes for this angle
///   const Index_type nhp = nhpaa_r[a];
///   const Index_type ohp = ohpaa_r[a];
///   // elements in this hyperplanes processed so far
///   Index_type s_nehp_done = 0;
///   Real_type A[ND*ND];
//    Real_type b[ND];
///   // This factor helps maintain stability in the solution of the matrix solve
///   // by eliminating the perturbation of the right-hand side.
///   Real_type Ffactor = fmax(sin(Adat[order_r[a*ne]*ND*ND + a*ne*ND*ND]) - 2.0, 0.0);
///   for (Index_type hp = 0; hp < nhp; ++hp) // loop over hyperplanes
///   {
///      // number of element in this hyperplane
///      const Index_type nehp = phpaa_r[ohp + hp];
///      // loop over all elements of this hyperplane
///      for (Index_type k = 0; k < nehp; ++k)
///      {
///         const Index_type e = order_r[k + s_nehp_done + a * ne];
///         for (Index_type j = 0; j < ND; ++j) // load B & A
///         {
///            b[j] = Bdat[j + e * ND + a * ne * ND];
///            for (Index_type i = 0; i < ND; ++i)
///            {
///               A[i + j * ND] = Adat[i + j * ND + e * ND * ND + a * ne * ND * ND];
///            }
///         }
///         for (Index_type face = 0; face < NLF; ++face) // local faces
///         {
///            const Index_type sf_gl = F_g2l[elem_to_faces[NLF * e + face]];
///            const Index_type f = sf_gl >= 0 ? sf_gl : -1 - sf_gl; // signed face
///            const Index_type s = (e == idx1[f * FDS] / ND) ? 0 : 1; // side
///            if ((AngleElem2FaceType[face + e * NLF + a * ne * NLF] == 0) || (sf_gl < 0)) // non-reentrant and outgoing, or boundary
///            {
///               continue;
///            }
///            for (Index_type j = 0; j < FDS; ++j) // face dofs
///            {
///               const Index_type ffj = f * FDS + j;
///               const Index_type djs = s == 0 ? idx1[ffj] : idx2[ffj];
///               Real_type F = 0.0;
///               for (Index_type i = 0; i < FDS; i++) // face dofs
///               {
///                  const Index_type ffi = f * FDS + i;
///                  const Index_type dis = s == 0 ? idx2[ffi] : idx1[ffi];
///                  // Fdat : 4 x 4 x 2 x nf_int x na
///                  // Note: s ^ 1 == 1 if s == 0, and s ^ 1 == 0 if s == 1.
///                  // The solution remains bounded when matrices and indirection arrays from a
///                  // finite element discretrization are used, but randomly generating matrix 
///                  // data can result in exponential growth instead. A specific trick of 
///                  // multiplying by zero, without the compiler optimizing it away, is 
///                  // necessary to prevent unbounded growth and avoid computing NaNs.
///                  F += Ffactor * Fdat[i + j * FDS + (s ^ 1) * FDS * FDS + f * 2 * FDS * FDS + a * sharedinteriorfaces * 2 * FDS * FDS] * Xdat[dis + g * ND * ne + a * ng * ND * ne];
///               } // i
///               b[djs % ND] -= F;
///            } // j
///         }    // local faces
///         const Real_type s = Sgdat[e + g * ne];
///         // A = A + s * M0, b is the result
///         SolveLinearSystemNxN<ND>(A, s, &M0dat[0 + 0 * ND + e * ND * ND], b, &Xdat[e * ND + g * ND * ne + a * ng * ND * ne]);  // 8x8 solve
///      }  // thread loop elems in hp
///      s_nehp_done += nehp;
///   } // hyperplanes
/// }
///
/// The RAJA variants of this kernel use RAJA multi-dimensional data layouts
/// and views to do the same thing without explicit index calculations (see
/// the loop body definitions below).
///

#ifndef RAJAPerf_Apps_FEMSWEEP_HPP
#define RAJAPerf_Apps_FEMSWEEP_HPP

#include <atomic>
#include <cstdint>
#include <vector>

#include "common/KernelBase.hpp"

#include "RAJA/RAJA.hpp"

#include "FEMSWEEPMeshGen.hpp"

#include "FEM_MACROS.hpp"
#include "FEM_SIMD.hpp"

constexpr long ND = 8;   // number of corners per element
constexpr long NLF = 6;  // number of faces per element
constexpr long FDS = 4;  // number of DOFs per face

#define FEMSWEEP_DATA_SETUP \
  Real_ptr Bdat = m_Bdat; \
  Real_ptr Adat = m_Adat; \
  Real_ptr Fdat = m_Fdat; \
  Real_ptr Xdat = m_Xdat; \
  Real_ptr Sgdat = m_Sgdat; \
  Real_ptr M0dat = m_M0dat; \
\
  Index_type ne = m_ne; \
  Index_type na = m_na; \
  Index_type ng = m_ng; \
  Index_type sharedinteriorfaces = m_sharedinteriorfaces; \
\
  Index_ptr nhpaa_r = m_nhpaa_r;  \
  Index_ptr ohpaa_r = m_ohpaa_r;  \
  Index_ptr phpaa_r = m_phpaa_r;  \
  Index_ptr order_r = m_order_r;  \
\
  Index_ptr AngleElem2FaceType = m_AngleElem2FaceType; \
  Index_ptr elem_to_faces      = m_elem_to_faces     ; \
  Index_ptr F_g2l              = m_F_g2l             ; \
  Index_ptr idx1               = m_idx1              ; \
  Index_ptr idx2               = m_idx2              ; \


#define FEMSWEEP_KERNEL_SETUP \
  const Index_type nhp = nhpaa_r[a]; \
  const Index_type ohp = ohpaa_r[a]; \
  Real_type Ffactor = fmax(sin(Adat[order_r[a*ne]*ND*ND + a*ne*ND*ND]) - 2.0, 0.0);

#define FEMSWEEP_KERNEL_HYPERPLANE_ELEMENT \
  Real_type A[ND*ND]; \
  Real_type b[ND]; \
  const Index_type e = order_r[k + nehp_pos + a * ne]; \
  for (Index_type j = 0; j < ND; ++j) \
  { \
    b[j] = Bdat[j + e * ND + a * ne * ND]; \
    for (Index_type i = 0; i < ND; ++i) \
    { \
      A[i + j * ND] = Adat[i + j * ND + e * ND * ND + a * ne * ND * ND]; \
    } \
  } \
  for (Index_type face = 0; face < NLF; ++face) \
  { \
    const Index_type sf_gl = F_g2l[elem_to_faces[NLF * e + face]]; \
    const Index_type f = sf_gl >= 0 ? sf_gl : -1 - sf_gl; \
    const Index_type s = (e == idx1[f * FDS] / ND) ? 0 : 1; \
    if ((AngleElem2FaceType[face + e * NLF + a * ne * NLF] == 0) || (sf_gl < 0)) \
    { \
      continue; \
    } \
    for (Index_type j = 0; j < FDS; ++j) \
    { \
      const Index_type ffj = f * FDS + j; \
      const Index_type djs = s == 0 ? idx1[ffj] : idx2[ffj]; \
      Real_type F = 0.0; \
      for (Index_type i = 0; i < FDS; i++) \
      { \
        const Index_type ffi = f * FDS + i; \
        const Index_type dis = s == 0 ? idx2[ffi] : idx1[ffi]; \
        F += Ffactor * Fdat[i + j * FDS + (s ^ 1) * FDS * FDS + f * 2 * FDS * FDS + a * sharedinteriorfaces * 2 * FDS * FDS] * Xdat[dis + g * ND * ne + a * ng * ND * ne]; \
      } \
      b[djs % ND] -= F; \
    } \
  } \
  const Real_type s = Sgdat[e + g * ne]; \
  SolveLinearSystemNxN<ND>(A, s, &M0dat[0 + 0 * ND + e * ND * ND], b, &Xdat[e * ND + g * ND * ne + a * ng * ND * ne]);

//
// CPU SoA variant: batch of FEM_VL energy groups [g0, g0+FEM_VL) of angle a,
// group (lane) index innermost. A, M0, F and all mesh connectivity depend
// only on (a, e), so control flow is identical across lanes.
//
#define FEMSWEEP_KERNEL_SETUP_SIMD \
  FEMSWEEP_KERNEL_SETUP \
  FEM_BATCH_SETUP(g0, ng);

#define FEMSWEEP_KERNEL_HYPERPLANE_ELEMENT_SIMD \
  const Index_type e = order_r[k + nehp_pos + a * ne]; \
  FEMSWEEP_ELEMENT_SIMD_BODY

#define FEMSWEEP_ELEMENT_SIMD_BODY \
  Real_type b[ND][FEM_VL]; \
  for (Index_type j = 0; j < ND; ++j) \
  { \
    const Real_type bj = Bdat[j + e * ND + a * ne * ND]; \
    CPU_FOREACH_LANE(l) { b[j][l] = bj; } \
  } \
  for (Index_type face = 0; face < NLF; ++face) \
  { \
    const Index_type sf_gl = F_g2l[elem_to_faces[NLF * e + face]]; \
    const Index_type f = sf_gl >= 0 ? sf_gl : -1 - sf_gl; \
    const Index_type s = (e == idx1[f * FDS] / ND) ? 0 : 1; \
    if ((AngleElem2FaceType[face + e * NLF + a * ne * NLF] == 0) || (sf_gl < 0)) \
    { \
      continue; \
    } \
    for (Index_type j = 0; j < FDS; ++j) \
    { \
      const Index_type ffj = f * FDS + j; \
      const Index_type djs = s == 0 ? idx1[ffj] : idx2[ffj]; \
      Real_type F[FEM_VL]; \
      CPU_FOREACH_LANE(l) { F[l] = 0.0; } \
      for (Index_type i = 0; i < FDS; i++) \
      { \
        const Index_type ffi = f * FDS + i; \
        const Index_type dis = s == 0 ? idx2[ffi] : idx1[ffi]; \
        const Real_type Fij = Ffactor * Fdat[i + j * FDS + (s ^ 1) * FDS * FDS + f * 2 * FDS * FDS + a * sharedinteriorfaces * 2 * FDS * FDS]; \
        /* gather AoS -> SoA */ \
        CPU_FOREACH_LANE(l) { F[l] += Fij * Xdat[dis + le[l] * ND * ne + a * ng * ND * ne]; } \
      } \
      CPU_FOREACH_LANE(l) { b[djs % ND][l] -= F[l]; } \
    } \
  } \
  Real_type sg[FEM_VL]; \
  CPU_FOREACH_LANE(l) { sg[l] = Sgdat[e + le[l] * ne]; } \
  Real_type xs[ND][FEM_VL]; \
  SolveLinearSystemNxN_simd<ND>(&Adat[e * ND * ND + a * ne * ND * ND], sg, &M0dat[e * ND * ND], b, xs); \
  /* scatter SoA -> AoS (valid lanes only) */ \
  for (Index_type ii = 0; ii < ND; ++ii) \
  { \
    CPU_FOREACH_LANE(l) { \
      if (l < nel) { Xdat[ii + e * ND + (g0 + l) * ND * ne + a * ng * ND * ne] = xs[ii][l]; } \
    } \
  }


namespace rajaperf
{

// LU factorization with no pivoting
template <long N>
RAJA_HOST_DEVICE inline void SolveLinearSystemNxN(Real_ptr A, 
                                                  const Real_type s,
                                                  Real_const_ptr M, 
                                                  Real_ptr b, 
                                                  Real_ptr x)
{
  Real_type tempA[N][N];
  Real_type L[N][N];
  Real_type U[N][N];
  Real_type D[N];
  Real_type tempx[N];

  // tempA = A + s * M0
  // set L to 0, U to identity
  for ( Index_type ii = 0; ii < N; ++ii )
  {
    for ( Index_type jj = 0; jj < N; ++jj )
    {
      tempA[ii][jj] = A[ii * N + jj] + s * M[ii * N + jj];
      L[ii][jj] = 0.0;
      if ( ii == jj )
      {
        U[ii][jj] = 1.0;
      }
      else
      {
        U[ii][jj] = 0.0;
      }
    }
  }

  // set first column of L, and first row of U
  for ( Index_type ii = 0; ii < N; ++ii )
  {
    L[ii][0] = tempA[ii][0];
  }

  for ( Index_type ii = 1; ii < N; ++ii )
  {
    U[0][ii] = tempA[0][ii]/tempA[0][0];
  }

  // form L & U
  // L formed one column at a time
  // U formed one row at a time
  for ( Index_type ii = 1; ii < N; ++ii )
  {
    // L column formation
    for ( Index_type jj = ii; jj < N; ++jj )
    {
      Real_type sum = 0.0;
      for ( Index_type kk = 0; kk < jj; ++kk )
      {
        sum += L[jj][kk] * U[kk][ii];
      }
      L[jj][ii] = tempA[jj][ii] - sum;
    }

    // U row formation
    for ( Index_type jj = ii+1; jj < N; ++jj )
    {
      Real_type sum = 0.0;
      for ( Index_type kk = 0; kk < ii; ++kk )
      {
        sum += L[ii][kk] * U[kk][jj];
      }
      U[ii][jj] = (tempA[ii][jj] - sum)/L[ii][ii];
    }
  }

  // forward substitution
  D[0] = b[0]/L[0][0];
  for ( Index_type ii = 1; ii < N; ++ii )
  {
    Real_type sum = 0.0;
    for ( Index_type jj = 0; jj < ii; ++jj )
    {
      sum += L[ii][jj] * D[jj];
    }
    D[ii] = (b[ii] - sum)/L[ii][ii];
  }

  // backward substitution
  x[N-1] = tempx[N-1] = D[N-1];
  for ( Index_type ii = N - 1 - 1; ii > -1; --ii )
  {
    Real_type sum = 0.0;
    for ( Index_type jj = ii+1; jj < N; ++jj )
    {
      sum += U[ii][jj] * tempx[jj];
    }
    x[ii] = tempx[ii] = D[ii] - sum;
  }

}

// SolveLinearSystemNxN for FEM_VL independent systems sharing A and M,
// system (lane) index innermost.
template <long N>
RAJA_INLINE void SolveLinearSystemNxN_simd(Real_const_ptr A,
                                           const Real_type (&s)[FEM_VL],
                                           Real_const_ptr M,
                                           const Real_type (&b)[N][FEM_VL],
                                           Real_type (&x)[N][FEM_VL])
{
  // In-place Crout LU: LU[i][j] holds L for j <= i and unit-diagonal U for j > i.
  // Same operation order as SolveLinearSystemNxN minus its products with
  // known-zero entries; one reciprocal per diagonal replaces repeated divides.
#if defined(RAJAPERF_FEM_AVX512)
  using namespace fem_simd;
  // fully unrolled: all N*N LU entries are SSA values, no lane loops
  vd LU[N][N];
  vd rinv[N];
  vd D[N];
  const vd vs = vload(s);

  RAJAPERF_FEM_UNROLL
  for ( Index_type ii = 0; ii < N; ++ii )
  {
    RAJAPERF_FEM_UNROLL
    for ( Index_type jj = 0; jj < N; ++jj )
    {
      LU[ii][jj] = vfma(vs, vset1(M[ii * N + jj]), vset1(A[ii * N + jj]));
    }
  }

  RAJAPERF_FEM_UNROLL
  for ( Index_type ii = 0; ii < N; ++ii )
  {
    RAJAPERF_FEM_UNROLL
    for ( Index_type jj = ii; jj < N; ++jj )
    {
      vd sum = vzero();
      RAJAPERF_FEM_UNROLL
      for ( Index_type kk = 0; kk < ii; ++kk )
      {
        sum = vfma(LU[jj][kk], LU[kk][ii], sum);
      }
      LU[jj][ii] = vsub(LU[jj][ii], sum);
    }

    rinv[ii] = vdiv(vset1(1.0), LU[ii][ii]);

    RAJAPERF_FEM_UNROLL
    for ( Index_type jj = ii+1; jj < N; ++jj )
    {
      vd sum = vzero();
      RAJAPERF_FEM_UNROLL
      for ( Index_type kk = 0; kk < ii; ++kk )
      {
        sum = vfma(LU[ii][kk], LU[kk][jj], sum);
      }
      LU[ii][jj] = vmul(vsub(LU[ii][jj], sum), rinv[ii]);
    }
  }

  RAJAPERF_FEM_UNROLL
  for ( Index_type ii = 0; ii < N; ++ii )
  {
    vd sum = vzero();
    RAJAPERF_FEM_UNROLL
    for ( Index_type jj = 0; jj < ii; ++jj )
    {
      sum = vfma(LU[ii][jj], D[jj], sum);
    }
    D[ii] = vmul(vsub(vload(b[ii]), sum), rinv[ii]);
  }

  vstore(x[N-1], D[N-1]);
  vd xv[N];
  xv[N-1] = D[N-1];
  RAJAPERF_FEM_UNROLL
  for ( Index_type ii = N - 1 - 1; ii > -1; --ii )
  {
    vd sum = vzero();
    RAJAPERF_FEM_UNROLL
    for ( Index_type jj = ii+1; jj < N; ++jj )
    {
      sum = vfma(LU[ii][jj], xv[jj], sum);
    }
    xv[ii] = vsub(D[ii], sum);
    vstore(x[ii], xv[ii]);
  }
#else
  Real_type LU[N][N][FEM_VL];
  Real_type rinv[N][FEM_VL];
  Real_type D[N][FEM_VL];
  Real_type sum[FEM_VL];

  // LU = A + s * M0
  for ( Index_type ii = 0; ii < N; ++ii )
  {
    for ( Index_type jj = 0; jj < N; ++jj )
    {
      const Real_type Aij = A[ii * N + jj];
      const Real_type Mij = M[ii * N + jj];
      CPU_FOREACH_LANE(l) { LU[ii][jj][l] = Aij + s[l] * Mij; }
    }
  }

  for ( Index_type ii = 0; ii < N; ++ii )
  {
    // L column formation
    for ( Index_type jj = ii; jj < N; ++jj )
    {
      CPU_FOREACH_LANE(l) { sum[l] = 0.0; }
      for ( Index_type kk = 0; kk < ii; ++kk )
      {
        CPU_FOREACH_LANE(l) { sum[l] += LU[jj][kk][l] * LU[kk][ii][l]; }
      }
      CPU_FOREACH_LANE(l) { LU[jj][ii][l] = LU[jj][ii][l] - sum[l]; }
    }

    CPU_FOREACH_LANE(l) { rinv[ii][l] = 1.0 / LU[ii][ii][l]; }

    // U row formation
    for ( Index_type jj = ii+1; jj < N; ++jj )
    {
      CPU_FOREACH_LANE(l) { sum[l] = 0.0; }
      for ( Index_type kk = 0; kk < ii; ++kk )
      {
        CPU_FOREACH_LANE(l) { sum[l] += LU[ii][kk][l] * LU[kk][jj][l]; }
      }
      CPU_FOREACH_LANE(l) { LU[ii][jj][l] = (LU[ii][jj][l] - sum[l]) * rinv[ii][l]; }
    }
  }

  // forward substitution
  for ( Index_type ii = 0; ii < N; ++ii )
  {
    CPU_FOREACH_LANE(l) { sum[l] = 0.0; }
    for ( Index_type jj = 0; jj < ii; ++jj )
    {
      CPU_FOREACH_LANE(l) { sum[l] += LU[ii][jj][l] * D[jj][l]; }
    }
    CPU_FOREACH_LANE(l) { D[ii][l] = (b[ii][l] - sum[l]) * rinv[ii][l]; }
  }

  // backward substitution
  CPU_FOREACH_LANE(l) { x[N-1][l] = D[N-1][l]; }
  for ( Index_type ii = N - 1 - 1; ii > -1; --ii )
  {
    CPU_FOREACH_LANE(l) { sum[l] = 0.0; }
    for ( Index_type jj = ii+1; jj < N; ++jj )
    {
      CPU_FOREACH_LANE(l) { sum[l] += LU[ii][jj][l] * x[jj][l]; }
    }
    CPU_FOREACH_LANE(l) { x[ii][l] = D[ii][l] - sum[l]; }
  }
#endif
}

// Spin barrier for a team of concurrently running threads.
struct alignas(64) FemsweepTeamBarrier
{
  std::atomic<int> count{0};
  std::atomic<int> sense{0};

  void wait(int nthreads, int& local_sense)
  {
    local_sense ^= 1;
    if (count.fetch_add(1, std::memory_order_acq_rel) == nthreads - 1) {
      count.store(0, std::memory_order_relaxed);
      sense.store(local_sense, std::memory_order_release);
    } else {
      while (sense.load(std::memory_order_acquire) != local_sense) {
#if defined(__x86_64__)
        __builtin_ia32_pause();
#endif
      }
    }
  }
};

// Team size (1,2,4,8) minimizing waves/team_size for ntasks sweeps on nthreads.
inline void femsweep_team_shape(Index_type ntasks, int nthreads,
                                int& team_size, int& nteams)
{
  double best = 0.0;
  team_size = 1;
  for (int ts = 1; ts <= 8 && ts <= nthreads; ts *= 2) {
    const Index_type nt = nthreads / ts;
    const double cost = static_cast<double>((ntasks + nt - 1) / nt) / ts;
    if (ts == 1 || cost < best - 1e-12) {
      best = cost;
      team_size = ts;
    }
  }
  nteams = nthreads / team_size;
}

//
// Sweeps of all (angle, group batch) tasks by thread 'slot': teams of
// team_size threads share one sweep, splitting each hyperplane's elements
// and synchronizing once per hyperplane.
//
#define FEMSWEEP_TEAM_SWEEP(slot) \
  { \
    const int team = (slot) / team_size; \
    const int rank = (slot) % team_size; \
    if (team < nteams) { \
      int local_sense = team_barriers[team].sense.load(std::memory_order_relaxed); \
      for (Index_type t = team; t < na * nbg; t += nteams) { \
        const Index_type a = t / nbg; \
        const Index_type g0 = (t % nbg) * FEM_VL; \
        FEMSWEEP_KERNEL_SETUP_SIMD; \
        Index_type nehp_pos = 0; \
        for (Index_type hp = 0; hp < nhp; ++hp) { \
          const Index_type nehp = phpaa_r[ohp + hp]; \
          for (Index_type k = rank; k < nehp; k += team_size) { \
            FEMSWEEP_KERNEL_HYPERPLANE_ELEMENT_SIMD; \
          } \
          if (team_size > 1) { team_barriers[team].wait(team_size, local_sense); } \
          nehp_pos += nehp; \
        } \
      } \
    } \
  }

// Angle-independent data of local face 'face' of element e (entry NLF*e +
// face), precomputed in setUp: what femsweep_element otherwise derives
// through the dependent loads elem_to_faces -> F_g2l -> idx1/idx2.
struct FemsweepFace
{
  int32_t fs;         // 2 * interior face index + side s, or -1 (boundary face)
  int32_t eu;         // upwind (neighbor) element
  uint8_t src[FDS];   // local dofs of eu holding the face unknowns
  uint8_t dst[FDS];   // local dofs of e receiving the face flux
};

// Fills ftab[NLF * ne]. Returns false if a face does not fit the table (dofs
// of one face spanning elements, or indices beyond int32).
inline bool femsweep_build_faces(Index_type ne, const Index_type* elem_to_faces,
                                 const Index_type* F_g2l, const Index_type* idx1,
                                 const Index_type* idx2, FemsweepFace* ftab)
{
  for (Index_type e = 0; e < ne; ++e) {
    for (Index_type face = 0; face < NLF; ++face) {
      FemsweepFace& fc = ftab[NLF * e + face];
      const Index_type sf_gl = F_g2l[elem_to_faces[NLF * e + face]];
      fc.fs = -1;
      fc.eu = 0;
      for (Index_type i = 0; i < FDS; ++i) { fc.src[i] = fc.dst[i] = 0; }
      if (sf_gl < 0) {
        continue;
      }
      const Index_type s = (e == idx1[sf_gl * FDS] / ND) ? 0 : 1;
      const Index_type *const dsrc = (s == 0 ? idx2 : idx1) + sf_gl * FDS;
      const Index_type *const ddst = (s == 0 ? idx1 : idx2) + sf_gl * FDS;
      const Index_type eu = dsrc[0] / ND;
      if (2 * sf_gl + 1 > INT32_MAX || eu > INT32_MAX) {
        return false;
      }
      for (Index_type i = 0; i < FDS; ++i) {
        if (dsrc[i] / ND != eu) {
          return false;
        }
        fc.src[i] = static_cast<uint8_t>(dsrc[i] - eu * ND);
        fc.dst[i] = static_cast<uint8_t>(ddst[i] % ND);
      }
      fc.fs = static_cast<int32_t>(2 * sf_gl + s);
      fc.eu = static_cast<int32_t>(eu);
    }
  }
  return true;
}

// Kernel data for femsweep_element.
struct FemsweepData
{
  Real_const_ptr Bdat, Adat, Fdat, Sgdat, M0dat;
  Real_ptr Xdat;
  const Index_type *AngleElem2FaceType, *elem_to_faces, *F_g2l, *idx1, *idx2;
  Index_type ne, ng, sharedinteriorfaces;
  const FemsweepFace *ftab;   // only used by the FT code paths
};

// Prefetch what femsweep_element<FT> reads and writes for element e.
template <bool FT = false>
RAJA_INLINE void femsweep_prefetch(const FemsweepData& d, const Index_type e,
                                   const Index_type a, const Index_type g0,
                                   const Index_type nel)
{
  const Index_type ne = d.ne;
  const Real_type *const A = d.Adat + e * ND * ND + a * ne * ND * ND;
  const Real_type *const M = d.M0dat + e * ND * ND;
  for (Index_type c = 0; c < ND * ND; c += 8) {
    __builtin_prefetch(A + c, 0, 3);
    __builtin_prefetch(M + c, 0, 3);
  }
  __builtin_prefetch(d.Bdat + e * ND + a * ne * ND, 0, 3);
  const Index_type *const ft = d.AngleElem2FaceType + e * NLF + a * ne * NLF;
  if constexpr (FT) {
    __builtin_prefetch(d.ftab + NLF * e, 0, 3);
    __builtin_prefetch(d.ftab + NLF * e + NLF - 1, 0, 3);
  }
  for (Index_type face = 0; face < NLF; ++face) {
    Index_type sf_gl, s;
    if constexpr (FT) {
      const int32_t fs = d.ftab[NLF * e + face].fs;
      sf_gl = fs >> 1;
      s = fs & 1;
    } else {
      sf_gl = d.F_g2l[d.elem_to_faces[NLF * e + face]];
    }
    if (ft[face] == 0 || sf_gl < 0) {
      continue;
    }
    if constexpr (!FT) {
      s = (e == d.idx1[sf_gl * FDS] / ND) ? 0 : 1;
    }
    const Real_type *const Ff = d.Fdat + (s ^ 1) * FDS * FDS + sf_gl * 2 * FDS * FDS +
                                a * d.sharedinteriorfaces * 2 * FDS * FDS;
    __builtin_prefetch(Ff, 0, 3);
    __builtin_prefetch(Ff + 8, 0, 3);
  }
  Real_type *const X = d.Xdat + (a * d.ng + g0) * ND * ne + e * ND;
  for (Index_type l = 0; l < nel; ++l) {
    __builtin_prefetch(X + l * ND * ne, 1, 3);
    __builtin_prefetch(d.Sgdat + e + (g0 + l) * ne, 0, 3);
  }
}

//
// FEMSWEEP_ELEMENT_SIMD_BODY for element e (angle a, groups [g0, g0+nel)) in
// an out-of-line function with an aligned frame: the LU spills to aligned
// slots, face unknowns are gathered once per face, and x is written as one
// contiguous row per group.
// FT: face data from the precomputed d.ftab instead of the index chain.
//
template <bool FT = false>
RAJAPERF_FEM_NOINLINE void femsweep_element(const FemsweepData& d,
                                            const Index_type e,
                                            const Index_type a,
                                            const Index_type g0,
                                            const Index_type nel,
                                            const Real_type Ffactor)
{
  const Index_type ne = d.ne, ng = d.ng;
  const Index_type sharedinteriorfaces = d.sharedinteriorfaces;
  const Index_type *const AngleElem2FaceType = d.AngleElem2FaceType;
  const Index_type *const elem_to_faces = d.elem_to_faces;
  const Index_type *const F_g2l = d.F_g2l;
  const Index_type *const idx1 = d.idx1;
  const Index_type *const idx2 = d.idx2;
#if defined(RAJAPERF_FEM_AVX512)
  using namespace fem_simd;
  const Index_type xa = a * ng * ND * ne;
  // lane l -> group g0 + min(l, nel - 1)
  const __m512i lg = _mm512_add_epi64(
    _mm512_min_epi64(_mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0),
                     _mm512_set1_epi64(nel - 1)),
    _mm512_set1_epi64(g0));
  const __m512i xoff = _mm512_mullo_epi64(lg, _mm512_set1_epi64(ND * ne));
  const __m512i soff = _mm512_mullo_epi64(lg, _mm512_set1_epi64(ne));

  alignas(64) vd b[ND];
  RAJAPERF_FEM_STACK_ALIGN(b);
  for (Index_type j = 0; j < ND; ++j) {
    b[j] = vset1(d.Bdat[j + e * ND + a * ne * ND]);
  }
  for (Index_type face = 0; face < NLF; ++face) {
    if constexpr (FT) {
      const FemsweepFace fc = d.ftab[NLF * e + face];
      if ((AngleElem2FaceType[face + e * NLF + a * ne * NLF] == 0) || (fc.fs < 0)) {
        continue;
      }
      const Index_type f = fc.fs >> 1;
      const Index_type s = fc.fs & 1;
      const Real_type *const Ff = d.Fdat + (s ^ 1) * FDS * FDS + f * 2 * FDS * FDS +
                                  a * sharedinteriorfaces * 2 * FDS * FDS;
      const Real_type *const xu = d.Xdat + xa + Index_type(fc.eu) * ND;
      vd xg[FDS];
      RAJAPERF_FEM_UNROLL
      for (Index_type i = 0; i < FDS; i++) {
        xg[i] = _mm512_i64gather_pd(xoff, xu + fc.src[i], 8);
      }
      RAJAPERF_FEM_UNROLL
      for (Index_type j = 0; j < FDS; ++j) {
        vd F = vzero();
        RAJAPERF_FEM_UNROLL
        for (Index_type i = 0; i < FDS; i++) {
          F = vfma(vset1(Ffactor * Ff[i + j * FDS]), xg[i], F);
        }
        b[fc.dst[j]] = vsub(b[fc.dst[j]], F);
      }
      continue;
    }
    const Index_type sf_gl = F_g2l[elem_to_faces[NLF * e + face]];
    const Index_type f = sf_gl >= 0 ? sf_gl : -1 - sf_gl;
    const Index_type s = (e == idx1[f * FDS] / ND) ? 0 : 1;
    if ((AngleElem2FaceType[face + e * NLF + a * ne * NLF] == 0) || (sf_gl < 0)) {
      continue;
    }
    const Real_type *const Ff = d.Fdat + (s ^ 1) * FDS * FDS + f * 2 * FDS * FDS +
                                a * sharedinteriorfaces * 2 * FDS * FDS;
    vd xg[FDS];
    RAJAPERF_FEM_UNROLL
    for (Index_type i = 0; i < FDS; i++) {
      const Index_type dis = s == 0 ? idx2[f * FDS + i] : idx1[f * FDS + i];
      xg[i] = _mm512_i64gather_pd(xoff, d.Xdat + dis + xa, 8);
    }
    RAJAPERF_FEM_UNROLL
    for (Index_type j = 0; j < FDS; ++j) {
      const Index_type djs = s == 0 ? idx1[f * FDS + j] : idx2[f * FDS + j];
      vd F = vzero();
      RAJAPERF_FEM_UNROLL
      for (Index_type i = 0; i < FDS; i++) {
        F = vfma(vset1(Ffactor * Ff[i + j * FDS]), xg[i], F);
      }
      b[djs % ND] = vsub(b[djs % ND], F);
    }
  }

  // SolveLinearSystemNxN_simd
  const vd vs = _mm512_i64gather_pd(soff, d.Sgdat + e, 8);
  const Real_type *const A = d.Adat + e * ND * ND + a * ne * ND * ND;
  const Real_type *const M = d.M0dat + e * ND * ND;
  vd LU[ND][ND];
  RAJAPERF_FEM_UNROLL
  for (Index_type ii = 0; ii < ND; ++ii) {
    RAJAPERF_FEM_UNROLL
    for (Index_type jj = 0; jj < ND; ++jj) {
      LU[ii][jj] = vfma(vs, vset1(M[ii * ND + jj]), vset1(A[ii * ND + jj]));
    }
  }
  vd rinv[ND];
  RAJAPERF_FEM_UNROLL
  for (Index_type ii = 0; ii < ND; ++ii) {
    RAJAPERF_FEM_UNROLL
    for (Index_type jj = ii; jj < ND; ++jj) {
      vd sum = vzero();
      RAJAPERF_FEM_UNROLL
      for (Index_type kk = 0; kk < ii; ++kk) {
        sum = vfma(LU[jj][kk], LU[kk][ii], sum);
      }
      LU[jj][ii] = vsub(LU[jj][ii], sum);
    }
    rinv[ii] = vdiv(vset1(1.0), LU[ii][ii]);
    RAJAPERF_FEM_UNROLL
    for (Index_type jj = ii + 1; jj < ND; ++jj) {
      vd sum = vzero();
      RAJAPERF_FEM_UNROLL
      for (Index_type kk = 0; kk < ii; ++kk) {
        sum = vfma(LU[ii][kk], LU[kk][jj], sum);
      }
      LU[ii][jj] = vmul(vsub(LU[ii][jj], sum), rinv[ii]);
    }
  }
  vd D[ND];
  RAJAPERF_FEM_UNROLL
  for (Index_type ii = 0; ii < ND; ++ii) {
    vd sum = vzero();
    RAJAPERF_FEM_UNROLL
    for (Index_type jj = 0; jj < ii; ++jj) {
      sum = vfma(LU[ii][jj], D[jj], sum);
    }
    D[ii] = vmul(vsub(b[ii], sum), rinv[ii]);
  }
  vd xv[ND];
  xv[ND - 1] = D[ND - 1];
  RAJAPERF_FEM_UNROLL
  for (Index_type ii = ND - 2; ii > -1; --ii) {
    vd sum = vzero();
    RAJAPERF_FEM_UNROLL
    for (Index_type jj = ii + 1; jj < ND; ++jj) {
      sum = vfma(LU[ii][jj], xv[jj], sum);
    }
    xv[ii] = vsub(D[ii], sum);
  }

  // lane (group) l of x is the contiguous row Xdat[e*ND .. e*ND+ND) of group g0+l
  static_assert(ND == FEM_VL, "row transpose needs ND == FEM_VL");
  transpose8(xv);
  Real_type *const xe = d.Xdat + xa + g0 * ND * ne + e * ND;
  for (Index_type l = 0; l < nel; ++l) {
    vstore(xe + l * ND * ne, xv[l]);
  }
#else
  RAJA_UNUSED_VAR(nel);
  {
  Real_const_ptr Bdat = d.Bdat;
  Real_const_ptr Adat = d.Adat;
  Real_const_ptr Fdat = d.Fdat;
  Real_const_ptr Sgdat = d.Sgdat;
  Real_const_ptr M0dat = d.M0dat;
  Real_ptr Xdat = d.Xdat;
  FEM_BATCH_SETUP(g0, ng);
  FEMSWEEP_ELEMENT_SIMD_BODY
  }
#endif
}

// FEMSWEEP_TEAM_SWEEP with femsweep_element. Element data is scattered in
// memory (DRAM latency bound), so the element pfd steps ahead in the sweep
// order is prefetched first. Tasks [t_begin, t_end) are shared by nt_ teams of
// ts_ threads with barriers bar_; ft selects the precomputed face table.
#define FEMSWEEP_PF 2
#define FEMSWEEP_TEAM_SWEEP_RANGE(slot, t_begin, t_end, ts_, nt_, bar_, pfd, ft) \
  { \
    const int team = (slot) / (ts_); \
    const int rank = (slot) % (ts_); \
    if (team < (nt_)) { \
      int local_sense = (bar_)[team].sense.load(std::memory_order_relaxed); \
      for (Index_type t = (t_begin) + team; t < (t_end); t += (nt_)) { \
        const Index_type a = t / nbg; \
        const Index_type g0 = (t % nbg) * FEM_VL; \
        FEMSWEEP_KERNEL_SETUP; \
        const Index_type nel = (ng - g0 < FEM_VL) ? ng - g0 : FEM_VL; \
        const Index_type* const orda = order_r + a * ne; \
        Index_type nehp_pos = 0; \
        for (Index_type hp = 0; hp < nhp; ++hp) { \
          const Index_type nehp = phpaa_r[ohp + hp]; \
          for (Index_type k = rank; k < nehp; k += (ts_)) { \
            const Index_type p = nehp_pos + k + (pfd) * (ts_); \
            if (p < ne) { femsweep_prefetch<ft>(fd, orda[p], a, g0, nel); } \
            femsweep_element<ft>(fd, orda[nehp_pos + k], a, g0, nel, Ffactor); \
          } \
          if ((ts_) > 1) { (bar_)[team].wait((ts_), local_sense); } \
          nehp_pos += nehp; \
        } \
      } \
    } \
  }

#define FEMSWEEP_TEAM_SWEEP_PF(slot) \
  FEMSWEEP_TEAM_SWEEP_RANGE(slot, 0, na * nbg, team_size, nteams, team_barriers, \
                            FEMSWEEP_PF, false)


class RunParams;

namespace apps
{

class FEMSWEEP : public KernelBase
{
public:

  FEMSWEEP(const RunParams& params);

  ~FEMSWEEP();

  void setSize(Index_type target_size, Index_type target_reps);
  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void defineSeqVariantTunings();
  void defineOpenMPVariantTunings();
  void defineCudaVariantTunings();
  void defineHipVariantTunings();

  void runSeqVariant(VariantID vid);
  template < size_t tune_idx >
  void runOpenMPVariant(VariantID vid);

  const FemsweepFace* faceTable() const { return m_ftab_ok ? m_ftab.data() : nullptr; }

  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);

private:
#if defined(RAJA_ENABLE_HIP)
  static const size_t default_gpu_block_size = 64;
#else
  static const size_t default_gpu_block_size = 128;
#endif
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size,
                                                         integer::MultipleOf<32>>;

  std::vector<FemsweepFace> m_ftab;   // see femsweep_build_faces
  bool m_ftab_ok = false;

  AngularQuadratureLite * m_angularquadrature;
  MeshGenerator * m_meshgen;

  Index_type m_nx;
  Index_type m_ny;
  Index_type m_nz;
  Index_type m_ne;
  Index_type m_na;
  Index_type m_ng;

  Index_type m_sharedinteriorfaces;
  Index_type m_boundaryfaces;
  Index_type m_hplanes;

  Real_ptr m_Bdat;
  Real_ptr m_Adat;
  Real_ptr m_Fdat;
  Real_ptr m_Sgdat;
  Real_ptr m_M0dat;
  Real_ptr m_Xdat;

  Index_type m_Blen;
  Index_type m_Alen;
  Index_type m_Flen;
  Index_type m_Sglen;
  Index_type m_M0len;
  Index_type m_Xlen;

  Index_type m_nhpaa_rlen;
  Index_type m_ohpaa_rlen;
  Index_type m_phpaa_rlen;
  Index_type m_order_rlen;

  Index_type m_AngleElem2FaceTypelen;
  Index_type m_elem_to_faceslen;
  Index_type m_F_g2llen;
  Index_type m_idx1len;
  Index_type m_idx2len;

  // Mesh data
  Index_ptr m_nhpaa_r;
  Index_ptr m_ohpaa_r;
  Index_ptr m_phpaa_r;
  Index_ptr m_order_r;

  Index_ptr m_AngleElem2FaceType;
  Index_ptr m_elem_to_faces     ;
  Index_ptr m_F_g2l             ;
  Index_ptr m_idx1              ;
  Index_ptr m_idx2              ;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
