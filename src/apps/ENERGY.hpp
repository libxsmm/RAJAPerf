//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// ENERGY kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   e_new[i] = e_old[i] - 0.5 * delvc[i] *
///              (p_old[i] + q_old[i]) + 0.5 * work[i];
/// }
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   if ( delvc[i] > 0.0 ) {
///      q_new[i] = 0.0 ;
///   }
///   else {
///      Real_type vhalf = 1.0 / (1.0 + compHalfStep[i]) ;
///      Real_type ssc = ( pbvc[i] * e_new[i]
///         + vhalf * vhalf * bvc[i] * pHalfStep[i] ) / rho0 ;
///      if ( ssc <= 0.1111111e-36 ) {
///         ssc = 0.3333333e-18 ;
///      } else {
///         ssc = sqrt(ssc) ;
///      }
///      q_new[i] = (ssc*ql_old[i] + qq_old[i]) ;
///   }
/// }
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   e_new[i] = e_new[i] + 0.5 * delvc[i]
///              * ( 3.0*(p_old[i] + q_old[i])
///                  - 4.0*(pHalfStep[i] + q_new[i])) ;
/// }
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   e_new[i] += 0.5 * work[i];
///   if ( fabs(e_new[i]) < e_cut ) { e_new[i] = 0.0  ; }
///   if ( e_new[i]  < emin ) { e_new[i] = emin ; }
/// }
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   Real_type q_tilde ;
///   if (delvc[i] > 0.0) {
///      q_tilde = 0. ;
///   }
///   else {
///      Real_type ssc = ( pbvc[i] * e_new[i]
///          + vnewc[i] * vnewc[i] * bvc[i] * p_new[i] ) / rho0 ;
///      if ( ssc <= 0.1111111e-36 ) {
///         ssc = 0.3333333e-18 ;
///      } else {
///         ssc = sqrt(ssc) ;
///      }
///      q_tilde = (ssc*ql_old[i] + qq_old[i]) ;
///   }
///   e_new[i] = e_new[i] - ( 7.0*(p_old[i] + q_old[i])
///                          - 8.0*(pHalfStep[i] + q_new[i])
///                          + (p_new[i] + q_tilde)) * delvc[i] / 6.0 ;
///   if ( fabs(e_new[i]) < e_cut ) {
///      e_new[i] = 0.0  ;
///   }
///   if ( e_new[i]  < emin ) {
///      e_new[i] = emin ;
///   }
/// }
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   if ( delvc[i] <= 0.0 ) {
///      Real_type ssc = ( pbvc[i] * e_new[i]
///              + vnewc[i] * vnewc[i] * bvc[i] * p_new[i] ) / rho0 ;
///      if ( ssc <= 0.1111111e-36 ) {
///         ssc = 0.3333333e-18 ;
///      } else {
///         ssc = sqrt(ssc) ;
///      }
///      q_new[i] = (ssc*ql_old[i] + qq_old[i]) ;
///      if (fabs(q_new[i]) < q_cut) q_new[i] = 0.0 ;
///   }
/// }
///

#ifndef RAJAPerf_Apps_ENERGY_HPP
#define RAJAPerf_Apps_ENERGY_HPP

#define ENERGY_DATA_SETUP \
  Real_ptr e_new = m_e_new; \
  Real_ptr e_old = m_e_old; \
  Real_ptr delvc = m_delvc; \
  Real_ptr p_new = m_p_new; \
  Real_ptr p_old = m_p_old; \
  Real_ptr q_new = m_q_new; \
  Real_ptr q_old = m_q_old; \
  Real_ptr work = m_work; \
  Real_ptr compHalfStep = m_compHalfStep; \
  Real_ptr pHalfStep = m_pHalfStep; \
  Real_ptr bvc = m_bvc; \
  Real_ptr pbvc = m_pbvc; \
  Real_ptr ql_old = m_ql_old; \
  Real_ptr qq_old = m_qq_old; \
  Real_ptr vnewc = m_vnewc; \
  const Real_type rho0 = m_rho0; \
  const Real_type e_cut = m_e_cut; \
  const Real_type emin = m_emin; \
  const Real_type q_cut = m_q_cut;

#define ENERGY_BODY1 \
  e_new[i] = e_old[i] - 0.5 * delvc[i] * \
             (p_old[i] + q_old[i]) + 0.5 * work[i];

#define ENERGY_BODY2 \
  if ( delvc[i] > 0.0 ) { \
     q_new[i] = 0.0 ; \
  } \
  else { \
     Real_type vhalf = 1.0 / (1.0 + compHalfStep[i]) ; \
     Real_type ssc = ( pbvc[i] * e_new[i] \
        + vhalf * vhalf * bvc[i] * pHalfStep[i] ) / rho0 ; \
     if ( ssc <= 0.1111111e-36 ) { \
        ssc = 0.3333333e-18 ; \
     } else { \
        ssc = sqrt(ssc) ; \
     } \
     q_new[i] = (ssc*ql_old[i] + qq_old[i]) ; \
  }

#define ENERGY_BODY3 \
  e_new[i] = e_new[i] + 0.5 * delvc[i] \
             * ( 3.0*(p_old[i] + q_old[i]) \
                 - 4.0*(pHalfStep[i] + q_new[i])) ;

#define ENERGY_BODY4 \
  e_new[i] += 0.5 * work[i]; \
  if ( fabs(e_new[i]) < e_cut ) { e_new[i] = 0.0  ; } \
  if ( e_new[i]  < emin ) { e_new[i] = emin ; }

#define ENERGY_BODY5 \
  Real_type q_tilde ; \
  if (delvc[i] > 0.0) { \
     q_tilde = 0. ; \
  } \
  else { \
     Real_type ssc = ( pbvc[i] * e_new[i] \
         + vnewc[i] * vnewc[i] * bvc[i] * p_new[i] ) / rho0 ; \
     if ( ssc <= 0.1111111e-36 ) { \
        ssc = 0.3333333e-18 ; \
     } else { \
        ssc = sqrt(ssc) ; \
     } \
     q_tilde = (ssc*ql_old[i] + qq_old[i]) ; \
  } \
  e_new[i] = e_new[i] - ( 7.0*(p_old[i] + q_old[i]) \
                         - 8.0*(pHalfStep[i] + q_new[i]) \
                         + (p_new[i] + q_tilde)) * delvc[i] / 6.0 ; \
  if ( fabs(e_new[i]) < e_cut ) { \
     e_new[i] = 0.0  ; \
  } \
  if ( e_new[i]  < emin ) { \
     e_new[i] = emin ; \
  }

#define ENERGY_BODY6 \
  if ( delvc[i] <= 0.0 ) { \
     Real_type ssc = ( pbvc[i] * e_new[i] \
             + vnewc[i] * vnewc[i] * bvc[i] * p_new[i] ) / rho0 ; \
     if ( ssc <= 0.1111111e-36 ) { \
        ssc = 0.3333333e-18 ; \
     } else { \
        ssc = sqrt(ssc) ; \
     } \
     q_new[i] = (ssc*ql_old[i] + qq_old[i]) ; \
     if (fabs(q_new[i]) < q_cut) q_new[i] = 0.0 ; \
  }

// ENERGY_BODY1..6 for one i in a single pass (all six loops are element-wise,
// so fusion is exact); each input is read once, e_new/q_new written once.
#define ENERGY_FUSED_COMPUTE(en, qn) \
  Real_type en, qn; \
  { \
    const Real_type dv = delvc[i]; \
    const Real_type pq_old = p_old[i] + q_old[i]; \
    const Real_type phs = pHalfStep[i]; \
    const Real_type hwork = 0.5 * work[i]; \
    const Real_type pbv = pbvc[i]; \
    const Real_type qlo = ql_old[i]; \
    const Real_type qqo = qq_old[i]; \
    const Real_type vnb = vnewc[i] * vnewc[i] * bvc[i] * p_new[i]; \
    en = e_old[i] - 0.5 * dv * pq_old + hwork; \
    if ( dv > 0.0 ) { \
      qn = 0.0; \
    } else { \
      const Real_type vhalf = 1.0 / (1.0 + compHalfStep[i]); \
      Real_type ssc = ( pbv * en + vhalf * vhalf * bvc[i] * phs ) / rho0; \
      ssc = ( ssc <= 0.1111111e-36 ) ? 0.3333333e-18 : sqrt(ssc); \
      qn = ssc * qlo + qqo; \
    } \
    en = en + 0.5 * dv * ( 3.0 * pq_old - 4.0 * (phs + qn) ); \
    en += hwork; \
    if ( fabs(en) < e_cut ) { en = 0.0; } \
    if ( en < emin ) { en = emin; } \
    Real_type q_tilde; \
    if ( dv > 0.0 ) { \
      q_tilde = 0.0; \
    } else { \
      Real_type ssc = ( pbv * en + vnb ) / rho0; \
      ssc = ( ssc <= 0.1111111e-36 ) ? 0.3333333e-18 : sqrt(ssc); \
      q_tilde = ssc * qlo + qqo; \
    } \
    en = en - ( 7.0 * pq_old - 8.0 * (phs + qn) + (p_new[i] + q_tilde) ) * dv / 6.0; \
    if ( fabs(en) < e_cut ) { en = 0.0; } \
    if ( en < emin ) { en = emin; } \
    if ( dv <= 0.0 ) { \
      Real_type ssc = ( pbv * en + vnb ) / rho0; \
      ssc = ( ssc <= 0.1111111e-36 ) ? 0.3333333e-18 : sqrt(ssc); \
      qn = ssc * qlo + qqo; \
      if ( fabs(qn) < q_cut ) { qn = 0.0; } \
    } \
  }

#include "FEM_SIMD.hpp"

constexpr rajaperf::Index_type ENERGY_TILE = 256;

// Tile [t, t+ENERGY_TILE) of the fused kernel, results staged in L1 so full
// aligned tiles can be written with streaming stores when STREAM is true.
#define ENERGY_FUSED_TILE(t, STREAM) \
  { \
    const Index_type tend_ = ((t) + ENERGY_TILE < iend) ? (t) + ENERGY_TILE : iend; \
    alignas(64) Real_type en_t[ENERGY_TILE]; \
    alignas(64) Real_type qn_t[ENERGY_TILE]; \
    FEM_OMP_SIMD \
    for (Index_type i = (t); i < tend_; ++i) { \
      ENERGY_FUSED_COMPUTE(en, qn); \
      en_t[i - (t)] = en; \
      qn_t[i - (t)] = qn; \
    } \
    if ((STREAM) && nt_ok && tend_ - (t) == ENERGY_TILE) { \
      fem_simd::stream_copy(e_new + (t), en_t, ENERGY_TILE); \
      fem_simd::stream_copy(q_new + (t), qn_t, ENERGY_TILE); \
    } else { \
      for (Index_type i = (t); i < tend_; ++i) { \
        e_new[i] = en_t[i - (t)]; \
        q_new[i] = qn_t[i - (t)]; \
      } \
    } \
  }

namespace rajaperf
{
namespace apps
{

struct EnergyArgs
{
  Real_ptr e_new, e_old, delvc, p_new, p_old, q_new, q_old, work;
  Real_ptr compHalfStep, pHalfStep, bvc, pbvc, ql_old, qq_old, vnewc;
  Real_type rho0, e_cut, emin, q_cut;
};

#define ENERGY_ARGS_SETUP \
  const EnergyArgs energy_args{e_new, e_old, delvc, p_new, p_old, q_new, \
      q_old, work, compHalfStep, pHalfStep, bvc, pbvc, ql_old, qq_old, vnewc, \
      rho0, e_cut, emin, q_cut};

//
// Fused ENERGY with the delvc <= 0 work skipped where no element needs it.
// That branch reads 6 more arrays (compHalfStep, bvc, pbvc, ql_old, qq_old,
// vnewc) and needs a division and two square roots; compiler if-conversion
// executes it for every element. Here it runs only for vectors (AVX-512) or
// tiles (portable path) that contain such an element, so the common case
// reads 7 arrays instead of 13.
//
#if defined(RAJAPERF_FEM_AVX512)

// ssc = (pbvc*e + v*v*bvc*p) / rho0; ssc <= 0.1111111e-36 ? 0.3333333e-18 : sqrt(ssc)
RAJA_INLINE __m512d energy_ssc_avx512(__m512d pbv, __m512d e, __m512d v,
                                      __m512d bv, __m512d p, Real_type rho0)
{
  const __m512d t = _mm512_mul_pd(_mm512_mul_pd(v, v), _mm512_mul_pd(bv, p));
  const __m512d s = _mm512_div_pd(_mm512_fmadd_pd(pbv, e, t), _mm512_set1_pd(rho0));
  const __mmask8 m = _mm512_cmp_pd_mask(s, _mm512_set1_pd(0.1111111e-36), _CMP_NLE_UQ);
  return _mm512_mask_sqrt_pd(_mm512_set1_pd(0.3333333e-18), m, s);
}

// |e| < e_cut -> 0, then e < emin -> emin (scalar order)
RAJA_INLINE __m512d energy_eclamp_avx512(__m512d e, Real_type e_cut, Real_type emin)
{
  const __mmask8 mc = _mm512_cmp_pd_mask(_mm512_abs_pd(e), _mm512_set1_pd(e_cut), _CMP_LT_OQ);
  e = _mm512_mask_mov_pd(e, mc, _mm512_setzero_pd());
  const __m512d vmin = _mm512_set1_pd(emin);
  return _mm512_mask_mov_pd(e, _mm512_cmp_pd_mask(e, vmin, _CMP_LT_OQ), vmin);
}

// ENERGY_BODY1..6 for the lanes in mv of [i, i+8). FULL: mv is all lanes.
// STREAM: e_new + i and q_new + i are 64B aligned and FULL.
template < bool FULL, bool STREAM >
RAJA_INLINE void energy_vec8_avx512(const EnergyArgs& a, Index_type i, __mmask8 mv)
{
  auto ld = [&](const Real_type* p) {
    return FULL ? _mm512_loadu_pd(p + i) : _mm512_maskz_loadu_pd(mv, p + i);
  };
  const __m512d zero = _mm512_setzero_pd();
  const __m512d half = _mm512_set1_pd(0.5);

  const __m512d dv = ld(a.delvc);
  const __m512d pq = _mm512_add_pd(ld(a.p_old), ld(a.q_old));
  const __m512d phs = ld(a.pHalfStep);
  const __m512d hwork = _mm512_mul_pd(half, ld(a.work));
  const __m512d pnew = ld(a.p_new);
  const __m512d hdv = _mm512_mul_pd(half, dv);

  // lanes taking the delvc <= 0 branches (bodies 2 and 5: !(dv > 0))
  const __mmask8 mq = _mm512_cmp_pd_mask(dv, zero, _CMP_NGT_UQ) & mv;

  // body 1
  __m512d en = _mm512_add_pd(_mm512_fnmadd_pd(hdv, pq, ld(a.e_old)), hwork);

  __m512d qn = zero;
  __m512d pbv = zero, bv = zero, vn = zero, qlo = zero, qqo = zero;
  if (mq) {
    pbv = _mm512_maskz_loadu_pd(mq, a.pbvc + i);
    bv = _mm512_maskz_loadu_pd(mq, a.bvc + i);
    vn = _mm512_maskz_loadu_pd(mq, a.vnewc + i);
    qlo = _mm512_maskz_loadu_pd(mq, a.ql_old + i);
    qqo = _mm512_maskz_loadu_pd(mq, a.qq_old + i);
    // body 2
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d vhalf = _mm512_div_pd(one,
        _mm512_add_pd(one, _mm512_maskz_loadu_pd(mq, a.compHalfStep + i)));
    const __m512d ssc = energy_ssc_avx512(pbv, en, vhalf, bv, phs, a.rho0);
    qn = _mm512_maskz_fmadd_pd(mq, ssc, qlo, qqo);
  }

  // bodies 3, 4
  const __m512d hq = _mm512_add_pd(phs, qn);
  en = _mm512_fmadd_pd(hdv,
         _mm512_fmsub_pd(_mm512_set1_pd(3.0), pq, _mm512_mul_pd(_mm512_set1_pd(4.0), hq)),
         en);
  en = energy_eclamp_avx512(_mm512_add_pd(en, hwork), a.e_cut, a.emin);

  // body 5
  __m512d qt = zero;
  if (mq) {
    const __m512d ssc = energy_ssc_avx512(pbv, en, vn, bv, pnew, a.rho0);
    qt = _mm512_maskz_fmadd_pd(mq, ssc, qlo, qqo);
  }
  __m512d t = _mm512_fmsub_pd(_mm512_set1_pd(7.0), pq, _mm512_mul_pd(_mm512_set1_pd(8.0), hq));
  t = _mm512_add_pd(t, _mm512_add_pd(pnew, qt));
  en = _mm512_sub_pd(en, _mm512_div_pd(_mm512_mul_pd(t, dv), _mm512_set1_pd(6.0)));
  en = energy_eclamp_avx512(en, a.e_cut, a.emin);

  // body 6 (dv <= 0)
  const __mmask8 m6 = _mm512_cmp_pd_mask(dv, zero, _CMP_LE_OQ) & mq;
  if (m6) {
    const __m512d ssc = energy_ssc_avx512(pbv, en, vn, bv, pnew, a.rho0);
    __m512d q6 = _mm512_fmadd_pd(ssc, qlo, qqo);
    const __mmask8 mc = _mm512_cmp_pd_mask(_mm512_abs_pd(q6), _mm512_set1_pd(a.q_cut), _CMP_LT_OQ);
    q6 = _mm512_mask_mov_pd(q6, mc, zero);
    qn = _mm512_mask_mov_pd(qn, m6, q6);
  }

  if constexpr (STREAM) {
    _mm512_stream_pd(a.e_new + i, en);
    _mm512_stream_pd(a.q_new + i, qn);
  } else if constexpr (FULL) {
    _mm512_storeu_pd(a.e_new + i, en);
    _mm512_storeu_pd(a.q_new + i, qn);
  } else {
    _mm512_mask_storeu_pd(a.e_new + i, mv, en);
    _mm512_mask_storeu_pd(a.q_new + i, mv, qn);
  }
}

#endif

// Fused, branch-skipping ENERGY over [t, tend). STREAM (AVX-512 only) writes
// full vectors with non-temporal stores when e_new and q_new are aligned.
template < bool STREAM >
RAJA_INLINE void energy_fused_skip_tile(const EnergyArgs& a, Index_type t, Index_type tend)
{
#if defined(RAJAPERF_FEM_AVX512)
  const bool nt = STREAM && fem_simd::is_aligned64(a.e_new + t) &&
                  fem_simd::is_aligned64(a.q_new + t);
  Index_type i = t;
  if (nt) {
    for (; i + 8 <= tend; i += 8) {
      energy_vec8_avx512<true, true>(a, i, 0xFF);
    }
  } else {
    for (; i + 8 <= tend; i += 8) {
      energy_vec8_avx512<true, false>(a, i, 0xFF);
    }
  }
  if (i < tend) {
    energy_vec8_avx512<false, false>(a, i, fem_simd::lane_mask(tend - i));
  }
#else
  static_assert(!STREAM, "streaming stores need the AVX-512 path");
  Real_ptr e_new = a.e_new; Real_ptr e_old = a.e_old; Real_ptr delvc = a.delvc;
  Real_ptr p_new = a.p_new; Real_ptr p_old = a.p_old; Real_ptr q_new = a.q_new;
  Real_ptr q_old = a.q_old; Real_ptr work = a.work;
  Real_ptr compHalfStep = a.compHalfStep; Real_ptr pHalfStep = a.pHalfStep;
  Real_ptr bvc = a.bvc; Real_ptr pbvc = a.pbvc; Real_ptr ql_old = a.ql_old;
  Real_ptr qq_old = a.qq_old; Real_ptr vnewc = a.vnewc;
  const Real_type rho0 = a.rho0; const Real_type e_cut = a.e_cut;
  const Real_type emin = a.emin; const Real_type q_cut = a.q_cut;
  int any_le = 0;
  for (Index_type i = t; i < tend; ++i) {
    any_le |= !(delvc[i] > 0.0);
  }
  if (any_le) {
    FEM_OMP_SIMD
    for (Index_type i = t; i < tend; ++i) {
      ENERGY_FUSED_COMPUTE(en, qn);
      e_new[i] = en;
      q_new[i] = qn;
    }
  } else {
    // ENERGY_FUSED_COMPUTE with delvc > 0: q_new = q_tilde = 0
    FEM_OMP_SIMD
    for (Index_type i = t; i < tend; ++i) {
      const Real_type dv = delvc[i];
      const Real_type pq_old = p_old[i] + q_old[i];
      const Real_type phs = pHalfStep[i];
      const Real_type hwork = 0.5 * work[i];
      Real_type en = e_old[i] - 0.5 * dv * pq_old + hwork;
      en = en + 0.5 * dv * ( 3.0 * pq_old - 4.0 * phs );
      en += hwork;
      if ( fabs(en) < e_cut ) { en = 0.0; }
      if ( en < emin ) { en = emin; }
      en = en - ( 7.0 * pq_old - 8.0 * phs + p_new[i] ) * dv / 6.0;
      if ( fabs(en) < e_cut ) { en = 0.0; }
      if ( en < emin ) { en = emin; }
      e_new[i] = en;
      q_new[i] = 0.0;
    }
  }
  RAJA_UNUSED_VAR(compHalfStep); RAJA_UNUSED_VAR(bvc); RAJA_UNUSED_VAR(pbvc);
  RAJA_UNUSED_VAR(ql_old); RAJA_UNUSED_VAR(qq_old); RAJA_UNUSED_VAR(vnewc);
  RAJA_UNUSED_VAR(rho0); RAJA_UNUSED_VAR(q_cut);
#endif
}

} // end namespace apps
} // end namespace rajaperf

#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace apps
{

class ENERGY : public KernelBase
{
public:

  ENERGY(const RunParams& params);

  ~ENERGY();

  void setSize(Index_type target_size, Index_type target_reps);
  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void defineSeqVariantTunings();
  void defineOpenMPVariantTunings();
  void defineOpenMPTargetVariantTunings();
  void defineCudaVariantTunings();
  void defineHipVariantTunings();
  void defineSyclVariantTunings();

  void runSeqVariant(VariantID vid);
  template < size_t tune_idx >
  void runOpenMPVariant(VariantID vid);
  void runOpenMPVariantReference(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);
  template < size_t work_group_size >
  void runSyclVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size>;

  Real_ptr m_e_new;
  Real_ptr m_e_old;
  Real_ptr m_delvc;
  Real_ptr m_p_new;
  Real_ptr m_p_old;
  Real_ptr m_q_new;
  Real_ptr m_q_old;
  Real_ptr m_work;
  Real_ptr m_compHalfStep;
  Real_ptr m_pHalfStep;
  Real_ptr m_bvc;
  Real_ptr m_pbvc;
  Real_ptr m_ql_old;
  Real_ptr m_qq_old;
  Real_ptr m_vnewc;

  Real_type m_rho0;
  Real_type m_e_cut;
  Real_type m_emin;
  Real_type m_q_cut;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
