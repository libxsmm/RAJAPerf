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
