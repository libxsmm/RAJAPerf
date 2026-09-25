//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// NODAL_ACCUMULATION_3D kernel reference implementation:
///
/// NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;
///
/// for (Index_type ii = ibegin; ii < iend; ++ii ) {
///   Index_type i = real_zones[ii];
///
///   Real_type val = 0.125 * vol[i] ;
///
///   x0[i] += val;
///   x1[i] += val;
///   x2[i] += val;
///   x3[i] += val;
///   x4[i] += val;
///   x5[i] += val;
///   x6[i] += val;
///   x7[i] += val;
///
/// }
///

#ifndef RAJAPerf_Apps_NODAL_ACCUMULATION_3D_HPP
#define RAJAPerf_Apps_NODAL_ACCUMULATION_3D_HPP

#include "FEM_MACROS.hpp"

#define NODAL_ACCUMULATION_3D_DATA_SETUP \
  Real_ptr x = m_x; \
  Real_ptr vol = m_vol; \
  \
  Real_ptr x0,x1,x2,x3,x4,x5,x6,x7; \
  \
  NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ; \
  \
  Index_ptr real_zones = m_real_zones;

#define NODAL_ACCUMULATION_3D_BODY_INDEX \
  Index_type i = real_zones[ii];

#define NODAL_ACCUMULATION_3D_BODY(atomicAdd) \
  Real_type val = 0.125 * vol[i]; \
  \
  atomicAdd(x0[i], val); \
  atomicAdd(x1[i], val); \
  atomicAdd(x2[i], val); \
  atomicAdd(x3[i], val); \
  atomicAdd(x4[i], val); \
  atomicAdd(x5[i], val); \
  atomicAdd(x6[i], val); \
  atomicAdd(x7[i], val);

//
// Node-centric (gather) form used by the CPU OpenMP variants: each real node
// pulls 0.125*vol from its (up to 8) adjacent real zones, so no atomics are
// needed and the inner i-loop is unit stride.
//
#define NODAL_ACCUMULATION_3D_GATHER_SETUP \
  const Index_type imin = m_domain->imin; \
  const Index_type imax = m_domain->imax; \
  const Index_type jmin = m_domain->jmin; \
  const Index_type jmax = m_domain->jmax; \
  const Index_type kmin = m_domain->kmin; \
  const Index_type kmax = m_domain->kmax; \
  const Index_type jp = m_domain->jp; \
  const Index_type kp = m_domain->kp; \
  const Index_type nj = jmax - jmin + 1; \
  const Index_type nrows = nj * (kmax - kmin + 1);

// One row of real nodes (fixed j,k); zone (i,j,k) has the same index as its
// lowest node, so the 8 zones around node n are n - {0,1} - {0,jp} - {0,kp}.
#define NODAL_ACCUMULATION_3D_GATHER_ROW(r) \
  const Index_type j = jmin + (r) % nj; \
  const Index_type k = kmin + (r) / nj; \
  const bool jlo = (j > jmin), jhi = (j < jmax); \
  const bool klo = (k > kmin), khi = (k < kmax); \
  Real_ptr xr = x + k*kp + j*jp; \
  Real_ptr v00 = vol + k*kp + j*jp; \
  Real_ptr v10 = v00 - jp; \
  Real_ptr v01 = v00 - kp; \
  Real_ptr v11 = v00 - jp - kp; \
  FEM_OMP_SIMD \
  for (Index_type i = imin; i <= imax; ++i) { \
    const bool ilo = (i > imin), ihi = (i < imax); \
    Real_type sum = 0.0; \
    sum += (ihi && jhi && khi) ? 0.125 * v00[i]   : 0.0; \
    sum += (ilo && jhi && khi) ? 0.125 * v00[i-1] : 0.0; \
    sum += (ihi && jlo && khi) ? 0.125 * v10[i]   : 0.0; \
    sum += (ilo && jlo && khi) ? 0.125 * v10[i-1] : 0.0; \
    sum += (ihi && jhi && klo) ? 0.125 * v01[i]   : 0.0; \
    sum += (ilo && jhi && klo) ? 0.125 * v01[i-1] : 0.0; \
    sum += (ihi && jlo && klo) ? 0.125 * v11[i]   : 0.0; \
    sum += (ilo && jlo && klo) ? 0.125 * v11[i-1] : 0.0; \
    xr[i] += sum; \
  }



#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace apps
{
class ADomain;

class NODAL_ACCUMULATION_3D : public KernelBase
{
public:

  NODAL_ACCUMULATION_3D(const RunParams& params);

  ~NODAL_ACCUMULATION_3D();

  void setSize(Index_type target_size, Index_type target_reps);
  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void defineSeqVariantTunings();
  void defineOpenMPVariantTunings();
  void defineOpenMPTargetVariantTunings();
  void defineCudaVariantTunings();
  void defineHipVariantTunings();
  void defineKokkosVariantTunings();

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);
  void runKokkosVariant(VariantID vid);

  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size>;

  Real_ptr m_x;
  Real_ptr m_vol;

  std::unique_ptr<ADomain> m_domain;
  Index_type* m_real_zones;
  Index_type m_nodal_array_length;
  Index_type m_zonal_array_length;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
