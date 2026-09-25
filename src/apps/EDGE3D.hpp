//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// EDGE3D kernel reference implementation:
///
/// NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;
/// NDPTRSET(m_domain->jp, m_domain->kp, y,y0,y1,y2,y3,y4,y5,y6,y7) ;
/// NDPTRSET(m_domain->jp, m_domain->kp, z,z0,z1,z2,z3,z4,z5,z6,z7) ;
///
/// for (Index_type i = ibegin ; i < iend ; ++i ) {
///
///   double x[NB] = {x0[i],x1[i],x2[i],x3[i],x4[i],x5[i],x6[i],x7[i]};
///   double y[NB] = {y0[i],y1[i],y2[i],y3[i],y4[i],y5[i],y6[i],y7[i]};
///   double z[NB] = {z0[i],z1[i],z2[i],z3[i],z4[i],z5[i],z6[i],z7[i]};
///   double edge_matrix[EB][EB];
///
  // Get integration points and weights
///   double qpts_1d[MAX_QUAD_ORDER];
///   double wgts_1d[MAX_QUAD_ORDER];
///
///   get_quadrature_rule(quad_type, quad_order, qpts_1d, wgts_1d);
///
  // Compute cell centered Jacobian
///   const double jxx_cc = Jxx(x, y, z, 0.25, 0.25, 0.25, 0.25);
///   const double jxy_cc = Jxy(x, y, z, 0.25, 0.25, 0.25, 0.25);
///   const double jxz_cc = Jxz(x, y, z, 0.25, 0.25, 0.25, 0.25);
///   const double jyx_cc = Jyx(x, y, z, 0.25, 0.25, 0.25, 0.25);
///   const double jyy_cc = Jyy(x, y, z, 0.25, 0.25, 0.25, 0.25);
///   const double jyz_cc = Jyz(x, y, z, 0.25, 0.25, 0.25, 0.25);
///   const double jzx_cc = Jzx(x, y, z, 0.25, 0.25, 0.25, 0.25);
///   const double jzy_cc = Jzy(x, y, z, 0.25, 0.25, 0.25, 0.25);
///   const double jzz_cc = Jzz(x, y, z, 0.25, 0.25, 0.25, 0.25);
///
  // Compute cell centered Jacobian determinant
///   const double detj_cc = compute_detj(
///     jxx_cc, jxy_cc, jxz_cc,
///     jyx_cc, jyy_cc, jyz_cc,
///     jzx_cc, jzy_cc, jzz_cc);
///
  // Initialize the stiffness matrix
///   for (int m = 0; m < EB; m++) {
///     for (int p = m; p < EB; p++) {
///       matrix[m][p] = 0.0;
///     }
///   }
///
  // Compute values at each quadrature point
///   for ( int i = 0; i < quad_order; i++ ) {
///
///     const double xloc = qpts_1d[i];
///     const double tmpx = 1. - xloc;
///
///     double dbasisx[EB] = {0};
///     curl_edgebasis_x(dbasisx, tmpx, xloc);
///
///     for ( int j = 0; j < quad_order; j++ ) {
///
///       const double yloc = qpts_1d[j];
///       const double wgtxy = wgts_1d[i]*wgts_1d[j];
///       const double tmpy = 1. - yloc;
///
///       double tmpxy    = tmpx*tmpy;
///       double xyloc    = xloc*yloc;
///       double tmpxyloc = tmpx*yloc;
///       double xloctmpy = xloc*tmpy;
///
///       const double jzx = Jzx(x, y, z, tmpxy, xloctmpy, xyloc, tmpxyloc);
///       const double jzy = Jzy(x, y, z, tmpxy, xloctmpy, xyloc, tmpxyloc);
///       const double jzz = Jzz(x, y, z, tmpxy, xloctmpy, xyloc, tmpxyloc);
///
///       double ebasisz[EB] = {0};
///       edgebasis_z(ebasisz, tmpxy, xloctmpy, xyloc, tmpxyloc);
///
///       double dbasisy[EB] = {0};
///       curl_edgebasis_y(dbasisy, tmpy, yloc);
///
      // Differeniate basis with respect to z at this quadrature point
///
///       for ( int k = 0; k < quad_order; k++ ) {
///
///         const double zloc = qpts_1d[k];
///         const double wgts = wgtxy*wgts_1d[k];
///         const double tmpz = 1. - zloc;
///
///         const double tmpxz    = tmpx*tmpz;
///         const double tmpyz    = tmpy*tmpz;
///
///         const double xzloc    = xloc*zloc;
///         const double yzloc    = yloc*zloc;
///
///         const double tmpyzloc = tmpy*zloc;
///         const double tmpxzloc = tmpx*zloc;
///
///         const double yloctmpz = yloc*tmpz;
///         const double xloctmpz = xloc*tmpz;
///
///         const double jxx = Jxx(x, y, z, tmpyz, yloctmpz, tmpyzloc, yzloc);
///         const double jxy = Jxy(x, y, z, tmpyz, yloctmpz, tmpyzloc, yzloc);
///         const double jxz = Jxz(x, y, z, tmpyz, yloctmpz, tmpyzloc, yzloc);
///         const double jyx = Jyx(x, y, z, tmpxz, xloctmpz, tmpxzloc, xzloc);
///         const double jyy = Jyy(x, y, z, tmpxz, xloctmpz, tmpxzloc, xzloc);
///         const double jyz = Jyz(x, y, z, tmpxz, xloctmpz, tmpxzloc, xzloc);
///
///         double jinvxx, jinvxy, jinvxz,
///                jinvyx, jinvyy, jinvyz,
///                jinvzx, jinvzy, jinvzz,
///                detj_unfixed, detj, abs_detj, invdetj;
///
///         jacobian_inv(
///           jxx, jxy, jxz,
///           jyx, jyy, jyz,
///           jzx, jzy, jzz,
///           detj_cc, detj_tol,
///           jinvxx, jinvxy, jinvxz,
///           jinvyx, jinvyy, jinvyz,
///           jinvzx, jinvzy, jinvzz,
///           detj_unfixed, detj, abs_detj, invdetj);
///
///         const double detjwgts = wgts*abs_detj;
///
///         double ebasisx[EB] = {0};
///         edgebasis_x(ebasisx, tmpyz, yloctmpz, tmpyzloc, yzloc);
///
///         double ebasisy[EB] = {0};
///         edgebasis_y(ebasisy, tmpxz, xloctmpz, tmpxzloc, xzloc);
///
///         double dbasisz[EB] = {0};
///         curl_edgebasis_z(dbasisz, tmpz, zloc);
///
///         const double inv_abs_detj = 1./(abs_detj+ptiny);
///
///         double tebasisx[EB] = {0};
///         double tebasisy[EB] = {0};
///         double tebasisz[EB] = {0};
///
///         transform_edge_basis(
///           jinvxx, jinvxy, jinvxz,
///           jinvyx, jinvyy, jinvyz,
///           jinvzx, jinvzy, jinvzz,
///           ebasisx, ebasisy, ebasisz,
///           tebasisx, tebasisy, tebasisz);
///
///         double tdbasisx[EB] = {0};
///         double tdbasisy[EB] = {0};
///         double tdbasisz[EB] = {0};
///
///         transform_curl_edge_basis(
///           jxx, jxy, jxz,
///           jyx, jyy, jyz,
///           jzx, jzy, jzz,
///           inv_abs_detj,
///           dbasisx, dbasisy, dbasisz,
///           tdbasisx, tdbasisy, tdbasisz);
///
        // the inner product: alpha*<w_i, w_j>
///         inner_product(
///           detjwgts*alpha,
///           tebasisx, tebasisy, tebasisz,
///           tebasisx, tebasisy, tebasisz,
///           matrix, true);
///
         // the inner product: beta*<Curl(w_i), Curl(w_j)>
///         inner_product(
///           detjwgts*beta,
///           tdbasisx, tdbasisy, tdbasisz,
///           tdbasisx, tdbasisy, tdbasisz,
///           matrix, true);
///
///       }
///     }
///   }
///   sum[i] = 0.0;
///   for (int m = 0; m < EB; m++) {
///     Real_type check = 0.0;
///     for (int p = 0; p < EB; p++) {
///       check += edge_matrix[m*EB + p];
///     }
///     sum[i] += check;
///   }
/// }

#ifndef RAJAPerf_Apps_EDGE3D_HPP
#define RAJAPerf_Apps_EDGE3D_HPP

#define NQ_1D 2

#include "mixed_fem_helper.hpp"

RAJA_HOST_DEVICE
RAJA_INLINE void edge_MpSmatrix(
  const rajaperf::Real_type  (&x)[NB],
  const rajaperf::Real_type  (&y)[NB],
  const rajaperf::Real_type  (&z)[NB],
  rajaperf::Real_type        alpha,
  rajaperf::Real_type        beta,
  const rajaperf::Real_type  detj_tol,
  const rajaperf::Int_type   quad_type,
  const rajaperf::Int_type   quad_order,
  rajaperf::Real_type        (&matrix)[EB][EB])
{
  // Get integration points and weights
  rajaperf::Real_type qpts_1d[MAX_QUAD_ORDER];
  rajaperf::Real_type wgts_1d[MAX_QUAD_ORDER];

  get_quadrature_rule(quad_type, quad_order, qpts_1d, wgts_1d);

  // Compute cell centered Jacobian
  const rajaperf::Real_type jxx_cc = Jxx(x, y, z, 0.25, 0.25, 0.25, 0.25);
  const rajaperf::Real_type jxy_cc = Jxy(x, y, z, 0.25, 0.25, 0.25, 0.25);
  const rajaperf::Real_type jxz_cc = Jxz(x, y, z, 0.25, 0.25, 0.25, 0.25);
  const rajaperf::Real_type jyx_cc = Jyx(x, y, z, 0.25, 0.25, 0.25, 0.25);
  const rajaperf::Real_type jyy_cc = Jyy(x, y, z, 0.25, 0.25, 0.25, 0.25);
  const rajaperf::Real_type jyz_cc = Jyz(x, y, z, 0.25, 0.25, 0.25, 0.25);
  const rajaperf::Real_type jzx_cc = Jzx(x, y, z, 0.25, 0.25, 0.25, 0.25);
  const rajaperf::Real_type jzy_cc = Jzy(x, y, z, 0.25, 0.25, 0.25, 0.25);
  const rajaperf::Real_type jzz_cc = Jzz(x, y, z, 0.25, 0.25, 0.25, 0.25);

  // Compute cell centered Jacobian determinant
  const rajaperf::Real_type detj_cc = compute_detj(
    jxx_cc, jxy_cc, jxz_cc,
    jyx_cc, jyy_cc, jyz_cc,
    jzx_cc, jzy_cc, jzz_cc);

  // Initialize the stiffness matrix
  for (rajaperf::Int_type m = 0; m < EB; m++) {
    for (rajaperf::Int_type p = m; p < EB; p++) {
      matrix[m][p] = 0.0;
    }
  }

  // Compute values at each quadrature point
  for ( rajaperf::Int_type i = 0; i < quad_order; i++ ) {

    const rajaperf::Real_type xloc = qpts_1d[i];
    const rajaperf::Real_type tmpx = 1. - xloc;

    rajaperf::Real_type dbasisx[EB] = {0};
    curl_edgebasis_x(dbasisx, tmpx, xloc);

    for ( rajaperf::Int_type j = 0; j < quad_order; j++ ) {

      const rajaperf::Real_type yloc = qpts_1d[j];
      const rajaperf::Real_type wgtxy = wgts_1d[i]*wgts_1d[j];
      const rajaperf::Real_type tmpy = 1. - yloc;

      rajaperf::Real_type tmpxy    = tmpx*tmpy;
      rajaperf::Real_type xyloc    = xloc*yloc;
      rajaperf::Real_type tmpxyloc = tmpx*yloc;
      rajaperf::Real_type xloctmpy = xloc*tmpy;

      const rajaperf::Real_type jzx = Jzx(x, y, z, tmpxy, xloctmpy, xyloc, tmpxyloc);
      const rajaperf::Real_type jzy = Jzy(x, y, z, tmpxy, xloctmpy, xyloc, tmpxyloc);
      const rajaperf::Real_type jzz = Jzz(x, y, z, tmpxy, xloctmpy, xyloc, tmpxyloc);

      rajaperf::Real_type ebasisz[EB] = {0};
      edgebasis_z(ebasisz, tmpxy, xloctmpy, xyloc, tmpxyloc);

      rajaperf::Real_type dbasisy[EB] = {0};
      curl_edgebasis_y(dbasisy, tmpy, yloc);

      // Differeniate basis with respect to z at this quadrature point

      for ( rajaperf::Int_type k = 0; k < quad_order; k++ ) {

        const rajaperf::Real_type zloc = qpts_1d[k];
        const rajaperf::Real_type wgts = wgtxy*wgts_1d[k];
        const rajaperf::Real_type tmpz = 1. - zloc;

        const rajaperf::Real_type tmpxz    = tmpx*tmpz;
        const rajaperf::Real_type tmpyz    = tmpy*tmpz;

        const rajaperf::Real_type xzloc    = xloc*zloc;
        const rajaperf::Real_type yzloc    = yloc*zloc;

        const rajaperf::Real_type tmpyzloc = tmpy*zloc;
        const rajaperf::Real_type tmpxzloc = tmpx*zloc;

        const rajaperf::Real_type yloctmpz = yloc*tmpz;
        const rajaperf::Real_type xloctmpz = xloc*tmpz;

        const rajaperf::Real_type jxx = Jxx(x, y, z, tmpyz, yloctmpz, tmpyzloc, yzloc);
        const rajaperf::Real_type jxy = Jxy(x, y, z, tmpyz, yloctmpz, tmpyzloc, yzloc);
        const rajaperf::Real_type jxz = Jxz(x, y, z, tmpyz, yloctmpz, tmpyzloc, yzloc);
        const rajaperf::Real_type jyx = Jyx(x, y, z, tmpxz, xloctmpz, tmpxzloc, xzloc);
        const rajaperf::Real_type jyy = Jyy(x, y, z, tmpxz, xloctmpz, tmpxzloc, xzloc);
        const rajaperf::Real_type jyz = Jyz(x, y, z, tmpxz, xloctmpz, tmpxzloc, xzloc);

        rajaperf::Real_type jinvxx, jinvxy, jinvxz,
                            jinvyx, jinvyy, jinvyz,
                            jinvzx, jinvzy, jinvzz,
                            detj_unfixed, detj, abs_detj, invdetj;

        jacobian_inv(
          jxx, jxy, jxz,
          jyx, jyy, jyz,
          jzx, jzy, jzz,
          detj_cc, detj_tol,
          jinvxx, jinvxy, jinvxz,
          jinvyx, jinvyy, jinvyz,
          jinvzx, jinvzy, jinvzz,
          detj_unfixed, detj, abs_detj, invdetj);

        const rajaperf::Real_type detjwgts = wgts*abs_detj;

        rajaperf::Real_type ebasisx[EB] = {0};
        edgebasis_x(ebasisx, tmpyz, yloctmpz, tmpyzloc, yzloc);

        rajaperf::Real_type ebasisy[EB] = {0};
        edgebasis_y(ebasisy, tmpxz, xloctmpz, tmpxzloc, xzloc);

        rajaperf::Real_type dbasisz[EB] = {0};
        curl_edgebasis_z(dbasisz, tmpz, zloc);

        const rajaperf::Real_type inv_abs_detj = 1./(abs_detj+ptiny);

        rajaperf::Real_type tebasisx[EB] = {0};
        rajaperf::Real_type tebasisy[EB] = {0};
        rajaperf::Real_type tebasisz[EB] = {0};

        transform_edge_basis(
          jinvxx, jinvxy, jinvxz,
          jinvyx, jinvyy, jinvyz,
          jinvzx, jinvzy, jinvzz,
          ebasisx, ebasisy, ebasisz,
          tebasisx, tebasisy, tebasisz);

        rajaperf::Real_type tdbasisx[EB] = {0};
        rajaperf::Real_type tdbasisy[EB] = {0};
        rajaperf::Real_type tdbasisz[EB] = {0};

        transform_curl_edge_basis(
          jxx, jxy, jxz,
          jyx, jyy, jyz,
          jzx, jzy, jzz,
          inv_abs_detj,
          dbasisx, dbasisy, dbasisz,
          tdbasisx, tdbasisy, tdbasisz);

        // the inner product: alpha*<w_i, w_j>
        inner_product(
          detjwgts*alpha,
          tebasisx, tebasisy, tebasisz,
          tebasisx, tebasisy, tebasisz,
          matrix, true);

         // the inner product: beta*<Curl(w_i), Curl(w_j)>
        inner_product(
          detjwgts*beta,
          tdbasisx, tdbasisy, tdbasisz,
          tdbasisx, tdbasisy, tdbasisz,
          matrix, true);
      }
    }
  }
}

//
// SoA (element-innermost) variant used by the OpenMP and OpenMP target
// variants: EDGE3D_VL elements are processed together, one per SIMD lane.
//
// One 512-bit vector: 8 lanes in double precision, 16 in single precision.
constexpr rajaperf::Int_type EDGE3D_VL = 64 / sizeof(rajaperf::Real_type);

#if defined(_OPENMP)
#define EDGE3D_OMP_SIMD _Pragma("omp simd")
#else
#define EDGE3D_OMP_SIMD
#endif

// SoA forms of Jx*, Jy*, Jz*; c is one coordinate (x, y or z) as [node][lane]
RAJA_INLINE rajaperf::Real_type soa_dxi(
  const rajaperf::Real_type (&c)[NB][EDGE3D_VL],
  const rajaperf::Int_type  e,
  const rajaperf::Real_type w0,
  const rajaperf::Real_type w1,
  const rajaperf::Real_type w2,
  const rajaperf::Real_type w3)
{
  return (c[1][e] - c[0][e])*w0 +
         (c[2][e] - c[3][e])*w1 +
         (c[5][e] - c[4][e])*w2 +
         (c[6][e] - c[7][e])*w3;
}

RAJA_INLINE rajaperf::Real_type soa_deta(
  const rajaperf::Real_type (&c)[NB][EDGE3D_VL],
  const rajaperf::Int_type  e,
  const rajaperf::Real_type w0,
  const rajaperf::Real_type w1,
  const rajaperf::Real_type w2,
  const rajaperf::Real_type w3)
{
  return (c[3][e] - c[0][e])*w0 +
         (c[2][e] - c[1][e])*w1 +
         (c[7][e] - c[4][e])*w2 +
         (c[6][e] - c[5][e])*w3;
}

RAJA_INLINE rajaperf::Real_type soa_dzeta(
  const rajaperf::Real_type (&c)[NB][EDGE3D_VL],
  const rajaperf::Int_type  e,
  const rajaperf::Real_type w0,
  const rajaperf::Real_type w1,
  const rajaperf::Real_type w2,
  const rajaperf::Real_type w3)
{
  return (c[4][e] - c[0][e])*w0 +
         (c[5][e] - c[1][e])*w1 +
         (c[6][e] - c[2][e])*w2 +
         (c[7][e] - c[3][e])*w3;
}

RAJA_INLINE void edge_MpSmatrix_simd(
  const rajaperf::Real_type  (&x)[NB][EDGE3D_VL],
  const rajaperf::Real_type  (&y)[NB][EDGE3D_VL],
  const rajaperf::Real_type  (&z)[NB][EDGE3D_VL],
  rajaperf::Real_type        alpha,
  rajaperf::Real_type        beta,
  const rajaperf::Real_type  detj_tol,
  const rajaperf::Int_type   quad_type,
  const rajaperf::Int_type   quad_order,
  rajaperf::Real_type        (&matrix)[EB][EB][EDGE3D_VL])
{
  using rajaperf::Real_type;
  using rajaperf::Int_type;
  constexpr Int_type VL = EDGE3D_VL;

  // Get integration points and weights (element independent)
  Real_type qpts_1d[MAX_QUAD_ORDER];
  Real_type wgts_1d[MAX_QUAD_ORDER];

  get_quadrature_rule(quad_type, quad_order, qpts_1d, wgts_1d);

  // Compute cell centered Jacobian determinant per element
  Real_type detj_cc[VL];

  EDGE3D_OMP_SIMD
  for (Int_type e = 0; e < VL; e++) {
    const Real_type jxx_cc = soa_dxi  (x, e, 0.25, 0.25, 0.25, 0.25);
    const Real_type jxy_cc = soa_dxi  (y, e, 0.25, 0.25, 0.25, 0.25);
    const Real_type jxz_cc = soa_dxi  (z, e, 0.25, 0.25, 0.25, 0.25);
    const Real_type jyx_cc = soa_deta (x, e, 0.25, 0.25, 0.25, 0.25);
    const Real_type jyy_cc = soa_deta (y, e, 0.25, 0.25, 0.25, 0.25);
    const Real_type jyz_cc = soa_deta (z, e, 0.25, 0.25, 0.25, 0.25);
    const Real_type jzx_cc = soa_dzeta(x, e, 0.25, 0.25, 0.25, 0.25);
    const Real_type jzy_cc = soa_dzeta(y, e, 0.25, 0.25, 0.25, 0.25);
    const Real_type jzz_cc = soa_dzeta(z, e, 0.25, 0.25, 0.25, 0.25);

    detj_cc[e] = compute_detj(
      jxx_cc, jxy_cc, jxz_cc,
      jyx_cc, jyy_cc, jyz_cc,
      jzx_cc, jzy_cc, jzz_cc);
  }

  // Initialize the stiffness matrix (upper triangle)
  for (Int_type m = 0; m < EB; m++) {
    for (Int_type p = m; p < EB; p++) {
      EDGE3D_OMP_SIMD
      for (Int_type e = 0; e < VL; e++) {
        matrix[m][p][e] = 0.0;
      }
    }
  }

  // Compute values at each quadrature point
  for ( Int_type i = 0; i < quad_order; i++ ) {

    const Real_type xloc = qpts_1d[i];
    const Real_type tmpx = 1. - xloc;

    Real_type dbasisx[EB] = {0};
    curl_edgebasis_x(dbasisx, tmpx, xloc);

    for ( Int_type j = 0; j < quad_order; j++ ) {

      const Real_type yloc = qpts_1d[j];
      const Real_type wgtxy = wgts_1d[i]*wgts_1d[j];
      const Real_type tmpy = 1. - yloc;

      const Real_type tmpxy    = tmpx*tmpy;
      const Real_type xyloc    = xloc*yloc;
      const Real_type tmpxyloc = tmpx*yloc;
      const Real_type xloctmpy = xloc*tmpy;

      Real_type jzx[VL], jzy[VL], jzz[VL];

      EDGE3D_OMP_SIMD
      for (Int_type e = 0; e < VL; e++) {
        jzx[e] = soa_dzeta(x, e, tmpxy, xloctmpy, xyloc, tmpxyloc);
        jzy[e] = soa_dzeta(y, e, tmpxy, xloctmpy, xyloc, tmpxyloc);
        jzz[e] = soa_dzeta(z, e, tmpxy, xloctmpy, xyloc, tmpxyloc);
      }

      Real_type ebasisz[EB] = {0};
      edgebasis_z(ebasisz, tmpxy, xloctmpy, xyloc, tmpxyloc);

      Real_type dbasisy[EB] = {0};
      curl_edgebasis_y(dbasisy, tmpy, yloc);

      for ( Int_type k = 0; k < quad_order; k++ ) {

        const Real_type zloc = qpts_1d[k];
        const Real_type wgts = wgtxy*wgts_1d[k];
        const Real_type tmpz = 1. - zloc;

        const Real_type tmpxz    = tmpx*tmpz;
        const Real_type tmpyz    = tmpy*tmpz;

        const Real_type xzloc    = xloc*zloc;
        const Real_type yzloc    = yloc*zloc;

        const Real_type tmpyzloc = tmpy*zloc;
        const Real_type tmpxzloc = tmpx*zloc;

        const Real_type yloctmpz = yloc*tmpz;
        const Real_type xloctmpz = xloc*tmpz;

        Real_type ebasisx[EB] = {0};
        edgebasis_x(ebasisx, tmpyz, yloctmpz, tmpyzloc, yzloc);

        Real_type ebasisy[EB] = {0};
        edgebasis_y(ebasisy, tmpxz, xloctmpz, tmpxzloc, xzloc);

        Real_type dbasisz[EB] = {0};
        curl_edgebasis_z(dbasisz, tmpz, zloc);

        // Per-element J^{-1}, (1/|J|) J^T and quadrature weights
        Real_type jinvxx[VL], jinvxy[VL], jinvxz[VL],
                  jinvyx[VL], jinvyy[VL], jinvyz[VL],
                  jinvzx[VL], jinvzy[VL], jinvzz[VL];
        Real_type cxx[VL], cxy[VL], cxz[VL],
                  cyx[VL], cyy[VL], cyz[VL],
                  czx[VL], czy[VL], czz[VL];
        Real_type walpha[VL], wbeta[VL];

        EDGE3D_OMP_SIMD
        for (Int_type e = 0; e < VL; e++) {
          const Real_type jxx = soa_dxi (x, e, tmpyz, yloctmpz, tmpyzloc, yzloc);
          const Real_type jxy = soa_dxi (y, e, tmpyz, yloctmpz, tmpyzloc, yzloc);
          const Real_type jxz = soa_dxi (z, e, tmpyz, yloctmpz, tmpyzloc, yzloc);
          const Real_type jyx = soa_deta(x, e, tmpxz, xloctmpz, tmpxzloc, xzloc);
          const Real_type jyy = soa_deta(y, e, tmpxz, xloctmpz, tmpxzloc, xzloc);
          const Real_type jyz = soa_deta(z, e, tmpxz, xloctmpz, tmpxzloc, xzloc);

          Real_type ixx, ixy, ixz,
                    iyx, iyy, iyz,
                    izx, izy, izz,
                    detj_unfixed, detj, abs_detj, invdetj;

          jacobian_inv(
            jxx, jxy, jxz,
            jyx, jyy, jyz,
            jzx[e], jzy[e], jzz[e],
            detj_cc[e], detj_tol,
            ixx, ixy, ixz,
            iyx, iyy, iyz,
            izx, izy, izz,
            detj_unfixed, detj, abs_detj, invdetj);

          jinvxx[e] = ixx; jinvxy[e] = ixy; jinvxz[e] = ixz;
          jinvyx[e] = iyx; jinvyy[e] = iyy; jinvyz[e] = iyz;
          jinvzx[e] = izx; jinvzy[e] = izy; jinvzz[e] = izz;

          const Real_type detjwgts = wgts*abs_detj;
          const Real_type inv_abs_detj = 1./(abs_detj+ptiny);

          walpha[e] = detjwgts*alpha;
          wbeta[e]  = detjwgts*beta;

          cxx[e] = jxx   *inv_abs_detj;
          cxy[e] = jyx   *inv_abs_detj;
          cxz[e] = jzx[e]*inv_abs_detj;
          cyx[e] = jxy   *inv_abs_detj;
          cyy[e] = jyy   *inv_abs_detj;
          cyz[e] = jzy[e]*inv_abs_detj;
          czx[e] = jxz   *inv_abs_detj;
          czy[e] = jyz   *inv_abs_detj;
          czz[e] = jzz[e]*inv_abs_detj;
        }

        // Transformed basis: w_i <- J^{-1} w_i, Curl(w_i) <- (1/|J|) J^T Curl(w_i)
        Real_type tebasisx[EB][VL], tebasisy[EB][VL], tebasisz[EB][VL];
        Real_type tdbasisx[EB][VL], tdbasisy[EB][VL], tdbasisz[EB][VL];

        for (Int_type m = 0; m < EB; m++) {
          EDGE3D_OMP_SIMD
          for (Int_type e = 0; e < VL; e++) {
            tebasisx[m][e] = jinvxx[e]*ebasisx[m] + jinvxy[e]*ebasisy[m] + jinvxz[e]*ebasisz[m];
            tebasisy[m][e] = jinvyx[e]*ebasisx[m] + jinvyy[e]*ebasisy[m] + jinvyz[e]*ebasisz[m];
            tebasisz[m][e] = jinvzx[e]*ebasisx[m] + jinvzy[e]*ebasisy[m] + jinvzz[e]*ebasisz[m];

            tdbasisx[m][e] = cxx[e]*dbasisx[m] + cxy[e]*dbasisy[m] + cxz[e]*dbasisz[m];
            tdbasisy[m][e] = cyx[e]*dbasisx[m] + cyy[e]*dbasisy[m] + cyz[e]*dbasisz[m];
            tdbasisz[m][e] = czx[e]*dbasisx[m] + czy[e]*dbasisy[m] + czz[e]*dbasisz[m];
          }
        }

        // alpha*<w_i, w_j> + beta*<Curl(w_i), Curl(w_j)>, upper triangle
        for (Int_type p = 0; p < EB; p++) {
          for (Int_type m = p; m < EB; m++) {
            EDGE3D_OMP_SIMD
            for (Int_type e = 0; e < VL; e++) {
              matrix[p][m][e] += walpha[e]*(tebasisx[p][e]*tebasisx[m][e] +
                                            tebasisy[p][e]*tebasisy[m][e] +
                                            tebasisz[p][e]*tebasisz[m][e]);
              matrix[p][m][e] += wbeta[e] *(tdbasisx[p][e]*tdbasisx[m][e] +
                                            tdbasisy[p][e]*tdbasisy[m][e] +
                                            tdbasisz[p][e]*tdbasisz[m][e]);
            }
          }
        }
      }
    }
  }

  // Symmetric fill of the lower triangle
  for (Int_type p = 0; p < EB; p++) {
    for (Int_type m = p+1; m < EB; m++) {
      EDGE3D_OMP_SIMD
      for (Int_type e = 0; e < VL; e++) {
        matrix[m][p][e] = matrix[p][m][e];
      }
    }
  }
}

#define EDGE3D_DATA_SETUP \
  Real_ptr x = m_x; \
  Real_ptr y = m_y; \
  Real_ptr z = m_z; \
  Real_ptr sum = m_sum; \
\
  Real_ptr x0,x1,x2,x3,x4,x5,x6,x7 ; \
  Real_ptr y0,y1,y2,y3,y4,y5,y6,y7 ; \
  Real_ptr z0,z1,z2,z3,z4,z5,z6,z7 ; \
\
  NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ; \
  NDPTRSET(m_domain->jp, m_domain->kp, y,y0,y1,y2,y3,y4,y5,y6,y7) ; \
  NDPTRSET(m_domain->jp, m_domain->kp, z,z0,z1,z2,z3,z4,z5,z6,z7) ;

#define EDGE3D_BODY \
  rajaperf::Real_type X[NB] = {x0[i],x1[i],x2[i],x3[i],x4[i],x5[i],x6[i],x7[i]};\
  rajaperf::Real_type Y[NB] = {y0[i],y1[i],y2[i],y3[i],y4[i],y5[i],y6[i],y7[i]};\
  rajaperf::Real_type Z[NB] = {z0[i],z1[i],z2[i],z3[i],z4[i],z5[i],z6[i],z7[i]};\
  rajaperf::Real_type edge_matrix[EB][EB];\
  edge_MpSmatrix(X, Y, Z, 1.0, 1.0, 0.0, 1.0, NQ_1D, edge_matrix);\
  rajaperf::Real_type local_sum = 0.0;\
  for (rajaperf::Int_type m = 0; m < EB; m++) {\
    rajaperf::Real_type check = 0.0;\
    for (rajaperf::Int_type p = 0; p < EB; p++) {\
      check += edge_matrix[m][p];\
    }\
    local_sum += check;\
  }\
  sum[i] = local_sum;\

// Processes elements [i0, min(i0+EDGE3D_VL, iend)): AoS->SoA gather, SoA
// compute, SoA->AoS scatter. Tail lanes replicate the last valid element.
#define EDGE3D_SIMD_BODY \
  const rajaperf::Index_type nel = \
    (iend - i0 < EDGE3D_VL) ? (iend - i0) : EDGE3D_VL;\
  rajaperf::Real_type X[NB][EDGE3D_VL];\
  rajaperf::Real_type Y[NB][EDGE3D_VL];\
  rajaperf::Real_type Z[NB][EDGE3D_VL];\
  EDGE3D_OMP_SIMD \
  for (rajaperf::Int_type e = 0; e < EDGE3D_VL; e++) {\
    const rajaperf::Index_type ie = i0 + ((e < nel) ? e : nel - 1);\
    X[0][e] = x0[ie]; X[1][e] = x1[ie]; X[2][e] = x2[ie]; X[3][e] = x3[ie];\
    X[4][e] = x4[ie]; X[5][e] = x5[ie]; X[6][e] = x6[ie]; X[7][e] = x7[ie];\
    Y[0][e] = y0[ie]; Y[1][e] = y1[ie]; Y[2][e] = y2[ie]; Y[3][e] = y3[ie];\
    Y[4][e] = y4[ie]; Y[5][e] = y5[ie]; Y[6][e] = y6[ie]; Y[7][e] = y7[ie];\
    Z[0][e] = z0[ie]; Z[1][e] = z1[ie]; Z[2][e] = z2[ie]; Z[3][e] = z3[ie];\
    Z[4][e] = z4[ie]; Z[5][e] = z5[ie]; Z[6][e] = z6[ie]; Z[7][e] = z7[ie];\
  }\
  rajaperf::Real_type edge_matrix[EB][EB][EDGE3D_VL];\
  edge_MpSmatrix_simd(X, Y, Z, 1.0, 1.0, 0.0, 1, NQ_1D, edge_matrix);\
  rajaperf::Real_type local_sum[EDGE3D_VL];\
  EDGE3D_OMP_SIMD \
  for (rajaperf::Int_type e = 0; e < EDGE3D_VL; e++) {\
    local_sum[e] = 0.0;\
  }\
  for (rajaperf::Int_type m = 0; m < EB; m++) {\
    rajaperf::Real_type check[EDGE3D_VL];\
    EDGE3D_OMP_SIMD \
    for (rajaperf::Int_type e = 0; e < EDGE3D_VL; e++) {\
      check[e] = 0.0;\
    }\
    for (rajaperf::Int_type p = 0; p < EB; p++) {\
      EDGE3D_OMP_SIMD \
      for (rajaperf::Int_type e = 0; e < EDGE3D_VL; e++) {\
        check[e] += edge_matrix[m][p][e];\
      }\
    }\
    EDGE3D_OMP_SIMD \
    for (rajaperf::Int_type e = 0; e < EDGE3D_VL; e++) {\
      local_sum[e] += check[e];\
    }\
  }\
  EDGE3D_OMP_SIMD \
  for (rajaperf::Int_type e = 0; e < nel; e++) {\
    sum[i0 + e] = local_sum[e];\
  }

#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace apps
{
class ADomain;

class EDGE3D : public KernelBase
{
public:

  EDGE3D(const RunParams& params);

  ~EDGE3D();

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
  void runOpenMPVariant(VariantID vid);
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

  Real_ptr m_x;
  Real_ptr m_y;
  Real_ptr m_z;
  Real_ptr m_sum;

  std::unique_ptr<ADomain> m_domain;
  Index_type m_array_length;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
