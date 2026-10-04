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
#include "FEM_SIMD.hpp"

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

//
// Same algorithm as edge_MpSmatrix_sum_avx512 below, in plain C++ with one
// compiler-vectorized lane loop per quadrature point.
//
template <rajaperf::Int_type quad_order>
RAJAPERF_FEM_NOINLINE void edge_MpSmatrix_sum_simd(
  const rajaperf::Real_type *const (&xn)[NB],
  const rajaperf::Real_type *const (&yn)[NB],
  const rajaperf::Real_type *const (&zn)[NB],
  const rajaperf::Index_type nel,
  const rajaperf::Real_type  alpha,
  const rajaperf::Real_type  beta,
  const rajaperf::Real_type  detj_tol,
  const rajaperf::Int_type   quad_type,
  rajaperf::Real_type       *sum)
{
  using rajaperf::Real_type;
  using rajaperf::Int_type;
  using rajaperf::Index_type;
  constexpr Int_type VL = EDGE3D_VL;
  constexpr Int_type NU = EB * (EB + 1) / 2;

  alignas(64) Real_type dxi[3][4][VL], deta[3][4][VL], dzeta[3][4][VL];
  alignas(64) Real_type detj_cc[VL];
  FEM_OMP_SIMD
  for (Int_type e = 0; e < VL; e++) {
    const Index_type l = (e < nel) ? e : nel - 1;
    Real_type C[3][NB];
    for (Int_type n = 0; n < NB; n++) {
      C[0][n] = xn[n][l];
      C[1][n] = yn[n][l];
      C[2][n] = zn[n][l];
    }
    for (Int_type c = 0; c < 3; c++) {
      dxi[c][0][e] = C[c][1] - C[c][0];   dxi[c][1][e] = C[c][2] - C[c][3];
      dxi[c][2][e] = C[c][5] - C[c][4];   dxi[c][3][e] = C[c][6] - C[c][7];
      deta[c][0][e] = C[c][3] - C[c][0];  deta[c][1][e] = C[c][2] - C[c][1];
      deta[c][2][e] = C[c][7] - C[c][4];  deta[c][3][e] = C[c][6] - C[c][5];
      dzeta[c][0][e] = C[c][4] - C[c][0]; dzeta[c][1][e] = C[c][5] - C[c][1];
      dzeta[c][2][e] = C[c][6] - C[c][2]; dzeta[c][3][e] = C[c][7] - C[c][3];
    }
    Real_type J[3][3];
    for (Int_type c = 0; c < 3; c++) {
      J[0][c] = 0.25 * (dxi[c][0][e] + dxi[c][1][e] + dxi[c][2][e] + dxi[c][3][e]);
      J[1][c] = 0.25 * (deta[c][0][e] + deta[c][1][e] + deta[c][2][e] + deta[c][3][e]);
      J[2][c] = 0.25 * (dzeta[c][0][e] + dzeta[c][1][e] + dzeta[c][2][e] + dzeta[c][3][e]);
    }
    detj_cc[e] = compute_detj(J[0][0], J[0][1], J[0][2], J[1][0], J[1][1],
                              J[1][2], J[2][0], J[2][1], J[2][2]);
  }

  Real_type qpts_1d[MAX_QUAD_ORDER];
  Real_type wgts_1d[MAX_QUAD_ORDER];
  get_quadrature_rule(quad_type, quad_order, qpts_1d, wgts_1d);

  alignas(64) Real_type Mu[NU][VL];
  RAJAPERF_FEM_STACK_ALIGN(Mu);
  for (Int_type u = 0; u < NU; u++) {
    FEM_OMP_SIMD
    for (Int_type e = 0; e < VL; e++) {
      Mu[u][e] = 0.0;
    }
  }

  constexpr Int_type nz1[3] = {1, 0, 0};
  constexpr Int_type nz2[3] = {2, 2, 1};
  constexpr Int_type sym[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};

  for (Int_type i = 0; i < quad_order; i++) {
    const Real_type xloc = qpts_1d[i];
    const Real_type tmpx = 1. - xloc;
    RAJAPERF_FEM_UNROLL
    for (Int_type j = 0; j < quad_order; j++) {
      const Real_type yloc = qpts_1d[j];
      const Real_type wgtxy = wgts_1d[i]*wgts_1d[j];
      const Real_type tmpy = 1. - yloc;
      const Real_type tmpxy    = tmpx*tmpy;
      const Real_type xyloc    = xloc*yloc;
      const Real_type tmpxyloc = tmpx*yloc;
      const Real_type xloctmpy = xloc*tmpy;
      RAJAPERF_FEM_UNROLL
      for (Int_type k = 0; k < quad_order; k++) {
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

        const Real_type s[EB] = {tmpyz, yloctmpz, tmpyzloc, yzloc,
                                 tmpxz, xloctmpz, tmpxzloc, xzloc,
                                 tmpxy, xloctmpy, tmpxyloc, xyloc};
        const Real_type db[EB][3] = {
          {0.0, -tmpy,  tmpz}, {0.0, -yloc, -tmpz},
          {0.0,  tmpy,  zloc}, {0.0,  yloc, -zloc},
          { tmpx, 0.0, -tmpz}, { xloc, 0.0,  tmpz},
          {-tmpx, 0.0, -zloc}, {-xloc, 0.0,  zloc},
          {-tmpx,  tmpy, 0.0}, {-xloc, -tmpy, 0.0},
          { tmpx,  yloc, 0.0}, { xloc, -yloc, 0.0}};

        FEM_OMP_SIMD
        for (Int_type e = 0; e < VL; e++) {
          auto comb = [e](const Real_type (&d)[4][VL], Real_type w0,
                          Real_type w1, Real_type w2, Real_type w3) {
            return d[0][e]*w0 + d[1][e]*w1 + d[2][e]*w2 + d[3][e]*w3;
          };
          const Real_type jxx = comb(dxi[0], tmpyz, yloctmpz, tmpyzloc, yzloc);
          const Real_type jxy = comb(dxi[1], tmpyz, yloctmpz, tmpyzloc, yzloc);
          const Real_type jxz = comb(dxi[2], tmpyz, yloctmpz, tmpyzloc, yzloc);
          const Real_type jyx = comb(deta[0], tmpxz, xloctmpz, tmpxzloc, xzloc);
          const Real_type jyy = comb(deta[1], tmpxz, xloctmpz, tmpxzloc, xzloc);
          const Real_type jyz = comb(deta[2], tmpxz, xloctmpz, tmpxzloc, xzloc);
          const Real_type jzx = comb(dzeta[0], tmpxy, xloctmpy, xyloc, tmpxyloc);
          const Real_type jzy = comb(dzeta[1], tmpxy, xloctmpy, xyloc, tmpxyloc);
          const Real_type jzz = comb(dzeta[2], tmpxy, xloctmpy, xyloc, tmpxyloc);

          const Real_type kxx = jyy*jzz - jyz*jzy;
          const Real_type kxy = jxz*jzy - jxy*jzz;
          const Real_type kxz = jxy*jyz - jxz*jyy;
          const Real_type kyx = jyz*jzx - jyx*jzz;
          const Real_type kyy = jxx*jzz - jxz*jzx;
          const Real_type kyz = jxz*jyx - jxx*jyz;
          const Real_type kzx = jyx*jzy - jyy*jzx;
          const Real_type kzy = jxy*jzx - jxx*jzy;
          const Real_type kzz = jxx*jyy - jxy*jyx;

          const Real_type detj_unfixed = jzx*kxz + jzy*kyz + jzz*kzz;
          const Real_type detj =
            (fabs(detj_unfixed) < detj_tol*fabs(detj_cc[e])) ? detj_cc[e]
                                                              : detj_unfixed;
          const Real_type abs_detj = fabs(detj);
          const Real_type inv_detj = 1.0/(detj + ptiny);
          const Real_type inv_abs_detj = 1.0/(abs_detj + ptiny);
          const Real_type wdet = wgts*abs_detj;
          const Real_type sG = alpha*wdet*inv_detj*inv_detj;
          const Real_type sH = beta*wdet*inv_abs_detj*inv_abs_detj;

          const Real_type WG[6] = {
            sG*(kxx*kxx + kyx*kyx + kzx*kzx), sG*(kxx*kxy + kyx*kyy + kzx*kzy),
            sG*(kxx*kxz + kyx*kyz + kzx*kzz), sG*(kxy*kxy + kyy*kyy + kzy*kzy),
            sG*(kxy*kxz + kyy*kyz + kzy*kzz), sG*(kxz*kxz + kyz*kyz + kzz*kzz)};
          const Real_type WH[6] = {
            sH*(jxx*jxx + jxy*jxy + jxz*jxz), sH*(jxx*jyx + jxy*jyy + jxz*jyz),
            sH*(jxx*jzx + jxy*jzy + jxz*jzz), sH*(jyx*jyx + jyy*jyy + jyz*jyz),
            sH*(jyx*jzx + jyy*jzy + jyz*jzz), sH*(jzx*jzx + jzy*jzy + jzz*jzz)};

          Real_type v[EB][3];
          RAJAPERF_FEM_UNROLL
          for (Int_type m = 0; m < EB; m++) {
            const Int_type a1 = nz1[m / 4], a2 = nz2[m / 4];
            RAJAPERF_FEM_UNROLL
            for (Int_type c = 0; c < 3; c++) {
              v[m][c] = WH[sym[c][a1]]*db[m][a1] + WH[sym[c][a2]]*db[m][a2];
            }
          }

          Int_type u = 0;
          RAJAPERF_FEM_UNROLL
          for (Int_type p = 0; p < EB; p++) {
            const Int_type gp = p / 4;
            const Int_type a1 = nz1[gp], a2 = nz2[gp];
            RAJAPERF_FEM_UNROLL
            for (Int_type m = p; m < EB; m++, u++) {
              Mu[u][e] += s[m]*(s[p]*WG[sym[gp][m / 4]]) +
                          db[p][a1]*v[m][a1] + db[p][a2]*v[m][a2];
            }
          }
        }
      }
    }
  }

  FEM_OMP_SIMD
  for (Int_type e = 0; e < VL; e++) {
    Real_type sdiag = 0.0, soff = 0.0;
    Int_type u = 0;
    for (Int_type p = 0; p < EB; p++) {
      sdiag += Mu[u++][e];
      for (Int_type m = p + 1; m < EB; m++) {
        soff += Mu[u++][e];
      }
    }
    if (e < nel) {
      sum[e] = sdiag + 2.0*soff;
    }
  }
}

//
// Element independent quadrature data of the register-blocked "metric_rb"
// kernels, built once per run: weights, the soa_dxi / soa_deta / soa_dzeta
// weights, edge basis scalars s_m and the two nonzero components of Curl(w_m).
//
template <rajaperf::Int_type quad_order>
struct Edge3DQpTable
{
  static constexpr rajaperf::Int_type NQ = quad_order * quad_order * quad_order;
  rajaperf::Real_type wgt[NQ];
  rajaperf::Real_type cw[NQ][3][4];
  rajaperf::Real_type s[NQ][EB];
  rajaperf::Real_type db[NQ][EB][2];

  explicit Edge3DQpTable(const rajaperf::Int_type quad_type)
  {
    using rajaperf::Real_type;
    using rajaperf::Int_type;
    Real_type qpts_1d[MAX_QUAD_ORDER];
    Real_type wgts_1d[MAX_QUAD_ORDER];
    get_quadrature_rule(quad_type, quad_order, qpts_1d, wgts_1d);
    constexpr Int_type nz[3][2] = {{1, 2}, {0, 2}, {0, 1}};
    for (Int_type i = 0; i < quad_order; i++) {
      const Real_type xloc = qpts_1d[i], tmpx = 1. - xloc;
      for (Int_type j = 0; j < quad_order; j++) {
        const Real_type yloc = qpts_1d[j], tmpy = 1. - yloc;
        for (Int_type k = 0; k < quad_order; k++) {
          const Real_type zloc = qpts_1d[k], tmpz = 1. - zloc;
          const Int_type q = (i * quad_order + j) * quad_order + k;
          const Real_type sv[EB] = {
            tmpy*tmpz, yloc*tmpz, tmpy*zloc, yloc*zloc,
            tmpx*tmpz, xloc*tmpz, tmpx*zloc, xloc*zloc,
            tmpx*tmpy, xloc*tmpy, tmpx*yloc, xloc*yloc};
          const Real_type dbv[EB][3] = {
            {0.0, -tmpy,  tmpz}, {0.0, -yloc, -tmpz},
            {0.0,  tmpy,  zloc}, {0.0,  yloc, -zloc},
            { tmpx, 0.0, -tmpz}, { xloc, 0.0,  tmpz},
            {-tmpx, 0.0, -zloc}, {-xloc, 0.0,  zloc},
            {-tmpx,  tmpy, 0.0}, {-xloc, -tmpy, 0.0},
            { tmpx,  yloc, 0.0}, { xloc, -yloc, 0.0}};
          wgt[q] = wgts_1d[i]*wgts_1d[j]*wgts_1d[k];
          for (Int_type a = 0; a < 4; a++) {
            cw[q][0][a] = sv[a];
            cw[q][1][a] = sv[4 + a];
          }
          cw[q][2][0] = sv[8];  cw[q][2][1] = sv[9];
          cw[q][2][2] = sv[11]; cw[q][2][3] = sv[10];
          for (Int_type m = 0; m < EB; m++) {
            s[q][m] = sv[m];
            db[q][m][0] = dbv[m][nz[m / 4][0]];
            db[q][m][1] = dbv[m][nz[m / 4][1]];
          }
        }
      }
    }
  }
};

// u index of (p, m), m >= p, in the row-major upper triangle
constexpr rajaperf::Int_type edge3d_upper_index(rajaperf::Int_type p,
                                                rajaperf::Int_type m)
{
  return p * EB - p * (p - 1) / 2 + (m - p);
}

//
// "metric_rb": the metric reformulation reordered so the matrix never goes
// through memory. Phase 1 stores the 12 per-qp metric terms
// WG = walpha K^T K / detj^2 and WH = wbeta J J^T / |detj|^2 for all qps
// (independent across qps); phase 2 contracts them over all qps with the
// accumulators of one 4 x MT tile of the upper triangle held in registers.
// Plain C++ version; edge_MpSmatrix_sum_rb_avx512 is the same algorithm.
//
template <rajaperf::Int_type quad_order, bool STORE_MATRIX = false, int NV = 1>
RAJAPERF_FEM_NOINLINE void edge_MpSmatrix_sum_rb_simd(
  const rajaperf::Real_type *const (&xn)[NB],
  const rajaperf::Real_type *const (&yn)[NB],
  const rajaperf::Real_type *const (&zn)[NB],
  const rajaperf::Index_type nel,
  const rajaperf::Real_type  alpha,
  const rajaperf::Real_type  beta,
  const rajaperf::Real_type  detj_tol,
  const Edge3DQpTable<quad_order> &qt,
  rajaperf::Real_type       *sum,
  rajaperf::Real_type       *matrix = nullptr)
{
  using rajaperf::Real_type;
  using rajaperf::Int_type;
  using rajaperf::Index_type;
  using rajaperf::fem_simd::static_for;
  constexpr Int_type VL = EDGE3D_VL * NV;
  constexpr Int_type NQ = Edge3DQpTable<quad_order>::NQ;
  static constexpr Int_type nz[3][2] = {{1, 2}, {0, 2}, {0, 1}};
  static constexpr Int_type sym[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};

  alignas(64) Real_type dxi[3][4][VL], deta[3][4][VL], dzeta[3][4][VL];
  alignas(64) Real_type detj_cc[VL];
  FEM_OMP_SIMD
  for (Int_type e = 0; e < VL; e++) {
    const Index_type l = (e < nel) ? e : nel - 1;
    Real_type C[3][NB];
    for (Int_type n = 0; n < NB; n++) {
      C[0][n] = xn[n][l];
      C[1][n] = yn[n][l];
      C[2][n] = zn[n][l];
    }
    for (Int_type c = 0; c < 3; c++) {
      dxi[c][0][e] = C[c][1] - C[c][0];   dxi[c][1][e] = C[c][2] - C[c][3];
      dxi[c][2][e] = C[c][5] - C[c][4];   dxi[c][3][e] = C[c][6] - C[c][7];
      deta[c][0][e] = C[c][3] - C[c][0];  deta[c][1][e] = C[c][2] - C[c][1];
      deta[c][2][e] = C[c][7] - C[c][4];  deta[c][3][e] = C[c][6] - C[c][5];
      dzeta[c][0][e] = C[c][4] - C[c][0]; dzeta[c][1][e] = C[c][5] - C[c][1];
      dzeta[c][2][e] = C[c][6] - C[c][2]; dzeta[c][3][e] = C[c][7] - C[c][3];
    }
    Real_type J[3][3];
    for (Int_type c = 0; c < 3; c++) {
      J[0][c] = 0.25 * (dxi[c][0][e] + dxi[c][1][e] + dxi[c][2][e] + dxi[c][3][e]);
      J[1][c] = 0.25 * (deta[c][0][e] + deta[c][1][e] + deta[c][2][e] + deta[c][3][e]);
      J[2][c] = 0.25 * (dzeta[c][0][e] + dzeta[c][1][e] + dzeta[c][2][e] + dzeta[c][3][e]);
    }
    detj_cc[e] = compute_detj(J[0][0], J[0][1], J[0][2], J[1][0], J[1][1],
                              J[1][2], J[2][0], J[2][1], J[2][2]);
  }

  // phase 1: W[q] = {WG[6], WH[6]}
  alignas(64) Real_type W[NQ][12][VL];
  RAJAPERF_FEM_STACK_ALIGN(W);
  for (Int_type q = 0; q < NQ; q++) {
    const Real_type (&cw)[3][4] = qt.cw[q];
    const Real_type wgts = qt.wgt[q];
    FEM_OMP_SIMD
    for (Int_type e = 0; e < VL; e++) {
      auto comb = [e](const Real_type (&d)[4][VL], const Real_type (&w)[4]) {
        return d[0][e]*w[0] + d[1][e]*w[1] + d[2][e]*w[2] + d[3][e]*w[3];
      };
      const Real_type jxx = comb(dxi[0], cw[0]);
      const Real_type jxy = comb(dxi[1], cw[0]);
      const Real_type jxz = comb(dxi[2], cw[0]);
      const Real_type jyx = comb(deta[0], cw[1]);
      const Real_type jyy = comb(deta[1], cw[1]);
      const Real_type jyz = comb(deta[2], cw[1]);
      const Real_type jzx = comb(dzeta[0], cw[2]);
      const Real_type jzy = comb(dzeta[1], cw[2]);
      const Real_type jzz = comb(dzeta[2], cw[2]);

      const Real_type kxx = jyy*jzz - jyz*jzy;
      const Real_type kxy = jxz*jzy - jxy*jzz;
      const Real_type kxz = jxy*jyz - jxz*jyy;
      const Real_type kyx = jyz*jzx - jyx*jzz;
      const Real_type kyy = jxx*jzz - jxz*jzx;
      const Real_type kyz = jxz*jyx - jxx*jyz;
      const Real_type kzx = jyx*jzy - jyy*jzx;
      const Real_type kzy = jxy*jzx - jxx*jzy;
      const Real_type kzz = jxx*jyy - jxy*jyx;

      const Real_type detj_unfixed = jzx*kxz + jzy*kyz + jzz*kzz;
      const Real_type detj =
        (fabs(detj_unfixed) < detj_tol*fabs(detj_cc[e])) ? detj_cc[e]
                                                          : detj_unfixed;
      const Real_type abs_detj = fabs(detj);
      const Real_type inv_detj = 1.0/(detj + ptiny);
      const Real_type inv_abs_detj = 1.0/(abs_detj + ptiny);
      const Real_type wdet = wgts*abs_detj;
      const Real_type sG = alpha*wdet*inv_detj*inv_detj;
      const Real_type sH = beta*wdet*inv_abs_detj*inv_abs_detj;

      W[q][0][e]  = sG*(kxx*kxx + kyx*kyx + kzx*kzx);
      W[q][1][e]  = sG*(kxx*kxy + kyx*kyy + kzx*kzy);
      W[q][2][e]  = sG*(kxx*kxz + kyx*kyz + kzx*kzz);
      W[q][3][e]  = sG*(kxy*kxy + kyy*kyy + kzy*kzy);
      W[q][4][e]  = sG*(kxy*kxz + kyy*kyz + kzy*kzz);
      W[q][5][e]  = sG*(kxz*kxz + kyz*kyz + kzz*kzz);
      W[q][6][e]  = sH*(jxx*jxx + jxy*jxy + jxz*jxz);
      W[q][7][e]  = sH*(jxx*jyx + jxy*jyy + jxz*jyz);
      W[q][8][e]  = sH*(jxx*jzx + jxy*jzy + jxz*jzz);
      W[q][9][e]  = sH*(jyx*jyx + jyy*jyy + jyz*jyz);
      W[q][10][e] = sH*(jyx*jzx + jyy*jzy + jyz*jzz);
      W[q][11][e] = sH*(jzx*jzx + jzy*jzy + jzz*jzz);
    }
  }

  // phase 2: one 4 x 4 group block (gp, gm) of the upper triangle at a time
  alignas(64) Real_type tot[VL];
  FEM_OMP_SIMD
  for (Int_type e = 0; e < VL; e++) {
    Real_type sdiag = 0.0, soff = 0.0;
    static_for<3>([&](auto GP) {
      constexpr Int_type gp = decltype(GP)::value;
      static_for<3>([&](auto GM) {
        constexpr Int_type gm = decltype(GM)::value;
        if constexpr (gm >= gp) {
          Real_type acc[4][4] = {};
          RAJAPERF_FEM_UNROLL
          for (Int_type q = 0; q < NQ; q++) {
            const Real_type (&s)[EB] = qt.s[q];
            const Real_type (&db)[EB][2] = qt.db[q];
            Real_type v[4][2];
            RAJAPERF_FEM_UNROLL
            for (Int_type mt = 0; mt < 4; mt++) {
              const Int_type m = 4 * gm + mt;
              RAJAPERF_FEM_UNROLL
              for (Int_type ci = 0; ci < 2; ci++) {
                const Int_type c = nz[gp][ci];
                v[mt][ci] = W[q][6 + sym[c][nz[gm][1]]][e]*db[m][1] +
                            W[q][6 + sym[c][nz[gm][0]]][e]*db[m][0];
              }
            }
            const Real_type wg = W[q][sym[gp][gm]][e];
            RAJAPERF_FEM_UNROLL
            for (Int_type pt = 0; pt < 4; pt++) {
              const Int_type p = 4 * gp + pt;
              const Real_type ap = s[p]*wg;
              RAJAPERF_FEM_UNROLL
              for (Int_type mt = 0; mt < 4; mt++) {
                if (4 * gm + mt >= p) {
                  acc[pt][mt] += s[4 * gm + mt]*ap + db[p][0]*v[mt][0] +
                                 db[p][1]*v[mt][1];
                }
              }
            }
          }
          RAJAPERF_FEM_UNROLL
          for (Int_type pt = 0; pt < 4; pt++) {
            RAJAPERF_FEM_UNROLL
            for (Int_type mt = 0; mt < 4; mt++) {
              const Int_type p = 4 * gp + pt, m = 4 * gm + mt;
              if (m > p) {
                soff += acc[pt][mt];
              } else if (m == p) {
                sdiag += acc[pt][mt];
              }
              if constexpr (STORE_MATRIX) {
                if (m >= p) {
                  matrix[edge3d_upper_index(p, m) * VL + e] = acc[pt][mt];
                }
              }
            }
          }
        }
      });
    });
    tot[e] = sdiag + 2.0*soff;
  }
  FEM_OMP_SIMD
  for (Int_type e = 0; e < VL; e++) {
    if (e < nel) {
      sum[e] = tot[e];
    }
  }
}

#if defined(RAJAPERF_FEM_AVX512)
template <int NV>
RAJA_INLINE rajaperf::fem_simd::vdx<NV> edge3d_comb(
  const rajaperf::fem_simd::vdx<NV> (&d)[4],
  rajaperf::Real_type w0, rajaperf::Real_type w1,
  rajaperf::Real_type w2, rajaperf::Real_type w3)
{
  using namespace rajaperf::fem_simd;
  return vfma(d[3], vsplat<NV>(vset1(w3)),
         vfma(d[2], vsplat<NV>(vset1(w2)),
         vfma(d[1], vsplat<NV>(vset1(w1)),
              vmul(d[0], vsplat<NV>(vset1(w0))))));
}

//
// Batch prelude of the AVX-512 kernels: loads 8*NV elements (tail lanes, and
// whole tail vectors, replicate the last valid element), forms the node
// differences of soa_dxi / soa_deta / soa_dzeta and returns detj at the center.
//
template <int NV>
RAJA_INLINE rajaperf::fem_simd::vdx<NV> edge3d_avx512_load(
  const rajaperf::Real_type *const (&xn)[NB],
  const rajaperf::Real_type *const (&yn)[NB],
  const rajaperf::Real_type *const (&zn)[NB],
  const rajaperf::Index_type nel,
  rajaperf::fem_simd::vdx<NV> (&dxi)[3][4],
  rajaperf::fem_simd::vdx<NV> (&deta)[3][4],
  rajaperf::fem_simd::vdx<NV> (&dzeta)[3][4])
{
  using namespace rajaperf::fem_simd;
  using rajaperf::Real_type;
  using rajaperf::Int_type;
  using rajaperf::Index_type;
  using V = vdx<NV>;
  constexpr Index_type BL = FEM_VL * NV;

  const Index_type last_blk = ((nel - 1) / FEM_VL) * FEM_VL;
  Index_type off[NV];
  __m512i cidx[NV];
  vmask lm[NV];
  RAJAPERF_FEM_UNROLL
  for (int k = 0; k < NV; k++) {
    off[k] = (k * FEM_VL < last_blk) ? k * FEM_VL : last_blk;
    cidx[k] = (k * FEM_VL < nel)
      ? _mm512_min_epi64(_mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0),
                         _mm512_set1_epi64(nel - 1 - off[k]))
      : _mm512_set1_epi64(nel - 1 - off[k]);
    lm[k] = lane_mask(nel - off[k]);
  }
  auto load_lanes = [&](const Real_type *p) {
    V o;
    RAJAPERF_FEM_UNROLL
    for (int k = 0; k < NV; k++) {
      o.r[k] = (nel == BL) ? vload(p + k * FEM_VL)
                           : _mm512_permutexvar_pd(cidx[k],
                                                   vload(lm[k], p + off[k]));
    }
    return o;
  };

  alignas(64) V C[3][NB];
  for (Int_type n = 0; n < NB; n++) {
    C[0][n] = load_lanes(xn[n]);
    C[1][n] = load_lanes(yn[n]);
    C[2][n] = load_lanes(zn[n]);
  }

  for (Int_type c = 0; c < 3; c++) {
    dxi[c][0] = vsub(C[c][1], C[c][0]);   dxi[c][1] = vsub(C[c][2], C[c][3]);
    dxi[c][2] = vsub(C[c][5], C[c][4]);   dxi[c][3] = vsub(C[c][6], C[c][7]);
    deta[c][0] = vsub(C[c][3], C[c][0]);  deta[c][1] = vsub(C[c][2], C[c][1]);
    deta[c][2] = vsub(C[c][7], C[c][4]);  deta[c][3] = vsub(C[c][6], C[c][5]);
    dzeta[c][0] = vsub(C[c][4], C[c][0]); dzeta[c][1] = vsub(C[c][5], C[c][1]);
    dzeta[c][2] = vsub(C[c][6], C[c][2]); dzeta[c][3] = vsub(C[c][7], C[c][3]);
  }

  // detj = jzx*kxz + jzy*kyz + jzz*kzz with cofactors k of J
  const V jxx = edge3d_comb(dxi[0], 0.25, 0.25, 0.25, 0.25);
  const V jxy = edge3d_comb(dxi[1], 0.25, 0.25, 0.25, 0.25);
  const V jxz = edge3d_comb(dxi[2], 0.25, 0.25, 0.25, 0.25);
  const V jyx = edge3d_comb(deta[0], 0.25, 0.25, 0.25, 0.25);
  const V jyy = edge3d_comb(deta[1], 0.25, 0.25, 0.25, 0.25);
  const V jyz = edge3d_comb(deta[2], 0.25, 0.25, 0.25, 0.25);
  const V jzx = edge3d_comb(dzeta[0], 0.25, 0.25, 0.25, 0.25);
  const V jzy = edge3d_comb(dzeta[1], 0.25, 0.25, 0.25, 0.25);
  const V jzz = edge3d_comb(dzeta[2], 0.25, 0.25, 0.25, 0.25);
  const V kxz = vfms(jxy, jyz, vmul(jxz, jyy));
  const V kyz = vfms(jxz, jyx, vmul(jxx, jyz));
  const V kzz = vfms(jxx, jyy, vmul(jxy, jyx));
  return vfma(jzz, kzz, vfma(jzy, kyz, vmul(jzx, kxz)));
}

// sum[0:nel) of a batch of 8*NV lanes
template <int NV>
RAJA_INLINE void edge3d_avx512_store_sum(rajaperf::Real_type *sum,
                                         const rajaperf::Index_type nel,
                                         const rajaperf::fem_simd::vdx<NV> &tot)
{
  using namespace rajaperf::fem_simd;
  RAJAPERF_FEM_UNROLL
  for (int k = 0; k < NV; k++) {
    const rajaperf::Index_type nk = nel - k * FEM_VL;
    vstore(sum + k * FEM_VL, lane_mask(nk > 0 ? nk : 0), tot.r[k]);
  }
}

//
// AVX-512 variant of edge_MpSmatrix_simd + row sum for one batch.
// Uses the basis structure: w_m = s_m e_{m/4} and Curl(w_m) has two nonzero
// components, so with G = J^{-T}J^{-1} and H = J J^T / |J|^2 every matrix
// entry update is three FMAs:
//   M[p][m] += s_p s_m walpha G[p/4][m/4] + wbeta db_p^T H db_m.
// A batch is 8*NV elements held as NV independent zmm per value (vdx<NV>),
// so every dependency chain carries NV independent instructions.
//
template <rajaperf::Int_type quad_order, bool STORE_MATRIX = false,
          int NV = 1>
RAJAPERF_FEM_NOINLINE void edge_MpSmatrix_sum_avx512(
  const rajaperf::Real_type *const (&xn)[NB],
  const rajaperf::Real_type *const (&yn)[NB],
  const rajaperf::Real_type *const (&zn)[NB],
  const rajaperf::Index_type nel,
  const rajaperf::Real_type  alpha,
  const rajaperf::Real_type  beta,
  const rajaperf::Real_type  detj_tol,
  const rajaperf::Int_type   quad_type,
  rajaperf::Real_type       *sum,
  rajaperf::Real_type       *matrix = nullptr)
{
  using namespace rajaperf::fem_simd;
  using rajaperf::Real_type;
  using rajaperf::Int_type;
  using rajaperf::Index_type;
  using V = vdx<NV>;
  constexpr Index_type BL = FEM_VL * NV;
  auto bc = [](Real_type a) { return vsplat<NV>(vset1(a)); };

  alignas(64) V dxi[3][4], deta[3][4], dzeta[3][4];
  const V detj_cc = edge3d_avx512_load<NV>(xn, yn, zn, nel, dxi, deta, dzeta);
  auto comb = [](const V (&d)[4], Real_type w0, Real_type w1,
                 Real_type w2, Real_type w3) {
    return edge3d_comb(d, w0, w1, w2, w3);
  };
  const V tol_abs_detj_cc = vmul(bc(detj_tol), vabs(detj_cc));
  const V vtiny = bc(ptiny);
  const V vone = bc(1.0);

  Real_type qpts_1d[MAX_QUAD_ORDER];
  Real_type wgts_1d[MAX_QUAD_ORDER];
  get_quadrature_rule(quad_type, quad_order, qpts_1d, wgts_1d);

  // upper triangle, row major
  constexpr Int_type NU = EB * (EB + 1) / 2;
  alignas(64) V Mu[NU];
  RAJAPERF_FEM_STACK_ALIGN(Mu);
  for (Int_type u = 0; u < NU; u++) {
    Mu[u] = vsplat<NV>(vzero());
  }

  // nonzero components of Curl(w_m) for m in group m/4
  constexpr Int_type nz1[3] = {1, 0, 0};
  constexpr Int_type nz2[3] = {2, 2, 1};
  // index into symmetric 3x3 storage {00, 01, 02, 11, 12, 22}
  constexpr Int_type sym[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};

  // NV > 1: keep the qp loops rolled, the unrolled body would overflow L1i/DSB
  constexpr Int_type qp_unroll = (NV == 1) ? quad_order : 1;
#if defined(__clang__)
#define EDGE3D_QP_UNROLL _Pragma("clang loop unroll_count(qp_unroll)")
#else
#define EDGE3D_QP_UNROLL
#endif

  for (Int_type i = 0; i < quad_order; i++) {

    const Real_type xloc = qpts_1d[i];
    const Real_type tmpx = 1. - xloc;

    EDGE3D_QP_UNROLL
    for (Int_type j = 0; j < quad_order; j++) {

      const Real_type yloc = qpts_1d[j];
      const Real_type wgtxy = wgts_1d[i]*wgts_1d[j];
      const Real_type tmpy = 1. - yloc;

      const Real_type tmpxy    = tmpx*tmpy;
      const Real_type xyloc    = xloc*yloc;
      const Real_type tmpxyloc = tmpx*yloc;
      const Real_type xloctmpy = xloc*tmpy;

      const V jzx = comb(dzeta[0], tmpxy, xloctmpy, xyloc, tmpxyloc);
      const V jzy = comb(dzeta[1], tmpxy, xloctmpy, xyloc, tmpxyloc);
      const V jzz = comb(dzeta[2], tmpxy, xloctmpy, xyloc, tmpxyloc);

      EDGE3D_QP_UNROLL
      for (Int_type k = 0; k < quad_order; k++) {

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

        const V jxx = comb(dxi[0], tmpyz, yloctmpz, tmpyzloc, yzloc);
        const V jxy = comb(dxi[1], tmpyz, yloctmpz, tmpyzloc, yzloc);
        const V jxz = comb(dxi[2], tmpyz, yloctmpz, tmpyzloc, yzloc);
        const V jyx = comb(deta[0], tmpxz, xloctmpz, tmpxzloc, xzloc);
        const V jyy = comb(deta[1], tmpxz, xloctmpz, tmpxzloc, xzloc);
        const V jyz = comb(deta[2], tmpxz, xloctmpz, tmpxzloc, xzloc);

        // cofactors: J^{-1} = K / detj
        const V kxx = vfms(jyy, jzz, vmul(jyz, jzy));
        const V kxy = vfms(jxz, jzy, vmul(jxy, jzz));
        const V kxz = vfms(jxy, jyz, vmul(jxz, jyy));
        const V kyx = vfms(jyz, jzx, vmul(jyx, jzz));
        const V kyy = vfms(jxx, jzz, vmul(jxz, jzx));
        const V kyz = vfms(jxz, jyx, vmul(jxx, jyz));
        const V kzx = vfms(jyx, jzy, vmul(jyy, jzx));
        const V kzy = vfms(jxy, jzx, vmul(jxx, jzy));
        const V kzz = vfms(jxx, jyy, vmul(jxy, jyx));

        // bad_zone_algorithm
        const V detj_unfixed = vfma(jzz, kzz, vfma(jzy, kyz, vmul(jzx, kxz)));
        const V detj = vsel_lt(vabs(detj_unfixed), tol_abs_detj_cc,
                               detj_cc, detj_unfixed);
        const V abs_detj = vabs(detj);
        const V inv_detj = vdiv(vone, vadd(detj, vtiny));
        const V inv_abs_detj = vdiv(vone, vadd(abs_detj, vtiny));

        // walpha / detj^2 and wbeta / |detj|^2
        const V wdet = vmul(bc(wgts), abs_detj);
        const V sG = vmul(vmul(bc(alpha), wdet), vmul(inv_detj, inv_detj));
        const V sH = vmul(vmul(bc(beta), wdet),
                          vmul(inv_abs_detj, inv_abs_detj));

        alignas(64) V WG[6], WH[6];
        WG[0] = vmul(sG, vfma(kzx, kzx, vfma(kyx, kyx, vmul(kxx, kxx))));
        WG[1] = vmul(sG, vfma(kzx, kzy, vfma(kyx, kyy, vmul(kxx, kxy))));
        WG[2] = vmul(sG, vfma(kzx, kzz, vfma(kyx, kyz, vmul(kxx, kxz))));
        WG[3] = vmul(sG, vfma(kzy, kzy, vfma(kyy, kyy, vmul(kxy, kxy))));
        WG[4] = vmul(sG, vfma(kzy, kzz, vfma(kyy, kyz, vmul(kxy, kxz))));
        WG[5] = vmul(sG, vfma(kzz, kzz, vfma(kyz, kyz, vmul(kxz, kxz))));
        WH[0] = vmul(sH, vfma(jxz, jxz, vfma(jxy, jxy, vmul(jxx, jxx))));
        WH[1] = vmul(sH, vfma(jxz, jyz, vfma(jxy, jyy, vmul(jxx, jyx))));
        WH[2] = vmul(sH, vfma(jxz, jzz, vfma(jxy, jzy, vmul(jxx, jzx))));
        WH[3] = vmul(sH, vfma(jyz, jyz, vfma(jyy, jyy, vmul(jyx, jyx))));
        WH[4] = vmul(sH, vfma(jyz, jzz, vfma(jyy, jzy, vmul(jyx, jzx))));
        WH[5] = vmul(sH, vfma(jzz, jzz, vfma(jzy, jzy, vmul(jzx, jzx))));

        // element independent basis values (edgebasis_*, curl_edgebasis_*)
        const Real_type s[EB] = {tmpyz, yloctmpz, tmpyzloc, yzloc,
                                 tmpxz, xloctmpz, tmpxzloc, xzloc,
                                 tmpxy, xloctmpy, tmpxyloc, xyloc};
        const Real_type db[EB][3] = {
          {0.0, -tmpy,  tmpz}, {0.0, -yloc, -tmpz},
          {0.0,  tmpy,  zloc}, {0.0,  yloc, -zloc},
          { tmpx, 0.0, -tmpz}, { xloc, 0.0,  tmpz},
          {-tmpx, 0.0, -zloc}, {-xloc, 0.0,  zloc},
          {-tmpx,  tmpy, 0.0}, {-xloc, -tmpy, 0.0},
          { tmpx,  yloc, 0.0}, { xloc, -yloc, 0.0}};

        // v_m = wbeta H db_m
        alignas(64) V v[EB][3];
        RAJAPERF_FEM_UNROLL
        for (Int_type m = 0; m < EB; m++) {
          const Int_type a1 = nz1[m / 4], a2 = nz2[m / 4];
          const V b1 = bc(db[m][a1]), b2 = bc(db[m][a2]);
          RAJAPERF_FEM_UNROLL
          for (Int_type c = 0; c < 3; c++) {
            v[m][c] = vfma(WH[sym[c][a2]], b2, vmul(WH[sym[c][a1]], b1));
          }
        }

        Int_type u = 0;
        RAJAPERF_FEM_UNROLL
        for (Int_type p = 0; p < EB; p++) {
          const Int_type gp = p / 4;
          const Int_type a1 = nz1[gp], a2 = nz2[gp];
          const V sp = bc(s[p]);
          const V ap[3] = {vmul(sp, WG[sym[gp][0]]), vmul(sp, WG[sym[gp][1]]),
                           vmul(sp, WG[sym[gp][2]])};
          const V bp1 = bc(db[p][a1]), bp2 = bc(db[p][a2]);
          RAJAPERF_FEM_UNROLL
          for (Int_type m = p; m < EB; m++, u++) {
            V acc = vfma(bc(s[m]), ap[m / 4], Mu[u]);
            acc = vfma(bp1, v[m][a1], acc);
            Mu[u] = vfma(bp2, v[m][a2], acc);
          }
        }
      }
    }
  }

  // sum of the full symmetric matrix
  V sdiag = vsplat<NV>(vzero()), soff = vsplat<NV>(vzero());
  Int_type u = 0;
  for (Int_type p = 0; p < EB; p++) {
    sdiag = vadd(sdiag, Mu[u++]);
    for (Int_type m = p + 1; m < EB; m++) {
      soff = vadd(soff, Mu[u++]);
    }
  }
  const V tot = vfma(soff, bc(2.0), sdiag);
  edge3d_avx512_store_sum<NV>(sum, nel, tot);
  // testing only: upper triangle as [NU][8*NV]
  if constexpr (STORE_MATRIX) {
    for (Int_type v = 0; v < NU; v++) {
      RAJAPERF_FEM_UNROLL
      for (int k = 0; k < NV; k++) {
        vstore(matrix + v * BL + k * FEM_VL, Mu[v].r[k]);
      }
    }
  }
}
#undef EDGE3D_QP_UNROLL

//
// AVX-512 "metric_rb" (see edge_MpSmatrix_sum_rb_simd). Phase 2 tiles the
// upper triangle into 4 x MT blocks, MT = 4/NV, so a tile keeps 16 zmm
// accumulators + 8 zmm of H db_m in registers for every NV; per qp and tile
// it only loads the 5 needed metric terms and broadcasts element independent
// scalars, and nothing is stored until the tile is finished.
//
template <rajaperf::Int_type quad_order, bool STORE_MATRIX = false, int NV = 1>
RAJAPERF_FEM_NOINLINE void edge_MpSmatrix_sum_rb_avx512(
  const rajaperf::Real_type *const (&xn)[NB],
  const rajaperf::Real_type *const (&yn)[NB],
  const rajaperf::Real_type *const (&zn)[NB],
  const rajaperf::Index_type nel,
  const rajaperf::Real_type  alpha,
  const rajaperf::Real_type  beta,
  const rajaperf::Real_type  detj_tol,
  const Edge3DQpTable<quad_order> &qt,
  rajaperf::Real_type       *sum,
  rajaperf::Real_type       *matrix = nullptr)
{
  using namespace rajaperf::fem_simd;
  using rajaperf::Real_type;
  using rajaperf::Int_type;
  using rajaperf::Index_type;
  using V = vdx<NV>;
  static_assert(4 % NV == 0, "NV must be 1, 2 or 4");
  constexpr Index_type BL = FEM_VL * NV;
  constexpr Int_type NQ = Edge3DQpTable<quad_order>::NQ;
  constexpr Int_type MT = 4 / NV;
  static constexpr Int_type nz[3][2] = {{1, 2}, {0, 2}, {0, 1}};
  static constexpr Int_type sym[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
  auto bc = [](Real_type a) { return vsplat<NV>(vset1(a)); };

  alignas(64) V dxi[3][4], deta[3][4], dzeta[3][4];
  const V detj_cc = edge3d_avx512_load<NV>(xn, yn, zn, nel, dxi, deta, dzeta);
  const V tol_abs_detj_cc = vmul(bc(detj_tol), vabs(detj_cc));
  const V vtiny = bc(ptiny);
  const V vone = bc(1.0);

  // phase 1: W[q] = {WG[6], WH[6]}
  alignas(64) V W[NQ][12];
  RAJAPERF_FEM_STACK_ALIGN(W);
  for (Int_type q = 0; q < NQ; q++) {
    const Real_type (&cw)[3][4] = qt.cw[q];
    auto comb = [](const V (&d)[4], const Real_type (&w)[4]) {
      return edge3d_comb(d, w[0], w[1], w[2], w[3]);
    };
    const V jxx = comb(dxi[0], cw[0]);
    const V jxy = comb(dxi[1], cw[0]);
    const V jxz = comb(dxi[2], cw[0]);
    const V jyx = comb(deta[0], cw[1]);
    const V jyy = comb(deta[1], cw[1]);
    const V jyz = comb(deta[2], cw[1]);
    const V jzx = comb(dzeta[0], cw[2]);
    const V jzy = comb(dzeta[1], cw[2]);
    const V jzz = comb(dzeta[2], cw[2]);

    const V kxx = vfms(jyy, jzz, vmul(jyz, jzy));
    const V kxy = vfms(jxz, jzy, vmul(jxy, jzz));
    const V kxz = vfms(jxy, jyz, vmul(jxz, jyy));
    const V kyx = vfms(jyz, jzx, vmul(jyx, jzz));
    const V kyy = vfms(jxx, jzz, vmul(jxz, jzx));
    const V kyz = vfms(jxz, jyx, vmul(jxx, jyz));
    const V kzx = vfms(jyx, jzy, vmul(jyy, jzx));
    const V kzy = vfms(jxy, jzx, vmul(jxx, jzy));
    const V kzz = vfms(jxx, jyy, vmul(jxy, jyx));

    const V detj_unfixed = vfma(jzz, kzz, vfma(jzy, kyz, vmul(jzx, kxz)));
    const V detj = vsel_lt(vabs(detj_unfixed), tol_abs_detj_cc,
                           detj_cc, detj_unfixed);
    const V abs_detj = vabs(detj);
    const V inv_detj = vdiv(vone, vadd(detj, vtiny));
    const V inv_abs_detj = vdiv(vone, vadd(abs_detj, vtiny));

    const V wdet = vmul(bc(qt.wgt[q]), abs_detj);
    const V sG = vmul(vmul(bc(alpha), wdet), vmul(inv_detj, inv_detj));
    const V sH = vmul(vmul(bc(beta), wdet), vmul(inv_abs_detj, inv_abs_detj));

    W[q][0]  = vmul(sG, vfma(kzx, kzx, vfma(kyx, kyx, vmul(kxx, kxx))));
    W[q][1]  = vmul(sG, vfma(kzx, kzy, vfma(kyx, kyy, vmul(kxx, kxy))));
    W[q][2]  = vmul(sG, vfma(kzx, kzz, vfma(kyx, kyz, vmul(kxx, kxz))));
    W[q][3]  = vmul(sG, vfma(kzy, kzy, vfma(kyy, kyy, vmul(kxy, kxy))));
    W[q][4]  = vmul(sG, vfma(kzy, kzz, vfma(kyy, kyz, vmul(kxy, kxz))));
    W[q][5]  = vmul(sG, vfma(kzz, kzz, vfma(kyz, kyz, vmul(kxz, kxz))));
    W[q][6]  = vmul(sH, vfma(jxz, jxz, vfma(jxy, jxy, vmul(jxx, jxx))));
    W[q][7]  = vmul(sH, vfma(jxz, jyz, vfma(jxy, jyy, vmul(jxx, jyx))));
    W[q][8]  = vmul(sH, vfma(jxz, jzz, vfma(jxy, jzy, vmul(jxx, jzx))));
    W[q][9]  = vmul(sH, vfma(jyz, jyz, vfma(jyy, jyy, vmul(jyx, jyx))));
    W[q][10] = vmul(sH, vfma(jyz, jzz, vfma(jyy, jzy, vmul(jyx, jzx))));
    W[q][11] = vmul(sH, vfma(jzz, jzz, vfma(jzy, jzy, vmul(jzx, jzx))));
  }

  // phase 2: M[p][m] = sum_q s_m (s_p WG) + db_p . (WH db_m), tile by tile
  V sdiag = vsplat<NV>(vzero()), soff = vsplat<NV>(vzero());
  static_for<3>([&](auto GP) RAJAPERF_FEM_LAMBDA_INLINE {
    constexpr Int_type gp = decltype(GP)::value;
    static_for<3>([&](auto GM) RAJAPERF_FEM_LAMBDA_INLINE {
      constexpr Int_type gm = decltype(GM)::value;
      if constexpr (gm >= gp) {
        static_for<4 / MT>([&](auto MB) RAJAPERF_FEM_LAMBDA_INLINE {
          constexpr Int_type m0 = 4 * gm + MT * decltype(MB)::value;
          V acc[4][MT];
          RAJAPERF_FEM_UNROLL
          for (Int_type pt = 0; pt < 4; pt++) {
            RAJAPERF_FEM_UNROLL
            for (Int_type mt = 0; mt < MT; mt++) {
              acc[pt][mt] = vsplat<NV>(vzero());
            }
          }
          for (Int_type q = 0; q < NQ; q++) {
            const Real_type (&s)[EB] = qt.s[q];
            const Real_type (&db)[EB][2] = qt.db[q];
            V v[MT][2];
            RAJAPERF_FEM_UNROLL
            for (Int_type mt = 0; mt < MT; mt++) {
              const V b1 = bc(db[m0 + mt][0]), b2 = bc(db[m0 + mt][1]);
              RAJAPERF_FEM_UNROLL
              for (Int_type ci = 0; ci < 2; ci++) {
                const Int_type c = nz[gp][ci];
                v[mt][ci] = vfma(W[q][6 + sym[c][nz[gm][1]]], b2,
                                 vmul(W[q][6 + sym[c][nz[gm][0]]], b1));
              }
            }
            const V wg = W[q][sym[gp][gm]];
            RAJAPERF_FEM_UNROLL
            for (Int_type pt = 0; pt < 4; pt++) {
              const Int_type p = 4 * gp + pt;
              const V ap = vmul(bc(s[p]), wg);
              const V bp1 = bc(db[p][0]), bp2 = bc(db[p][1]);
              RAJAPERF_FEM_UNROLL
              for (Int_type mt = 0; mt < MT; mt++) {
                if (m0 + mt >= p) {
                  V a = vfma(bc(s[m0 + mt]), ap, acc[pt][mt]);
                  a = vfma(bp1, v[mt][0], a);
                  acc[pt][mt] = vfma(bp2, v[mt][1], a);
                }
              }
            }
          }
          RAJAPERF_FEM_UNROLL
          for (Int_type pt = 0; pt < 4; pt++) {
            RAJAPERF_FEM_UNROLL
            for (Int_type mt = 0; mt < MT; mt++) {
              const Int_type p = 4 * gp + pt, m = m0 + mt;
              if (m > p) {
                soff = vadd(soff, acc[pt][mt]);
              } else if (m == p) {
                sdiag = vadd(sdiag, acc[pt][mt]);
              }
              if constexpr (STORE_MATRIX) {
                if (m >= p) {
                  RAJAPERF_FEM_UNROLL
                  for (int k = 0; k < NV; k++) {
                    vstore(matrix + edge3d_upper_index(p, m) * BL + k * FEM_VL,
                           acc[pt][mt].r[k]);
                  }
                }
              }
            }
          }
        });
      }
    });
  });

  edge3d_avx512_store_sum<NV>(sum, nel, vfma(soff, bc(2.0), sdiag));
}

#define EDGE3D_METRIC_FN edge_MpSmatrix_sum_avx512
#define EDGE3D_METRIC_RB_FN edge_MpSmatrix_sum_rb_avx512
#else
#define EDGE3D_METRIC_FN edge_MpSmatrix_sum_simd
#define EDGE3D_METRIC_RB_FN edge_MpSmatrix_sum_rb_simd
#endif

// "metric" tunings: batch [i0, min(i0+BL, iend)) with the metric-tensor
// reformulation; RAJAPerf's FLOP count still describes the reference algorithm.
#define EDGE3D_METRIC_BATCH(BL, QUAD, ...) \
  {\
    const rajaperf::Index_type nel_ = \
      (iend - i0 < (BL)) ? (iend - i0) : (BL);\
    const rajaperf::Real_type *const xn_[NB] = \
      {x0+i0, x1+i0, x2+i0, x3+i0, x4+i0, x5+i0, x6+i0, x7+i0};\
    const rajaperf::Real_type *const yn_[NB] = \
      {y0+i0, y1+i0, y2+i0, y3+i0, y4+i0, y5+i0, y6+i0, y7+i0};\
    const rajaperf::Real_type *const zn_[NB] = \
      {z0+i0, z1+i0, z2+i0, z3+i0, z4+i0, z5+i0, z6+i0, z7+i0};\
    __VA_ARGS__(xn_, yn_, zn_, nel_, 1.0, 1.0, 0.0, QUAD, sum + i0);\
  }

#define EDGE3D_METRIC_BODY \
  EDGE3D_METRIC_BATCH(EDGE3D_VL, 1, EDGE3D_METRIC_FN<NQ_1D>)

#if defined(RAJAPERF_FEM_AVX512)
// NV zmm per value: EDGE3D_VL*NV elements per batch, NV-way ILP
#define EDGE3D_METRIC_BODY_NV(NV) \
  EDGE3D_METRIC_BATCH(EDGE3D_VL*(NV), 1, \
                      edge_MpSmatrix_sum_avx512<NQ_1D, false, NV>)
#endif

// register-blocked metric; needs `const Edge3DQpTable<NQ_1D> edge3d_qpt(1);`
#define EDGE3D_METRIC_RB_BODY(NV) \
  EDGE3D_METRIC_BATCH(EDGE3D_VL*(NV), edge3d_qpt, \
                      EDGE3D_METRIC_RB_FN<NQ_1D, false, NV>)

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
  template < size_t tune_idx >
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
