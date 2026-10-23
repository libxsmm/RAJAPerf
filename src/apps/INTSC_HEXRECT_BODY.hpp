//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJAPerf_Apps_INTSC_HEXRECT_BODY_HPP
#define RAJAPerf_Apps_INTSC_HEXRECT_BODY_HPP

#include "FEM_SIMD.hpp"

namespace rajaperf {

// Helper functions for the INTSC_HEXRECT kernel

//   Clip a polygon returning polygon with active coordinate ain >= cut
//     Return number of vertices after clipping (might be zero).
//
RAJA_HOST_DEVICE
RAJA_INLINE Int_type clip_polygon_ge
    ( Real_ptr ain,              // input active coordinates
      Real_ptr bin, Real_ptr cin,  // input passive coordinates
      Bool_type const etob,   // Whether to clip from end to begin of ain
      Real_type const cut,         // cut value of active coordinate
      Int_type nv_in )       // number of sides in
{
  Int_type const max = max_polygon_pts ;   // max_polygon_pts = 10
  Int_type j, jbeg, jend, jr, inc ;

  if ( etob ) {    // source at end of ain, destination at beginning
    jbeg = max - nv_in ;    jend = max ;
    j    = max - 1 ;         inc = 1 ;          jr = 0 ;
  } else {         // source at beginning of ain, destination at end
    jbeg = nv_in - 1 ;      jend = -1 ;
    j    = 0 ;               inc = -1 ;         jr = max - 1 ;
  }

  //  Ensure that if bin[j] == bin[jj] we get exactly bin[j]
  //   from interpolation.  This prevents roundoff fluctuations
  //   from being introduced from interpolation.
  //
  //  Also ensure clip_polygon_ge and _lt have positive den and
  //  eta in all four cases, ensuring consistency when _ge and _lt
  //  are called for the same source polygon (ain), so that the
  //  _ge and _lt output are correctly complementary to each other.
  //
  for ( Index_type jj = jbeg ; jj != jend ; jj += inc ) {
    if ( ain[j] >= cut ) {
      ain[jr] = ain[j] ; bin[jr] = bin[j] ; cin[jr] = cin[j] ;
      jr += inc ;
      if ( ain[jj] < cut ) {
        Real_type den =  ain[j] - ain[jj] ;
        Real_type eta = ( cut - ain[jj] ) / den ;
        ain[jr] = cut ;
        bin[jr] = bin[jj] + ( bin[j] - bin[jj] ) * eta ;
        cin[jr] = cin[jj] + ( cin[j] - cin[jj] ) * eta ;
        jr += inc ;
      }
    } else if ( ain[jj] >= cut ) {
      Real_type den = ain[jj] - ain[j] ;
      Real_type eta = ( cut - ain[j] ) / den ;
      ain[jr] = cut ;
      bin[jr] = bin[j] + ( bin[jj] - bin[j] ) * eta ;
      cin[jr] = cin[j] + ( cin[jj] - cin[j] ) * eta ;
      jr += inc ;
    }
    j = jj ;
  }
  Int_type ret = ( etob ) ? jr : ( max - 1 - jr ) ;
  return ret ;
}



//   Clip a polygon returning polygon with active coordinate ain < cut
//   etob is always false here (source is at beginning of ain,
//   destination is at end), hence removed etob parameter.
//     Return number of vertices after clipping (might be zero).
//
RAJA_HOST_DEVICE
RAJA_INLINE Int_type clip_polygon_lt
    ( Real_ptr ain,              // input active coordinates
      Real_ptr bin, Real_ptr cin,  // input passive coordinates
      Real_type const cut,         // cut value of active coordinate
      Int_type nv_in )       // number of sides in
{
  //     See comments in clip_polygon_ge.
  Int_type const max = max_polygon_pts ;   // max_polygon_pts = 10
  Int_type j, jbeg, jend, jr, inc ;

  // etob is false : source at beginning of ain, destination at end
  jbeg = nv_in - 1 ;      jend = -1 ;
  j    = 0 ;               inc = -1 ;         jr = max - 1 ;

  for ( Index_type jj = jbeg ; jj != jend ; jj += inc ) {
    if ( ain[j] < cut ) {
      ain[jr] = ain[j] ; bin[jr] = bin[j] ; cin[jr] = cin[j] ;
      jr += inc ;
      if ( ain[jj] >= cut ) {
        Real_type den =  ain[jj] - ain[j] ;
        Real_type eta = ( cut - ain[j] ) / den ;
        ain[jr] = cut ;
        bin[jr] = bin[j] + ( bin[jj] - bin[j] ) * eta ;
        cin[jr] = cin[j] + ( cin[jj] - cin[j] ) * eta ;
        jr += inc ;
      }
    } else if ( ain[jj] < cut ) {
      Real_type den = ain[j] - ain[jj] ;
      Real_type eta = ( cut - ain[jj] ) / den ;
      ain[jr] = cut ;
      bin[jr] = bin[jj] + ( bin[j] - bin[jj] ) * eta ;
      cin[jr] = cin[jj] + ( cin[j] - cin[jj] ) * eta ;
      jr += inc ;
    }
    j = jj ;
  }
  Int_type ret = max - 1 - jr ;   // ret = number of vertices after clipping
  return ret ;
}



//   Return the vertex count tally.
//
//   3D : metric factor is z values of donor polygon
//
RAJA_HOST_DEVICE
RAJA_INLINE Int_type intsc24_shxf1
      ( Real_type const dtx,    // target zone x length
        Real_type const dty,    // target zone y length
        Real_type const x0,     // target zone lower x coordinate
        Real_type const y0,     // target zone lower y coordinate
        Real_type const z0,     // target zone lower z coordinate, for moment
        Real_ptr qx,          // clipped donor polygon (circular order)
        Int_type const shn,       // Number of vertices in donor polygon max 5
        Real_type & sum0,       // output area or volume
        Real_type & sumx,       // output x moment
        Real_type & sumy,       // output y moment
        Real_type & sumz )      // output z moment
{
  Real_type const one3   = 0.33333333333333333 ;

  Real_const_ptr qy = qx + max_polygon_pts ;  //  max_polygon_pts is 10
  Real_const_ptr qz = qy + max_polygon_pts ;

  Real_type xc0 = qx[0], xc1 = qx[1] ;
  Real_type yc0 = qy[0], yc1 = qy[1] ;
  Real_type zc0 = qz[0], zc1 = qz[1] ;
  for ( Index_type kk = 2 ; kk < shn ; kk++ ) {

    Real_type xc2 = qx[kk] ;
    Real_type yc2 = qy[kk] ;
    Real_type zc2 = qz[kk] ;

    Real_type s0tri = 0.0, sxtri = 0.0, sytri = 0.0, sztri = 0.0 ;
    Real_type metfac ;

    // Edge Midpoint quadrature each triangle
    //  For the z moment we have an additional 0.5 factor because
    //  z ranges from 0 to the plane of the polygon.
    metfac = 0.5 * ( zc0 + zc1 ) ;
    s0tri += metfac ;
    sxtri += metfac * ( x0 + 0.5 * dtx * ( xc0 + xc1 ) ) ;
    sytri += metfac * ( y0 + 0.5 * dty * ( yc0 + yc1 ) ) ;
    sztri += metfac * ( z0 + 0.5 * metfac ) ;

    metfac = 0.5 * ( zc1 + zc2 ) ;
    s0tri += metfac ;
    sxtri += metfac * ( x0 + 0.5 * dtx * ( xc1 + xc2 ) ) ;
    sytri += metfac * ( y0 + 0.5 * dty * ( yc1 + yc2 ) ) ;
    sztri += metfac * ( z0 + 0.5 * metfac ) ;

    metfac = 0.5 * ( zc2 + zc0 ) ;
    s0tri += metfac ;
    sxtri += metfac * ( x0 + 0.5 * dtx * ( xc2 + xc0 ) ) ;
    sytri += metfac * ( y0 + 0.5 * dty * ( yc2 + yc0 ) ) ;
    sztri += metfac * ( z0 + 0.5 * metfac ) ;

    //   area is positive for counterclockwise triangle (standard)
    Real_type area = 0.5 *
        ( (xc0-xc1) * (yc0+yc1) + (xc1-xc2) * (yc1+yc2) +
          (xc2-xc0) * (yc2+yc0) ) ;

    sum0 += one3 * area * s0tri ;
    sumx += one3 * area * sxtri ;
    sumy += one3 * area * sytri ;
    sumz += one3 * area * sztri ;

    xc1 = xc2 ;         // advance to next edge of clipped polygon
    yc1 = yc2 ;
    zc1 = zc2 ;
  }
  return shn ;
}


//   Convert a 24 bit mask of predicates to a list where the bit is set.
//       One bit per facet of the triangle.
//
//   Compactifying mask to list might reduce branch divergence.
//
RAJA_HOST_DEVICE
RAJA_INLINE Int_type intsc24_hex_mask_to_list
    ( Int_type const mask,            // mask (used bits 0 to 23)
      Uchar_type (&mylist)[24] ) //  list of which mask bits are set
{
  Int_type count=0 ;
  for ( Index_type bit = 0 ; bit < 24 ; ++bit ) {
    if ( 0 != ( mask & (1<<bit) ) ) { mylist[count++] = bit ; }
  }
  return count ;
}



//   Get a triangle facet of the hexahedron,
//   transformed to frame of Cartesian target zone is 0<=(x,y)<=1 and
//   minimum z is 0.
//
RAJA_HOST_DEVICE
RAJA_INLINE void intsc24_hex_get_tri
    ( Real_const_ptr xd,    // [8] donor x coordinates
      Real_const_ptr yd,    // [8] donor y coordinates
      Real_const_ptr zd,    // [8] donor z coordinates
      Real_type const xt0,    // target zone lower x boundary
      Real_type const yt0,    // target zone lower y boundary
      Real_type const zt0,    // target zone lower z boundary
      Real_type const a11,    // x multiplier
      Real_type const a22,    // y multiplier
      Int_type const f,         // which of six faces
      Int_type const k0,        // which of four facets of face
      Real_ptr xf,            // transformed facet x
      Real_ptr yf,            // transformed facet y
      Real_ptr zf )           // transformed facet z
{
  Int_type v0, v1, v2, v3 ;

  switch (f) {
  case 0:  v0 = 0 ; v1 = 2 ; v2 = 6 ; v3 = 4 ; break ;
  case 1:  v0 = 1 ; v1 = 5 ; v2 = 7 ; v3 = 3 ; break ;
  case 2:  v0 = 0 ; v1 = 4 ; v2 = 5 ; v3 = 1 ; break ;
  case 3:  v0 = 2 ; v1 = 3 ; v2 = 7 ; v3 = 6 ; break ;
  case 4:  v0 = 0 ; v1 = 1 ; v2 = 3 ; v3 = 2 ; break ;
  default: v0 = 4 ; v1 = 6 ; v2 = 7 ; v3 = 5 ; break ;
  }

  //   Face center is triangle point 1.
  //   Transformation of face center to the unit target frame 0<(x,y)<1.
  xf[1] = ( 0.25 * (xd[v0] + xd[v1] + xd[v2] + xd[v3]) - xt0 ) * a11 ;
  yf[1] = ( 0.25 * (yd[v0] + yd[v1] + yd[v2] + yd[v3]) - yt0 ) * a22 ;
  zf[1] =   0.25 * (zd[v0] + zd[v1] + zd[v2] + zd[v3]) - zt0 ;

  // triangle vertices
  Int_type vv0, vv2 ;

  switch (k0) {
  case 0:   vv0 = v0 ;   vv2 = v1 ;  break ;
  case 1:   vv0 = v1 ;   vv2 = v2 ;  break ;
  case 2:   vv0 = v2 ;   vv2 = v3 ;  break ;
  default:  vv0 = v3 ;   vv2 = v0 ;  break ;
  }
  //   Coordinates of a triangle facet of the donor hexahedron.
  xf[0] = a11 * ( xd[vv0] - xt0 ) ;   xf[2] = a11 * ( xd[vv2] - xt0 ) ;
  yf[0] = a22 * ( yd[vv0] - yt0 ) ;   yf[2] = a22 * ( yd[vv2] - yt0 ) ;
  zf[0] =         zd[vv0] - zt0   ;   zf[2] =         zd[vv2] - zt0   ;
}




//   Categorize the triangles.
//      1) interior         0<=x,y<=1,  0<=z<=dzt
//      2) above z=dzt, and 0<=x,y<=1
//      3) needs clip
//      4) outside
RAJA_HOST_DEVICE
RAJA_INLINE Int_type intsc24_hex_filter
    ( Real_const_ptr xd,    // [8] donor x coordinates
      Real_const_ptr yd,    // [8] donor y coordinates
      Real_const_ptr zd,    // [8] donor z coordinates
      Real_type const xt0,    // target zone lower x boundary
      Real_type const xt1,    // target zone upper x boundary
      Real_type const yt0,    // target zone lower y boundary
      Real_type const yt1,    // target zone upper y boundary
      Real_type const zt0,    // target zone lower z boundary
      Real_type const zt1,    // target zone upper z boundary
      Int_type &inside,
      Int_type &abovez,
      Int_type &clip )          // clip - 12 out of 24 triangles is common
{
  inside = abovez = clip = 0 ;   // initialize masks.

  //  target z spacing
  Real_type const dzt = ( zt1 > zt0 ) ? ( zt1 - zt0 ) : 0.0 ;

  Real_type a11, a22 ;    // Transformation to unit target frame 0<(x,y)<1
  {

    Real_type dtx = xt1 - xt0, dty = yt1 - yt0 ;
    Real_type det  = dtx * dty ;    // area of the target zone (determinant)

    //   Transformation to the unit target frame 0<(x,y)<1.
    Real_type deti = det / (det*det + 1.0e-80);

    a11 =  dty * deti;
    a22 =  dtx * deti;
  }

  Int_type count = 0 ;
  for ( Index_type f = 0 ; f < 6 ; ++f ) {   // six faces of the hexahedron

    for ( Index_type k0 = 0 ; k0 < 4 ; ++k0 ) {   // four triangles of the face

      // transformed facet coordinates
      Real_type xf[3], yf[3], zf[3] ;

      intsc24_hex_get_tri
          ( xd, yd, zd, xt0, yt0, zt0, a11, a22, f, k0, xf, yf, zf ) ;

      //  Rule out facet if it is definitely outside a plane.
      if ( ! ( ( ( xf[0] <  0.0 ) && ( xf[1] <  0.0 ) && ( xf[2] <  0.0 ) ) ||
               ( ( xf[0] >= 1.0 ) && ( xf[1] >= 1.0 ) && ( xf[2] >= 1.0 ) ) ||
               ( ( yf[0] <  0.0 ) && ( yf[1] <  0.0 ) && ( yf[2] <  0.0 ) ) ||
               ( ( yf[0] >= 1.0 ) && ( yf[1] >= 1.0 ) && ( yf[2] >= 1.0 ) ) ||
               ( ( zf[0] <  0.0 ) && ( zf[1] <  0.0 ) && ( zf[2] <  0.0 ) ) )) {

        Int_type mask = 1 << count ;

        // test whether interior to x and y ranges
        if  ( ! (( xf[0] >= 0.0 ) && ( xf[1] >= 0.0 ) && ( xf[2] >= 0.0 ) &&
                 ( xf[0] <  1.0 ) && ( xf[1] <  1.0 ) && ( xf[2] <  1.0 ) &&
                 ( yf[0] >= 0.0 ) && ( yf[1] >= 0.0 ) && ( yf[2] >= 0.0 ) &&
                 ( yf[0] <  1.0 ) && ( yf[1] <  1.0 ) && ( yf[2] <  1.0 )) ) {

          clip |= mask ;   // Not interior, will run clip on this facet
        } else {

          //  Facet is interior to x and y ranges.

          if ( ( zf[0] >= dzt ) && ( zf[1] >= dzt ) && ( zf[2] >= dzt ) ) {

            abovez |= mask ;   // facet above dzt
          } else {

            if ( ( zf[0] >= 0.0 ) && ( zf[1] >= 0.0 ) && ( zf[2] >= 0.0 ) &&
                 ( zf[0] <  dzt ) && ( zf[1] <  dzt ) && ( zf[2] <  dzt ) ) {

              inside |= mask ;
            } else {
              clip |= mask ;    // Not interior, will run clip on facet.
            }
          }
        }
      }
      ++count ;
    }
  }
  return 0 ;
}





//  Map a hexahedral donor zone (24 triangular facets) onto
//  a Cartesian zone.
//
//
RAJA_HOST_DEVICE
RAJA_INLINE Int_type intsc24_hex
      ( Real_ptr xd,    // [24] donor x coordinates, workspace
        Real_ptr qx_work,   // [3*max_polygon_pts] workspace for polygons
        Real_type const xt0,    // target zone lower x boundary
        Real_type const xt1,    // target zone upper x boundary
        Real_type const yt0,    // target zone lower y boundary
        Real_type const yt1,    // target zone upper y boundary
        Real_type const zt0,    // target zone lower z boundary
        Real_type const zt1,    // target zone upper z boundary
        Real_type & sum0,       // output area or volume
        Real_type & sumx,       // output x moment
        Real_type & sumy,       // output y moment
        Real_type & sumz )      // output z moment
{
  sum0 = sumx = sumy = sumz = 0.0 ;
  Int_type vtxcnt = 0 ;

  //  Points oriented so that (v0, face center, v2) points out of positive zone
  //  static Int_type const vface[24] =
  //  {0, 2, 6, 4, 1, 5, 7, 3, 0, 4, 5, 1, 2, 3, 7, 6, 0, 1, 3, 2, 4, 6, 7, 5} ;

  // polygon for overlay
  Real_ptr yd = xd + 8 ;
  Real_ptr zd = yd + 8 ;

  Real_ptr rx = qx_work ;
  Real_ptr ry = rx + max_polygon_pts ;    // max_polygon_pts is 10
  Real_ptr rz = ry + max_polygon_pts ;

  //  target z spacing
  Real_type const dzt = ( zt1 > zt0 ) ? ( zt1 - zt0 ) : 0.0 ;

  Real_type a11, a22 ;    // Transformation to unit target frame 0<(x,y)<1
  {

    Real_type dtx = xt1 - xt0, dty = yt1 - yt0 ;
    Real_type det  = dtx * dty ;    // area of the target zone (determinant)

    //   Transformation to the unit target frame 0<(x,y)<1.
    Real_type deti = det / (det*det + 1.0e-80);

    a11 =  dty * deti;
    a22 =  dtx * deti;
  }

  Int_type inside, abovez, clip ;   // which facets interior, above, or clipped.

  intsc24_hex_filter
      ( xd, yd, zd, xt0, xt1, yt0, yt1, zt0, zt1, inside, abovez, clip ) ;

  Uchar_type facet_list[24] ;
  Int_type nclip = intsc24_hex_mask_to_list ( clip, facet_list ) ;

  for ( Index_type fi = 0 ; fi < nclip ; fi++ ) {  // facet index in facet_list

    Int_type f  = facet_list[fi] >> 2 ;        //  which face is facet/4
    Int_type k0 = facet_list[fi] & 3 ;         //  which facet within face

    intsc24_hex_get_tri
        ( xd, yd, zd, xt0, yt0, zt0, a11, a22, f, k0, rx, ry, rz ) ;

    //   Clip on y=1
    Int_type shn1 = clip_polygon_lt
        ( ry, rx, rz, 1.0, 3 ) ;

    //   Clip on y=0, the x axis.
    Int_type shn2 = clip_polygon_ge
        ( ry, rx, rz, true, 0.0, shn1 ) ;

    //   Clip on x=1
    Int_type shn3 = clip_polygon_lt
        ( rx, ry, rz, 1.0, shn2 ) ;

    //   Clip on x=0, the y axis.
    Int_type shn4 = clip_polygon_ge
        ( rx, ry, rz, true, 0.0, shn3 ) ;

    Real_type sx[24] ;
    Real_ptr sy = sx + 8 ;
    Real_ptr sz = sy + 8 ;
    for ( Index_type jj = 0 ; jj < shn4 ; ++jj ) {
      sx[jj] = rx[jj] ;   sy[jj] = ry[jj] ;  sz[jj] = rz[jj] ;
    }

    //  Upper polygon - clip above z=dzt
    Int_type shn = clip_polygon_ge
        ( rz, rx, ry, false, dzt, shn4 ) ;

    if ( shn >= 3 ) {     //  There is an upper polygon

      for ( Index_type jj = 0 ; jj < shn ; ++jj ) {
        rz[jj + max_polygon_pts - shn] = dzt ;   // project to upper face
      }

      Real_type asum0 = 0.0, asumx = 0.0, asumy = 0.0, asumz = 0.0 ;
      Real_type dtx = xt1 - xt0, dty = yt1 - yt0 ;

      vtxcnt += intsc24_shxf1
          ( dtx, dty, xt0, yt0, zt0, rx + max_polygon_pts - shn, shn,
            asum0, asumx, asumy, asumz ) ;

      Real_type det  = dtx * dty ;    // area of the target zone (determinant)

      sum0 += asum0 * det ;
      sumx += asumx * det ;
      sumy += asumy * det ;
      sumz += asumz * det ;
    }

    for ( Index_type jj = 0 ; jj < shn4 ; ++jj ) {
      rx[jj] = sx[jj] ;   ry[jj] = sy[jj] ;  rz[jj] = sz[jj] ;
    }

    // Central polygon - clip below z=dzt
    shn1 = clip_polygon_lt
        ( rz, rx, ry, dzt, shn4 ) ;

    //  Central polygon - clip above z=0
    shn = clip_polygon_ge
        ( rz, rx, ry, true, 0.0, shn1 ) ;

    if ( shn >= 3 ) {     //  There is an intersecting polygon

      Real_type asum0 = 0.0, asumx = 0.0, asumy = 0.0, asumz = 0.0 ;

      Real_type dtx = xt1 - xt0, dty = yt1 - yt0 ;

      vtxcnt += intsc24_shxf1
          ( dtx, dty, xt0, yt0, zt0, rx, shn,
            asum0, asumx, asumy, asumz ) ;

      Real_type det  = dtx * dty ;    // area of the target zone (determinant)

      sum0 += asum0 * det ;
      sumx += asumx * det ;
      sumy += asumy * det ;
      sumz += asumz * det ;
    }
  }

  // abovez (contribution from facet above the target zone).
  nclip = intsc24_hex_mask_to_list ( abovez, facet_list ) ;

  for ( Index_type fi = 0 ; fi < nclip ; fi++ ) {  // facet index in facet_list

    Int_type f  = facet_list[fi] >> 2 ;        //  which face is facet/4
    Int_type k0 = facet_list[fi] & 3 ;         //  which facet within face

    intsc24_hex_get_tri
        ( xd, yd, zd, xt0, yt0, zt0, a11, a22, f, k0, rx, ry, rz ) ;

    rz[0] = rz[1] = rz[2] = dzt ;     // project Z to upper face.

    Int_type shn = 3 ;

    Real_type asum0 = 0.0, asumx = 0.0, asumy = 0.0, asumz = 0.0 ;
    Real_type dtx = xt1 - xt0, dty = yt1 - yt0 ;

    vtxcnt += intsc24_shxf1
        ( dtx, dty, xt0, yt0, zt0, rx, shn,
          asum0, asumx, asumy, asumz ) ;

    Real_type det  = dtx * dty ;    // area of the target zone (determinant)

    sum0 += asum0 * det ;
    sumx += asumx * det ;
    sumy += asumy * det ;
    sumz += asumz * det ;
  }

  // inside (contribution from facets inside the target zone)
  nclip = intsc24_hex_mask_to_list ( inside, facet_list ) ;

  for ( Index_type fi = 0 ; fi < nclip ; fi++ ) {  // facet index in facet_list

    Int_type f  = facet_list[fi] >> 2 ;        //  which face is facet/4
    Int_type k0 = facet_list[fi] & 3 ;         //  which facet within face

    intsc24_hex_get_tri
        ( xd, yd, zd, xt0, yt0, zt0, a11, a22, f, k0, rx, ry, rz ) ;

    Int_type shn = 3 ;

    Real_type asum0 = 0.0, asumx = 0.0, asumy = 0.0, asumz = 0.0 ;
    Real_type dtx = xt1 - xt0, dty = yt1 - yt0 ;

    vtxcnt += intsc24_shxf1
        ( dtx, dty, xt0, yt0, zt0, rx, shn,
          asum0, asumx, asumy, asumz ) ;

    Real_type det  = dtx * dty ;    // area of the target zone (determinant)

    sum0 += asum0 * det ;
    sumx += asumx * det ;
    sumy += asumy * det ;
    sumz += asumz * det ;
  }

  return vtxcnt ;
}


//  ---------------------------------------------------------------------------
//  Plane-splitting variant: one donor hexahedron against its 2x2x2 block of
//  target zones in one pass.  Each facet is split once by the shared x, y
//  and z planes (instead of being clipped against six planes per target),
//  and the pieces are integrated exactly as in intsc24_hex.
//  ---------------------------------------------------------------------------

struct IntscPoly
{
  static constexpr Int_type cap = 16 ;   // triangle clipped by <= 8 planes
  Real_type c[3][cap] ;                  // x, y, z coordinates
  Int_type n = 0 ;
} ;

//   Split polygon p by plane c[axis] = cut into lo (< cut) and hi (>= cut),
//   using one shared intersection vertex for both halves.
RAJA_INLINE void intsc_split_polygon
    ( IntscPoly const& p, Int_type const axis, Real_type const cut,
      IntscPoly& lo, IntscPoly& hi )
{
  lo.n = hi.n = 0 ;
  for ( Int_type k = 0 ; k < p.n ; ++k ) {
    Int_type const j = ( k + 1 == p.n ) ? 0 : k + 1 ;
    Real_type const a = p.c[axis][k], b = p.c[axis][j] ;
    bool const alo = a < cut, blo = b < cut ;
    IntscPoly& side = alo ? lo : hi ;
    for ( Int_type d = 0 ; d < 3 ; ++d ) { side.c[d][side.n] = p.c[d][k] ; }
    ++side.n ;
    if ( alo != blo ) {
      Real_type const eta = ( cut - a ) / ( b - a ) ;
      for ( Int_type d = 0 ; d < 3 ; ++d ) {
        Real_type const v = ( d == axis ) ? cut
                          : p.c[d][k] + ( p.c[d][j] - p.c[d][k] ) * eta ;
        lo.c[d][lo.n] = v ;  hi.c[d][hi.n] = v ;
      }
      ++lo.n ;  ++hi.n ;
    }
  }
}

//   Keep only the part of p with c[axis] >= cut (KEEP_HI) or < cut.
template < bool KEEP_HI >
RAJA_INLINE void intsc_clip_polygon
    ( IntscPoly& p, Int_type const axis, Real_type const cut )
{
  IntscPoly lo, hi ;
  intsc_split_polygon( p, axis, cut, lo, hi ) ;
  p = KEEP_HI ? hi : lo ;
}

//   Split p like intsc_split_polygon, but pass p through uncopied if it lies
//   on one side; lo/hi point to p, to slo/shi, or are null when empty.
RAJA_INLINE void intsc_split_or_pass
    ( IntscPoly const& p, Int_type const axis, Real_type const cut,
      IntscPoly& slo, IntscPoly& shi,
      IntscPoly const*& lo, IntscPoly const*& hi )
{
  lo = hi = nullptr ;
  if ( p.n < 3 ) { return ; }
  Real_type mn = p.c[axis][0], mx = mn ;
  for ( Int_type k = 1 ; k < p.n ; ++k ) {
    mn = ( p.c[axis][k] < mn ) ? p.c[axis][k] : mn ;
    mx = ( p.c[axis][k] > mx ) ? p.c[axis][k] : mx ;
  }
  if ( mx < cut )  { lo = &p ; return ; }
  if ( mn >= cut ) { hi = &p ; return ; }
  intsc_split_polygon( p, axis, cut, slo, shi ) ;
  lo = ( slo.n >= 3 ) ? &slo : nullptr ;
  hi = ( shi.n >= 3 ) ? &shi : nullptr ;
}

//   The part of p with c[axis] >= cut (KEEP_HI) or < cut, written to out with
//   the vertices of the corresponding intsc_split_polygon side; only the
//   valid vertices of the kept side are stored.
template < bool KEEP_HI >
RAJA_INLINE void intsc_clip_to
    ( IntscPoly const& p, Int_type const axis, Real_type const cut,
      IntscPoly& out )
{
  Int_type n = 0 ;
  for ( Int_type k = 0 ; k < p.n ; ++k ) {
    Int_type const j = ( k + 1 == p.n ) ? 0 : k + 1 ;
    Real_type const a = p.c[axis][k], b = p.c[axis][j] ;
    bool const alo = a < cut, blo = b < cut ;
    if ( alo != KEEP_HI ) {
      for ( Int_type d = 0 ; d < 3 ; ++d ) { out.c[d][n] = p.c[d][k] ; }
      ++n ;
    }
    if ( alo != blo ) {
      Real_type const eta = ( cut - a ) / ( b - a ) ;
      for ( Int_type d = 0 ; d < 3 ; ++d ) {
        out.c[d][n] = ( d == axis ) ? cut
                    : p.c[d][k] + ( p.c[d][j] - p.c[d][k] ) * eta ;
      }
      ++n ;
    }
  }
  out.n = n ;
}

//   One side of intsc_split_or_pass: p, out or null.
template < bool KEEP_HI >
RAJA_INLINE IntscPoly const* intsc_clip_or_pass
    ( IntscPoly const& p, Int_type const axis, Real_type const cut,
      IntscPoly& out )
{
  if ( p.n < 3 ) { return nullptr ; }
  Real_type mn = p.c[axis][0], mx = mn ;
  for ( Int_type k = 1 ; k < p.n ; ++k ) {
    mn = ( p.c[axis][k] < mn ) ? p.c[axis][k] : mn ;
    mx = ( p.c[axis][k] > mx ) ? p.c[axis][k] : mx ;
  }
  if ( mx < cut )  { return KEEP_HI ? nullptr : &p ; }
  if ( mn >= cut ) { return KEEP_HI ? &p : nullptr ; }
  intsc_clip_to<KEEP_HI>( p, axis, cut, out ) ;
  return ( out.n >= 3 ) ? &out : nullptr ;
}

//   Add the column integral of polygon p (already inside the target's xy
//   extent) over the target frame to the sums; heights are z - zt0, or dzt
//   for a polygon projected to the target's top (PROJECT).
//   LEAN: the frame transform is applied while integrating (same arithmetic
//   as intsc24_shxf1) instead of staging the polygon in a q[] buffer.
template < bool PROJECT, bool LEAN = false >
RAJA_INLINE void intsc_add_polygon
    ( IntscPoly const& p,
      Real_type const xt0, Real_type const dtx, Real_type const a11,
      Real_type const yt0, Real_type const dty, Real_type const a22,
      Real_type const zt0, Real_type const dzt,
      Real_type (&sums)[4] )
{
  if ( p.n < 3 ) { return ; }
  if constexpr ( LEAN ) {
    Real_type const one3 = 0.33333333333333333 ;
    auto tx = [&] ( Int_type k ) { return ( p.c[0][k] - xt0 ) * a11 ; } ;
    auto ty = [&] ( Int_type k ) { return ( p.c[1][k] - yt0 ) * a22 ; } ;
    auto tz = [&] ( Int_type k ) { return PROJECT ? dzt : p.c[2][k] - zt0 ; } ;
    Real_type asum0 = 0.0, asumx = 0.0, asumy = 0.0, asumz = 0.0 ;
    Real_type const xc0 = tx(0), yc0 = ty(0), zc0 = tz(0) ;
    Real_type xc1 = tx(1), yc1 = ty(1), zc1 = tz(1) ;
    for ( Int_type kk = 2 ; kk < p.n ; kk++ ) {
      Real_type const xc2 = tx(kk), yc2 = ty(kk), zc2 = tz(kk) ;
      Real_type s0tri = 0.0, sxtri = 0.0, sytri = 0.0, sztri = 0.0 ;
      Real_type metfac ;
      metfac = 0.5 * ( zc0 + zc1 ) ;
      s0tri += metfac ;
      sxtri += metfac * ( xt0 + 0.5 * dtx * ( xc0 + xc1 ) ) ;
      sytri += metfac * ( yt0 + 0.5 * dty * ( yc0 + yc1 ) ) ;
      sztri += metfac * ( zt0 + 0.5 * metfac ) ;
      metfac = 0.5 * ( zc1 + zc2 ) ;
      s0tri += metfac ;
      sxtri += metfac * ( xt0 + 0.5 * dtx * ( xc1 + xc2 ) ) ;
      sytri += metfac * ( yt0 + 0.5 * dty * ( yc1 + yc2 ) ) ;
      sztri += metfac * ( zt0 + 0.5 * metfac ) ;
      metfac = 0.5 * ( zc2 + zc0 ) ;
      s0tri += metfac ;
      sxtri += metfac * ( xt0 + 0.5 * dtx * ( xc2 + xc0 ) ) ;
      sytri += metfac * ( yt0 + 0.5 * dty * ( yc2 + yc0 ) ) ;
      sztri += metfac * ( zt0 + 0.5 * metfac ) ;
      Real_type const area = 0.5 *
          ( (xc0-xc1) * (yc0+yc1) + (xc1-xc2) * (yc1+yc2) +
            (xc2-xc0) * (yc2+yc0) ) ;
      asum0 += one3 * area * s0tri ;
      asumx += one3 * area * sxtri ;
      asumy += one3 * area * sytri ;
      asumz += one3 * area * sztri ;
      xc1 = xc2 ;  yc1 = yc2 ;  zc1 = zc2 ;
    }
    Real_type const det = dtx * dty ;
    sums[0] += asum0 * det ;
    sums[1] += asumx * det ;
    sums[2] += asumy * det ;
    sums[3] += asumz * det ;
    return ;
  }
  Real_type q[ 3*max_polygon_pts ] ;
  for ( Int_type k = 0 ; k < p.n ; ++k ) {
    q[k]                     = ( p.c[0][k] - xt0 ) * a11 ;
    q[k + max_polygon_pts]   = ( p.c[1][k] - yt0 ) * a22 ;
    q[k + 2*max_polygon_pts] = PROJECT ? dzt : p.c[2][k] - zt0 ;
  }
  Real_type asum0 = 0.0, asumx = 0.0, asumy = 0.0, asumz = 0.0 ;
  intsc24_shxf1( dtx, dty, xt0, yt0, zt0, q, p.n, asum0, asumx, asumy, asumz ) ;
  Real_type const det = dtx * dty ;
  sums[0] += asum0 * det ;
  sums[1] += asumx * det ;
  sums[2] += asumy * det ;
  sums[3] += asumz * det ;
}

//   Faces of the hexahedron (node lists), as in intsc24_hex.
static constexpr Int_type intsc_vface[6][4] =
  { {0, 2, 6, 4}, {1, 5, 7, 3}, {0, 4, 5, 1},
    {2, 3, 7, 6}, {0, 1, 3, 2}, {4, 6, 7, 5} } ;

//   Per-target frame factors of the 2x2x2 target block, as in intsc24_hex.
struct IntscFrame
{
  Real_type dtx[2], dty[2], dzt[2], a11[2][2], a22[2][2] ;
} ;

RAJA_INLINE void intsc_frame
    ( Real_const_ptr xp, Real_const_ptr yp, Real_const_ptr zp, IntscFrame& F )
{
  for ( Int_type k = 0 ; k < 2 ; ++k ) {
    F.dtx[k] = xp[k+1] - xp[k] ;
    F.dty[k] = yp[k+1] - yp[k] ;
    F.dzt[k] = ( zp[k+1] > zp[k] ) ? ( zp[k+1] - zp[k] ) : 0.0 ;
  }
  for ( Int_type ix = 0 ; ix < 2 ; ++ix ) {
    for ( Int_type iy = 0 ; iy < 2 ; ++iy ) {
      Real_type const det = F.dtx[ix] * F.dty[iy] ;
      Real_type const deti = det / ( det*det + 1.0e-80 ) ;
      F.a11[ix][iy] = F.dty[iy] * deti ;
      F.a22[ix][iy] = F.dtx[ix] * deti ;
    }
  }
}

//   Contribution of one triangular facet tri to the 2x2x2 targets (see
//   intsc24_hex_split8), added to out.
template < bool LEAN >
RAJA_INLINE void intsc_split8_facet
      ( IntscPoly& tri,
        Real_const_ptr xp, Real_const_ptr yp, Real_const_ptr zp,  // [3] planes
        IntscFrame const& F, Real_type (&out)[8][4] )
{
  Real_type const (&dtx)[2] = F.dtx ;
  Real_type const (&dty)[2] = F.dty ;
  Real_type const (&dzt)[2] = F.dzt ;
  Real_type const (&a11)[2][2] = F.a11 ;
  Real_type const (&a22)[2][2] = F.a22 ;
  Real_type lo[3], hi[3] ;
  for ( Int_type d = 0 ; d < 3 ; ++d ) {
    lo[d] = hi[d] = tri.c[d][0] ;
    for ( Int_type k = 1 ; k < 3 ; ++k ) {
      lo[d] = ( tri.c[d][k] < lo[d] ) ? tri.c[d][k] : lo[d] ;
      hi[d] = ( tri.c[d][k] > hi[d] ) ? tri.c[d][k] : hi[d] ;
    }
  }
  // facet outside the block's xy extent or below it contributes nothing
  if ( hi[0] < xp[0] || lo[0] >= xp[2] || hi[1] < yp[0] || lo[1] >= yp[2] ||
       hi[2] < zp[0] ) {
    return ;
  }
  IntscPoly const* cur = &tri ;
  IntscPoly cbuf[2] ;
  if constexpr ( LEAN ) {
    Int_type nb = 0 ;
    if ( lo[0] < xp[0] )  { intsc_clip_to<true >( *cur, 0, xp[0], cbuf[nb] ) ; cur = &cbuf[nb] ; nb ^= 1 ; }
    if ( hi[0] >= xp[2] ) { intsc_clip_to<false>( *cur, 0, xp[2], cbuf[nb] ) ; cur = &cbuf[nb] ; nb ^= 1 ; }
    if ( lo[1] < yp[0] )  { intsc_clip_to<true >( *cur, 1, yp[0], cbuf[nb] ) ; cur = &cbuf[nb] ; nb ^= 1 ; }
    if ( hi[1] >= yp[2] ) { intsc_clip_to<false>( *cur, 1, yp[2], cbuf[nb] ) ; cur = &cbuf[nb] ; }
  } else {
    if ( lo[0] < xp[0] )  { intsc_clip_polygon<true >( tri, 0, xp[0] ) ; }
    if ( hi[0] >= xp[2] ) { intsc_clip_polygon<false>( tri, 0, xp[2] ) ; }
    if ( lo[1] < yp[0] )  { intsc_clip_polygon<true >( tri, 1, yp[0] ) ; }
    if ( hi[1] >= yp[2] ) { intsc_clip_polygon<false>( tri, 1, yp[2] ) ; }
  }

  IntscPoly sx[2] ;
  IntscPoly const* px[2] ;
  intsc_split_or_pass( *cur, 0, xp[1], sx[0], sx[1], px[0], px[1] ) ;

  for ( Int_type ix = 0 ; ix < 2 ; ++ix ) {
    if ( px[ix] == nullptr ) { continue ; }
    IntscPoly sy[2] ;
    IntscPoly const* pxy[2] ;
    intsc_split_or_pass( *px[ix], 1, yp[1], sy[0], sy[1], pxy[0], pxy[1] ) ;

    for ( Int_type iy = 0 ; iy < 2 ; ++iy ) {
      if ( pxy[iy] == nullptr ) { continue ; }
      Real_type const ax = a11[ix][iy], ay = a22[ix][iy] ;

      // split at the shared z plane: pa below, pb above
      IntscPoly sa, sb ;
      IntscPoly const *pa, *pb ;
      intsc_split_or_pass( *pxy[iy], 2, zp[1], sa, sb, pa, pb ) ;

      // lower target: pa above zp[0] (central), pb projected to its top
      Real_type (&s0)[4] = out[ix + 2*iy] ;
      if ( pa != nullptr ) {
        IntscPoly sl, sh ;
        IntscPoly const *pl, *ph ;
        if constexpr ( LEAN ) {
          ph = intsc_clip_or_pass<true>( *pa, 2, zp[0], sh ) ;
        } else {
          intsc_split_or_pass( *pa, 2, zp[0], sl, sh, pl, ph ) ;
        }
        if ( ph != nullptr ) {
          intsc_add_polygon<false, LEAN>( *ph, xp[ix], dtx[ix], ax, yp[iy], dty[iy], ay,
                                    zp[0], dzt[0], s0 ) ;
        }
      }
      if ( pb != nullptr ) {
        intsc_add_polygon<true, LEAN>( *pb, xp[ix], dtx[ix], ax, yp[iy], dty[iy], ay,
                                 zp[0], dzt[0], s0 ) ;

        // upper target: pb below zp[2] (central), above it projected
        Real_type (&s1)[4] = out[ix + 2*iy + 4] ;
        IntscPoly sc, su ;
        IntscPoly const *pc, *pu ;
        intsc_split_or_pass( *pb, 2, zp[2], sc, su, pc, pu ) ;
        if ( pc != nullptr ) {
          intsc_add_polygon<false, LEAN>( *pc, xp[ix], dtx[ix], ax, yp[iy], dty[iy], ay,
                                    zp[1], dzt[1], s1 ) ;
        }
        if ( pu != nullptr ) {
          intsc_add_polygon<true, LEAN>( *pu, xp[ix], dtx[ix], ax, yp[iy], dty[iy], ay,
                                   zp[1], dzt[1], s1 ) ;
        }
      }
    }
  }
}

//   Volume and moments of one donor hexahedron intersected with the 2x2x2
//   target zones bounded by planes xp[0..2], yp[0..2], zp[0..2].
//   out[r][0..3] for target r = ix + 2*iy + 4*iz.
//   LEAN: clips write only the kept side's valid vertices into ping-pong
//   buffers and polygons are integrated without staging copies.
template < bool LEAN = false >
RAJA_INLINE void intsc24_hex_split8
      ( Real_const_ptr xd, Real_const_ptr yd, Real_const_ptr zd,  // [8] donor
        Real_const_ptr xp, Real_const_ptr yp, Real_const_ptr zp,  // [3] planes
        Real_type (&out)[8][4] )
{
  for ( Int_type r = 0 ; r < 8 ; ++r ) {
    out[r][0] = out[r][1] = out[r][2] = out[r][3] = 0.0 ;
  }
  IntscFrame F ;
  intsc_frame( xp, yp, zp, F ) ;

  for ( Int_type f = 0 ; f < 6 ; ++f ) {
    Int_type const* v = intsc_vface[f] ;
    Real_type const fx = 0.25 * ( xd[v[0]] + xd[v[1]] + xd[v[2]] + xd[v[3]] ) ;
    Real_type const fy = 0.25 * ( yd[v[0]] + yd[v[1]] + yd[v[2]] + yd[v[3]] ) ;
    Real_type const fz = 0.25 * ( zd[v[0]] + zd[v[1]] + zd[v[2]] + zd[v[3]] ) ;

    for ( Int_type k0 = 0 ; k0 < 4 ; ++k0 ) {
      Int_type const vv0 = v[k0], vv2 = v[(k0 + 1) & 3] ;

      // facet (vv0, face center, vv2), same orientation as intsc24_hex_get_tri
      IntscPoly tri ;
      tri.n = 3 ;
      tri.c[0][0] = xd[vv0] ;  tri.c[0][1] = fx ;  tri.c[0][2] = xd[vv2] ;
      tri.c[1][0] = yd[vv0] ;  tri.c[1][1] = fy ;  tri.c[1][2] = yd[vv2] ;
      tri.c[2][0] = zd[vv0] ;  tri.c[2][1] = fz ;  tri.c[2][2] = zd[vv2] ;
      intsc_split8_facet<LEAN>( tri, xp, yp, zp, F, out ) ;
    }
  }
}

//   One donor zone (records 8*dz .. 8*dz+7, targets in the order of
//   INTSC_HEXRECT::setupIntscPairs) with the plane-splitting algorithm.
#define INTSC_HEXRECT_SPLIT_BODY(dz) INTSC_HEXRECT_SPLIT_BODY_T(dz, false)
#define INTSC_HEXRECT_SPLIT_BODY_T(dz, LEAN) \
  { \
    Index_type const irec0 = 8 * (dz) ; \
    Real_type xd[8], yd[8], zd[8] ; \
    { \
      Int_type const dzone = intsc_d[irec0] ; \
      for ( Index_type j = 0 ; j < 8 ; j++ ) { \
        Int_type const node = znlist[ 8*dzone + j ] ; \
        xd[j] = xdnode[node] ; \
        yd[j] = ydnode[node] ; \
        zd[j] = zdnode[node] ; \
      } \
    } \
    auto plane = reinterpret_cast<typename INTSC_HEXRECT::Plane*>(ncord_gpu) ; \
    Int_type const nyzones = plane->ncord[1] ; \
    Int_type const nxzones = plane->ncord[2] ; \
    Int_type const tz = intsc_t[irec0] ; \
    Int_type const jz = tz / ( nxzones * nyzones ) ; \
    Int_type const jy = ( tz / nxzones ) % nyzones ; \
    Int_type const jx = tz % nxzones ; \
    Real_type out[8][4] ; \
    intsc24_hex_split8<LEAN>( xd, yd, zd, plane->planes[2] + jx, plane->planes[1] + jy, \
                        plane->planes[0] + jz, out ) ; \
    Real_ptr record = records + 4 * irec0 ; \
    for ( Index_type r = 0 ; r < 8 ; ++r ) { \
      for ( Index_type v = 0 ; v < 4 ; ++v ) { \
        record[4*r + v] = out[r][v] ; \
      } \
    } \
  }

#if defined(RAJAPERF_FEM_AVX512)
//  ---------------------------------------------------------------------------
//  SIMD plane splitting: intsc24_hex_split8<true> for 8 donor zones at once,
//  one zone per AVX-512 lane.  Polygons share one vertex count across lanes,
//  so every decision (which side of a plane each vertex lies on, bounding box
//  tests) must agree in all lanes; the functions return false as soon as one
//  does not, and that facet is recomputed with the scalar code lane by lane.
//  Neighboring donor zones of a regular mesh mostly agree.
//  ---------------------------------------------------------------------------
namespace intsc_simd {

using fem_simd::vd ;
using fem_simd::vmask ;

struct VPoly
{
  static constexpr Int_type cap = 12 ;   // triangle clipped by <= 8 planes
  vd c[3][cap] ;
  Int_type n = 0 ;
} ;

RAJA_INLINE vmask vlt( vd a, vd b ) { return _mm512_cmp_pd_mask( a, b, _CMP_LT_OQ ) ; }

//   1 if m holds for all lanes, 0 for none, -1 if mixed.
RAJA_INLINE int uniform( vmask m ) { return m == 0xff ? 1 : ( m == 0 ? 0 : -1 ) ; }

//   For each vertex k: below[k] = c[axis][k] < cut in all lanes (true) or in
//   none (false).  Returns false if mixed.
RAJA_INLINE bool classify( VPoly const& p, Int_type const axis, vd const cut,
                           bool (&below)[VPoly::cap] )
{
  for ( Int_type k = 0 ; k < p.n ; ++k ) {
    int const u = uniform( vlt( p.c[axis][k], cut ) ) ;
    if ( u < 0 ) { return false ; }
    below[k] = u ;
  }
  return true ;
}

RAJA_INLINE void cross( VPoly const& p, Int_type const axis, vd const cut,
                        Int_type const k, Int_type const j, vd (&v)[3] )
{
  vd const a = p.c[axis][k] ;
  vd const eta = fem_simd::vdiv( fem_simd::vsub( cut, a ),
                                 fem_simd::vsub( p.c[axis][j], a ) ) ;
  for ( Int_type d = 0 ; d < 3 ; ++d ) {
    v[d] = ( d == axis ) ? cut
         : fem_simd::vfma( fem_simd::vsub( p.c[d][j], p.c[d][k] ), eta, p.c[d][k] ) ;
  }
}

//   intsc_split_or_pass for uniform vertex sides.
RAJA_INLINE bool split_or_pass( VPoly const& p, Int_type const axis, vd const cut,
                                VPoly& slo, VPoly& shi,
                                VPoly const*& lo, VPoly const*& hi )
{
  lo = hi = nullptr ;
  if ( p.n < 3 ) { return true ; }
  bool below[VPoly::cap] ;
  if ( !classify( p, axis, cut, below ) ) { return false ; }
  bool all_lo = true, all_hi = true ;
  for ( Int_type k = 0 ; k < p.n ; ++k ) { all_lo &= below[k] ; all_hi &= !below[k] ; }
  if ( all_lo ) { lo = &p ; return true ; }
  if ( all_hi ) { hi = &p ; return true ; }
  slo.n = shi.n = 0 ;
  for ( Int_type k = 0 ; k < p.n ; ++k ) {
    Int_type const j = ( k + 1 == p.n ) ? 0 : k + 1 ;
    VPoly& side = below[k] ? slo : shi ;
    for ( Int_type d = 0 ; d < 3 ; ++d ) { side.c[d][side.n] = p.c[d][k] ; }
    ++side.n ;
    if ( below[k] != below[j] ) {
      vd v[3] ;
      cross( p, axis, cut, k, j, v ) ;
      for ( Int_type d = 0 ; d < 3 ; ++d ) { slo.c[d][slo.n] = v[d] ;  shi.c[d][shi.n] = v[d] ; }
      ++slo.n ;  ++shi.n ;
    }
  }
  lo = ( slo.n >= 3 ) ? &slo : nullptr ;
  hi = ( shi.n >= 3 ) ? &shi : nullptr ;
  return true ;
}

//   intsc_clip_to (no pass-through) for uniform vertex sides.
template < bool KEEP_HI >
RAJA_INLINE bool clip_to( VPoly const& p, Int_type const axis, vd const cut, VPoly& out )
{
  bool below[VPoly::cap] ;
  if ( !classify( p, axis, cut, below ) ) { return false ; }
  Int_type n = 0 ;
  for ( Int_type k = 0 ; k < p.n ; ++k ) {
    Int_type const j = ( k + 1 == p.n ) ? 0 : k + 1 ;
    if ( below[k] != KEEP_HI ) {
      for ( Int_type d = 0 ; d < 3 ; ++d ) { out.c[d][n] = p.c[d][k] ; }
      ++n ;
    }
    if ( below[k] != below[j] ) {
      vd v[3] ;
      cross( p, axis, cut, k, j, v ) ;
      for ( Int_type d = 0 ; d < 3 ; ++d ) { out.c[d][n] = v[d] ; }
      ++n ;
    }
  }
  out.n = n ;
  return true ;
}

//   intsc_clip_or_pass<true> (keep c[axis] >= cut) for uniform vertex sides.
RAJA_INLINE bool clip_hi_or_pass( VPoly const& p, Int_type const axis, vd const cut,
                                  VPoly& out, VPoly const*& res )
{
  res = nullptr ;
  if ( p.n < 3 ) { return true ; }
  bool below[VPoly::cap] ;
  if ( !classify( p, axis, cut, below ) ) { return false ; }
  bool all_lo = true, all_hi = true ;
  for ( Int_type k = 0 ; k < p.n ; ++k ) { all_lo &= below[k] ; all_hi &= !below[k] ; }
  if ( all_lo ) { return true ; }
  if ( all_hi ) { res = &p ; return true ; }
  if ( !clip_to<true>( p, axis, cut, out ) ) { return false ; }
  res = ( out.n >= 3 ) ? &out : nullptr ;
  return true ;
}

//   intsc_add_polygon<PROJECT, true> per lane.
template < bool PROJECT >
RAJA_INLINE void add_polygon( VPoly const& p,
                              vd const xt0, vd const dtx, vd const a11,
                              vd const yt0, vd const dty, vd const a22,
                              vd const zt0, vd const dzt, vd (&sums)[4] )
{
  using namespace fem_simd ;
  if ( p.n < 3 ) { return ; }
  vd const one3 = vset1( 0.33333333333333333 ), half = vset1( 0.5 ) ;
  auto tx = [&] ( Int_type k ) { return vmul( vsub( p.c[0][k], xt0 ), a11 ) ; } ;
  auto ty = [&] ( Int_type k ) { return vmul( vsub( p.c[1][k], yt0 ), a22 ) ; } ;
  auto tz = [&] ( Int_type k ) { return PROJECT ? dzt : vsub( p.c[2][k], zt0 ) ; } ;
  vd const hdx = vmul( half, dtx ), hdy = vmul( half, dty ) ;
  vd asum0 = vzero(), asumx = vzero(), asumy = vzero(), asumz = vzero() ;
  vd const xc0 = tx(0), yc0 = ty(0), zc0 = tz(0) ;
  vd xc1 = tx(1), yc1 = ty(1), zc1 = tz(1) ;
  for ( Int_type kk = 2 ; kk < p.n ; kk++ ) {
    vd const xc2 = tx(kk), yc2 = ty(kk), zc2 = tz(kk) ;
    vd s0tri, sxtri, sytri, sztri, m ;
    m = vmul( half, vadd( zc0, zc1 ) ) ;
    s0tri = m ;
    sxtri = vmul( m, vfma( hdx, vadd( xc0, xc1 ), xt0 ) ) ;
    sytri = vmul( m, vfma( hdy, vadd( yc0, yc1 ), yt0 ) ) ;
    sztri = vmul( m, vfma( half, m, zt0 ) ) ;
    m = vmul( half, vadd( zc1, zc2 ) ) ;
    s0tri = vadd( s0tri, m ) ;
    sxtri = vfma( m, vfma( hdx, vadd( xc1, xc2 ), xt0 ), sxtri ) ;
    sytri = vfma( m, vfma( hdy, vadd( yc1, yc2 ), yt0 ), sytri ) ;
    sztri = vfma( m, vfma( half, m, zt0 ), sztri ) ;
    m = vmul( half, vadd( zc2, zc0 ) ) ;
    s0tri = vadd( s0tri, m ) ;
    sxtri = vfma( m, vfma( hdx, vadd( xc2, xc0 ), xt0 ), sxtri ) ;
    sytri = vfma( m, vfma( hdy, vadd( yc2, yc0 ), yt0 ), sytri ) ;
    sztri = vfma( m, vfma( half, m, zt0 ), sztri ) ;
    vd area = vmul( vsub( xc0, xc1 ), vadd( yc0, yc1 ) ) ;
    area = vfma( vsub( xc1, xc2 ), vadd( yc1, yc2 ), area ) ;
    area = vfma( vsub( xc2, xc0 ), vadd( yc2, yc0 ), area ) ;
    vd const w = vmul( vmul( half, one3 ), area ) ;
    asum0 = vfma( w, s0tri, asum0 ) ;
    asumx = vfma( w, sxtri, asumx ) ;
    asumy = vfma( w, sytri, asumy ) ;
    asumz = vfma( w, sztri, asumz ) ;
    xc1 = xc2 ;  yc1 = yc2 ;  zc1 = zc2 ;
  }
  vd const det = vmul( dtx, dty ) ;
  sums[0] = vfma( asum0, det, sums[0] ) ;
  sums[1] = vfma( asumx, det, sums[1] ) ;
  sums[2] = vfma( asumy, det, sums[2] ) ;
  sums[3] = vfma( asumz, det, sums[3] ) ;
}

//   Frame factors of the 2x2x2 target block per lane, as intsc_frame.
struct VFrame
{
  vd dtx[2], dty[2], dzt[2], a11[2][2], a22[2][2] ;
} ;

//   Contribution of one facet (intsc_split8_facet<true> per lane) to the
//   facet-local sums acc[r] of the targets r in 'touched'.  Returns false,
//   with acc only partially updated, if the lanes take different paths.
RAJA_INLINE bool vfacet( VPoly const& tri,
                         vd const (&xp)[3], vd const (&yp)[3], vd const (&zp)[3],
                         VFrame const& F, vd (&acc)[8][4], unsigned& touched )
{
  using namespace fem_simd ;
  auto target = [&] ( Int_type r ) -> vd (&)[4] {
    if ( !( ( touched >> r ) & 1u ) ) {
      acc[r][0] = acc[r][1] = acc[r][2] = acc[r][3] = vzero() ;
      touched |= 1u << r ;
    }
    return acc[r] ;
  } ;

  vd lo[3], hi[3] ;
  for ( Int_type d = 0 ; d < 3 ; ++d ) {
    lo[d] = _mm512_min_pd( _mm512_min_pd( tri.c[d][0], tri.c[d][1] ), tri.c[d][2] ) ;
    hi[d] = _mm512_max_pd( _mm512_max_pd( tri.c[d][0], tri.c[d][1] ), tri.c[d][2] ) ;
  }
  vmask const skip = vlt( hi[0], xp[0] ) | ( vmask )~vlt( lo[0], xp[2] ) |
                     vlt( hi[1], yp[0] ) | ( vmask )~vlt( lo[1], yp[2] ) |
                     vlt( hi[2], zp[0] ) ;
  int const us = uniform( skip ) ;
  if ( us < 0 ) { return false ; }
  if ( us == 1 ) { return true ; }

  VPoly const* cur = &tri ;
  VPoly cbuf[2] ;
  Int_type nb = 0 ;
  int u ;
  if ( ( u = uniform( vlt( lo[0], xp[0] ) ) ) != 0 ) {
    if ( u < 0 || !clip_to<true>( *cur, 0, xp[0], cbuf[nb] ) ) { return false ; }
    cur = &cbuf[nb] ;  nb ^= 1 ;
  }
  if ( ( u = uniform( ( vmask )~vlt( hi[0], xp[2] ) ) ) != 0 ) {
    if ( u < 0 || !clip_to<false>( *cur, 0, xp[2], cbuf[nb] ) ) { return false ; }
    cur = &cbuf[nb] ;  nb ^= 1 ;
  }
  if ( ( u = uniform( vlt( lo[1], yp[0] ) ) ) != 0 ) {
    if ( u < 0 || !clip_to<true>( *cur, 1, yp[0], cbuf[nb] ) ) { return false ; }
    cur = &cbuf[nb] ;  nb ^= 1 ;
  }
  if ( ( u = uniform( ( vmask )~vlt( hi[1], yp[2] ) ) ) != 0 ) {
    if ( u < 0 || !clip_to<false>( *cur, 1, yp[2], cbuf[nb] ) ) { return false ; }
    cur = &cbuf[nb] ;
  }

  VPoly sx[2] ;
  VPoly const* px[2] ;
  if ( !split_or_pass( *cur, 0, xp[1], sx[0], sx[1], px[0], px[1] ) ) { return false ; }

  for ( Int_type ix = 0 ; ix < 2 ; ++ix ) {
    if ( px[ix] == nullptr ) { continue ; }
    VPoly sy[2] ;
    VPoly const* pxy[2] ;
    if ( !split_or_pass( *px[ix], 1, yp[1], sy[0], sy[1], pxy[0], pxy[1] ) ) { return false ; }

    for ( Int_type iy = 0 ; iy < 2 ; ++iy ) {
      if ( pxy[iy] == nullptr ) { continue ; }
      vd const ax = F.a11[ix][iy], ay = F.a22[ix][iy] ;
      VPoly sa, sb ;
      VPoly const *pa, *pb ;
      if ( !split_or_pass( *pxy[iy], 2, zp[1], sa, sb, pa, pb ) ) { return false ; }

      if ( pa != nullptr ) {
        VPoly sh ;
        VPoly const* ph ;
        if ( !clip_hi_or_pass( *pa, 2, zp[0], sh, ph ) ) { return false ; }
        if ( ph != nullptr ) {
          add_polygon<false>( *ph, xp[ix], F.dtx[ix], ax, yp[iy], F.dty[iy], ay,
                              zp[0], F.dzt[0], target( ix + 2*iy ) ) ;
        }
      }
      if ( pb != nullptr ) {
        add_polygon<true>( *pb, xp[ix], F.dtx[ix], ax, yp[iy], F.dty[iy], ay,
                           zp[0], F.dzt[0], target( ix + 2*iy ) ) ;
        VPoly sc, su ;
        VPoly const *pc, *pu ;
        if ( !split_or_pass( *pb, 2, zp[2], sc, su, pc, pu ) ) { return false ; }
        if ( pc != nullptr ) {
          add_polygon<false>( *pc, xp[ix], F.dtx[ix], ax, yp[iy], F.dty[iy], ay,
                              zp[1], F.dzt[1], target( ix + 2*iy + 4 ) ) ;
        }
        if ( pu != nullptr ) {
          add_polygon<true>( *pu, xp[ix], F.dtx[ix], ax, yp[iy], F.dty[iy], ay,
                             zp[1], F.dzt[1], target( ix + 2*iy + 4 ) ) ;
        }
      }
    }
  }
  return true ;
}

//   intsc24_hex_split8<true> for 8 donor zones; out[4*r + v] for target r.
//   ld[l], lxp[l] are lane l's node coordinates (x[8], y[8], z[8]) and
//   planes (x[3], y[3], z[3]).  A facet whose lanes take different paths is
//   recomputed with intsc_split8_facet<true> lane by lane.
RAJA_INLINE void hex_split8( Real_type const (&ld)[8][24], Real_type const (&lp)[8][9],
                             vd (&out)[32] )
{
  using namespace fem_simd ;
  vd c[24], pl[9] ;
  {
    Real_type const* rows[8] ;
    for ( Int_type l = 0 ; l < 8 ; ++l ) { rows[l] = ld[l] ; }
    load_transpose<24>( rows, c ) ;
    for ( Int_type l = 0 ; l < 8 ; ++l ) { rows[l] = lp[l] ; }
    load_transpose<9>( rows, pl ) ;
  }
  vd const* xd = c ;
  vd const* yd = c + 8 ;
  vd const* zd = c + 16 ;
  vd const (&xp)[3] = reinterpret_cast<vd const (&)[3]>( pl[0] ) ;
  vd const (&yp)[3] = reinterpret_cast<vd const (&)[3]>( pl[3] ) ;
  vd const (&zp)[3] = reinterpret_cast<vd const (&)[3]>( pl[6] ) ;

  for ( Int_type r = 0 ; r < 32 ; ++r ) { out[r] = vzero() ; }

  VFrame F ;
  for ( Int_type k = 0 ; k < 2 ; ++k ) {
    F.dtx[k] = vsub( xp[k+1], xp[k] ) ;
    F.dty[k] = vsub( yp[k+1], yp[k] ) ;
    F.dzt[k] = _mm512_max_pd( vsub( zp[k+1], zp[k] ), vzero() ) ;
  }
  for ( Int_type ix = 0 ; ix < 2 ; ++ix ) {
    for ( Int_type iy = 0 ; iy < 2 ; ++iy ) {
      vd const det = vmul( F.dtx[ix], F.dty[iy] ) ;
      vd const deti = vdiv( det, vfma( det, det, vset1( 1.0e-80 ) ) ) ;
      F.a11[ix][iy] = vmul( F.dty[iy], deti ) ;
      F.a22[ix][iy] = vmul( F.dtx[ix], deti ) ;
    }
  }

  bool scalar_used = false ;
  IntscFrame sf[8] ;
  Real_type so[8][8][4] ;
  vd const quarter = vset1( 0.25 ) ;

  for ( Int_type f = 0 ; f < 6 ; ++f ) {
    Int_type const* v = intsc_vface[f] ;
    vd const fx = vmul( quarter, vadd( vadd( xd[v[0]], xd[v[1]] ), vadd( xd[v[2]], xd[v[3]] ) ) ) ;
    vd const fy = vmul( quarter, vadd( vadd( yd[v[0]], yd[v[1]] ), vadd( yd[v[2]], yd[v[3]] ) ) ) ;
    vd const fz = vmul( quarter, vadd( vadd( zd[v[0]], zd[v[1]] ), vadd( zd[v[2]], zd[v[3]] ) ) ) ;

    for ( Int_type k0 = 0 ; k0 < 4 ; ++k0 ) {
      Int_type const vv0 = v[k0], vv2 = v[(k0 + 1) & 3] ;
      VPoly tri ;
      tri.n = 3 ;
      tri.c[0][0] = xd[vv0] ;  tri.c[0][1] = fx ;  tri.c[0][2] = xd[vv2] ;
      tri.c[1][0] = yd[vv0] ;  tri.c[1][1] = fy ;  tri.c[1][2] = yd[vv2] ;
      tri.c[2][0] = zd[vv0] ;  tri.c[2][1] = fz ;  tri.c[2][2] = zd[vv2] ;

      vd acc[8][4] ;
      unsigned touched = 0 ;
      if ( vfacet( tri, xp, yp, zp, F, acc, touched ) ) {
        for ( Int_type r = 0 ; r < 8 ; ++r ) {
          if ( ( touched >> r ) & 1u ) {
            for ( Int_type q = 0 ; q < 4 ; ++q ) {
              out[4*r + q] = vadd( out[4*r + q], acc[r][q] ) ;
            }
          }
        }
        continue ;
      }

      if ( !scalar_used ) {
        scalar_used = true ;
        for ( Int_type l = 0 ; l < 8 ; ++l ) {
          intsc_frame( lp[l], lp[l] + 3, lp[l] + 6, sf[l] ) ;
          for ( Int_type r = 0 ; r < 8 ; ++r ) {
            so[l][r][0] = so[l][r][1] = so[l][r][2] = so[l][r][3] = 0.0 ;
          }
        }
      }
      for ( Int_type l = 0 ; l < 8 ; ++l ) {
        Real_type const* lx = ld[l] ;
        Real_type const* ly = ld[l] + 8 ;
        Real_type const* lz = ld[l] + 16 ;
        IntscPoly stri ;
        stri.n = 3 ;
        stri.c[0][0] = lx[vv0] ;  stri.c[0][2] = lx[vv2] ;
        stri.c[1][0] = ly[vv0] ;  stri.c[1][2] = ly[vv2] ;
        stri.c[2][0] = lz[vv0] ;  stri.c[2][2] = lz[vv2] ;
        stri.c[0][1] = 0.25 * ( lx[v[0]] + lx[v[1]] + lx[v[2]] + lx[v[3]] ) ;
        stri.c[1][1] = 0.25 * ( ly[v[0]] + ly[v[1]] + ly[v[2]] + ly[v[3]] ) ;
        stri.c[2][1] = 0.25 * ( lz[v[0]] + lz[v[1]] + lz[v[2]] + lz[v[3]] ) ;
        intsc_split8_facet<true>( stri, lp[l], lp[l] + 3, lp[l] + 6, sf[l], so[l] ) ;
      }
    }
  }

  if ( scalar_used ) {
    alignas(64) Real_type t[8] ;
    for ( Int_type r = 0 ; r < 8 ; ++r ) {
      for ( Int_type q = 0 ; q < 4 ; ++q ) {
        for ( Int_type l = 0 ; l < 8 ; ++l ) { t[l] = so[l][r][q] ; }
        out[4*r + q] = vadd( out[4*r + q], _mm512_load_pd( t ) ) ;
      }
    }
  }
}

//   Donor zones [dz0, dz0 + nel), nel <= 8: gather the zones and their target
//   planes into lanes (tail lanes replicate the last zone), run hex_split8 and
//   write the records.
RAJAPERF_FEM_NOINLINE void split8_batch(
    Real_const_ptr xdnode, Real_const_ptr ydnode, Real_const_ptr zdnode,
    Int_type const* znlist, Int_type const* intsc_d, Int_type const* intsc_t,
    Real_const_ptr xplane, Real_const_ptr yplane, Real_const_ptr zplane,
    Int_type const nxzones, Int_type const nyzones,
    Real_ptr records, Index_type const dz0, Index_type const nel )
{
  alignas(64) Real_type ld[8][24], lp[8][9] ;
  for ( Index_type l = 0 ; l < 8 ; ++l ) {
    Index_type const irec0 = 8 * ( dz0 + ( ( l < nel ) ? l : nel - 1 ) ) ;
    Int_type const dzone = intsc_d[irec0] ;
    for ( Index_type j = 0 ; j < 8 ; ++j ) {
      Int_type const node = znlist[ 8*dzone + j ] ;
      ld[l][j]      = xdnode[node] ;
      ld[l][j + 8]  = ydnode[node] ;
      ld[l][j + 16] = zdnode[node] ;
    }
    Int_type const tz = intsc_t[irec0] ;
    Int_type const jz = tz / ( nxzones * nyzones ) ;
    Int_type const jy = ( tz / nxzones ) % nyzones ;
    Int_type const jx = tz % nxzones ;
    for ( Index_type k = 0 ; k < 3 ; ++k ) {
      lp[l][k]     = xplane[jx + k] ;
      lp[l][k + 3] = yplane[jy + k] ;
      lp[l][k + 6] = zplane[jz + k] ;
    }
  }

  alignas(64) vd out[32] ;
  RAJAPERF_FEM_STACK_ALIGN( out ) ;
  hex_split8( ld, lp, out ) ;

  Real_type* rows[8] ;
  for ( Index_type l = 0 ; l < 8 ; ++l ) {
    rows[l] = records + 32 * ( dz0 + ( ( l < nel ) ? l : nel - 1 ) ) ;
  }
  fem_simd::transpose_store<32>( rows, nel, out ) ;
}

}  // end namespace intsc_simd

//   Donor zones [8*b, 8*b + 8) of nzones in SIMD lanes.
#define INTSC_HEXRECT_SPLIT_SIMD_BODY(b, nzones) \
  { \
    Index_type const dz0 = 8 * (b) ; \
    Index_type const nel = ( (nzones) - dz0 < 8 ) ? (nzones) - dz0 : 8 ; \
    auto plane = reinterpret_cast<typename INTSC_HEXRECT::Plane*>(ncord_gpu) ; \
    intsc_simd::split8_batch( xdnode, ydnode, zdnode, znlist, intsc_d, intsc_t, \
                              plane->planes[2], plane->planes[1], plane->planes[0], \
                              plane->ncord[2], plane->ncord[1], records, dz0, nel ) ; \
  }
#endif

}  // end namespace rajaperf


#define INTSC_HEXRECT_BODY \
  if ( irec < nrecords ) { \
    Real_ptr record = ((Real_type*)records) + 4 * irec ; \
    Real_type xd[24] ; \
    Real_ptr yd = xd + 8 ; \
    Real_ptr zd = yd + 8 ; \
    { \
      Int_type dzone = intsc_d[irec] ; \
      for ( Index_type j=0 ; j<8 ; j++) { \
        Int_type node = znlist[ 8*dzone + j ] ; \
        xd[j] = xdnode[node] ; \
        yd[j] = ydnode[node] ; \
        zd[j] = zdnode[node] ; } \
    } \
    { \
      Real_type sum0, sumx, sumy, sumz ; \
      Real_const_ptr zplane ; \
      Real_const_ptr yplane ; \
      Real_const_ptr xplane ; \
      Int_type jz, jy, jx ; \
      { \
        auto plane = reinterpret_cast<typename INTSC_HEXRECT::Plane*>(ncord_gpu) ; \
        zplane = plane->planes[0] ; \
        yplane = plane->planes[1] ; \
        xplane = plane->planes[2] ; \
        Int_type const nyzones = plane->ncord[1] ; \
        Int_type const nxzones = plane->ncord[2] ; \
        Int_type tz = intsc_t[irec] ; \
        jz = tz / ( nxzones * nyzones ) ; \
        jy = ( tz / nxzones ) % nyzones ; \
        jx = tz % nxzones ; \
      } \
      intsc24_hex \
          ( xd, my_qx, \
            xplane[jx], xplane[jx+1], yplane[jy], yplane[jy+1], \
            zplane[jz], zplane[jz+1], \
            sum0, sumx, sumy, sumz ) ; \
      { \
        Real_type vf   = 1.0 ; \
        record[0] = vf * sum0 ; \
        record[1] = vf * sumx ; \
        record[2] = vf * sumy ; \
        record[3] = vf * sumz ; \
      } \
    } \
  }

#define INTSC_HEXRECT_SEQ(i) \
  Index_type irec = i ; \
  Real_type xd_work[ 3*max_polygon_pts+1 ] ; \
  Real_ptr my_qx = xd_work ; \
  INTSC_HEXRECT_BODY ;

#define INTSC_HEXRECT_OMP(i) \
  INTSC_HEXRECT_SEQ(i)

#endif // closing include guard RAJAPerf_Apps_INTSC_HEXRECT_BODY_HPP
