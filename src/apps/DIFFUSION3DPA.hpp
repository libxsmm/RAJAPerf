//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//clang-format off
///
/// Element-wise action of the 3D finite element volume diffusion operator
/// via partial assembly and sum factorization
///
/// Based on MFEM's/CEED algorithms.
/// Reference implementation - MFEM-v4.9
/// https://github.com/mfem/mfem/blob/v4.9/fem/integ/bilininteg_diffusion_kernels.hpp
///
/// for (Index_type e = 0; e < NE; ++e) {
///
///   constexpr Index_type MQ1 = diff::Q1D;
///   constexpr Index_type MD1 = diff::D1D;
///   constexpr Index_type MDQ = (MQ1 >  ? MQ1 : MD1;
///   Real_type sBG[MQ1*MD1];
///   Real_type (*B)[MD1] = (Real_type (*)[MD1]) sBG;
///   Real_type (*G)[MD1] = (Real_type (*)[MD1]) sBG;
///   Real_type (*Bt)[MQ1] = (Real_type (*)[MQ1]) sBG;
///   Real_type (*Gt)[MQ1] = (Real_type (*)[MQ1]) sBG;
///   Real_type sm0[3][MDQ*MDQ*MDQ];
///   Real_type sm1[3][MDQ*MDQ*MDQ];
///   Real_type (*X)[MD1][MD1]    = (Real_type (*)[MD1][MD1]) (sm0+2);
///   Real_type (*DDQ0)[MD1][MQ1] = (Real_type (*)[MD1][MQ1]) (sm0+0);
///   Real_type (*DDQ1)[MD1][MQ1] = (Real_type (*)[MD1][MQ1]) (sm0+1);
///   Real_type (*DQQ0)[MQ1][MQ1] = (Real_type (*)[MQ1][MQ1]) (sm1+0);
///   Real_type (*DQQ1)[MQ1][MQ1] = (Real_type (*)[MQ1][MQ1]) (sm1+1);
///   Real_type (*DQQ2)[MQ1][MQ1] = (Real_type (*)[MQ1][MQ1]) (sm1+2);
///   Real_type (*QQQ0)[MQ1][MQ1] = (Real_type (*)[MQ1][MQ1]) (sm0+0);
///   Real_type (*QQQ1)[MQ1][MQ1] = (Real_type (*)[MQ1][MQ1]) (sm0+1);
///   Real_type (*QQQ2)[MQ1][MQ1] = (Real_type (*)[MQ1][MQ1]) (sm0+2);
///   Real_type (*QQD0)[MQ1][MD1] = (Real_type (*)[MQ1][MD1]) (sm1+0);
///   Real_type (*QQD1)[MQ1][MD1] = (Real_type (*)[MQ1][MD1]) (sm1+1);
///   Real_type (*QQD2)[MQ1][MD1] = (Real_type (*)[MQ1][MD1]) (sm1+2);
///   Real_type (*QDD0)[MD1][MD1] = (Real_type (*)[MD1][MD1]) (sm0+0);
///   Real_type (*QDD1)[MD1][MD1] = (Real_type (*)[MD1][MD1]) (sm0+1);
///   Real_type (*QDD2)[MD1][MD1] = (Real_type (*)[MD1][MD1]) (sm0+2);
///
///   for(Index_type dz=0; dz<diff::D1D; ++dz)
///   {
///    for(Index_type dy=0; dy<diff::D1D; ++dy)
///    {
///      for(Index_type dx=0; dx<diff::D1D; ++dx)
///      {
///        s_X[dz][dy][dx] = DPA_X(dx,dy,dz,e);
///      }
///    }
///  }
///
///  for(Index_type dy=0; dy<diff::D1D; ++dy)
///  {
///    for(Index_type qx=0; qx<diff::Q1D; ++qx)
///    {
///      B[qx][dy] = DPA_b(qx,dy);
///      G[qx][dy] = DPA_g(qx,dy);
///    }
///  }
///
///  for(Index_type dz=0; dz<diff::D1D; ++dz)
///  {
///    for(Index_type dy=0; dy<diff::D1D; ++dy)
///    {
///      for(Index_type qx=0; qx<diff::Q1D; ++qx)
///      {
///        Real_type u = 0.0, v = 0.0;
///        RAJAPERF_UNROLL(MD1)
///        for (Index_type dx = 0; dx < diff::D1D; ++dx)
///        {
///          const Real_type coords = s_X[dz][dy][dx];
///          u += coords * B[qx][dx];
///          v += coords * G[qx][dx];
///        }
///        DDQ0[dz][dy][qx] = u;
///        DDQ1[dz][dy][qx] = v;
///      }
///    }
///  }
///
///  for(Index_type dz=0; dz<diff::D1D; ++dz)
///  {
///    for(Index_type qy=0; qy<diff::Q1D; ++qy)
///    {
///      for(Index_type qx=0; qx<diff::Q1D; ++qx)
///      {
///        Real_type u = 0.0, v = 0.0, w = 0.0;
///        RAJAPERF_UNROLL(MD1)
///        for (Index_type dy = 0; dy < diff::D1D; ++dy)
///        {
///          u += DDQ1[dz][dy][qx] * B[qy][dy];
///          v += DDQ0[dz][dy][qx] * G[qy][dy];
///          w += DDQ0[dz][dy][qx] * B[qy][dy];
///        }
///        DQQ0[dz][qy][qx] = u;
///        DQQ1[dz][qy][qx] = v;
///        DQQ2[dz][qy][qx] = w;
///      }
///    }
///  }
///
///  for(Index_type qz=0; qz<diff::Q1D; ++qz)
///  {
///    for(Index_type qy=0; qy<diff::Q1D; ++qy)
///    {
///      for(Index_type qx=0; qx<diff::Q1D; ++qx)
///      {
///        Real_type u = 0.0, v = 0.0, w = 0.0;
///        RAJAPERF_UNROLL(MD1)
///        for (Index_type dz = 0; dz < diff::D1D; ++dz)
///        {
///          u += DQQ0[dz][qy][qx] * B[qz][dz];
///          v += DQQ1[dz][qy][qx] * B[qz][dz];
///          w += DQQ2[dz][qy][qx] * G[qz][dz];
///        }
///        const Real_type O11 = DPA_d(qx,qy,qz,0,e);
///        const Real_type O12 = DPA_d(qx,qy,qz,1,e);
///        const Real_type O13 = DPA_d(qx,qy,qz,2,e);
///        const Real_type O21 = symmetric ? O12 : DPA_d(qx,qy,qz,3,e);
///        const Real_type O22 = symmetric ? DPA_d(qx,qy,qz,3,e) : DPA_d(qx,qy,qz,4,e);
///        const Real_type O23 = symmetric ? DPA_d(qx,qy,qz,4,e) : DPA_d(qx,qy,qz,5,e);
///        const Real_type O31 = symmetric ? O13 : DPA_d(qx,qy,qz,6,e);
///        const Real_type O32 = symmetric ? O23 : DPA_d(qx,qy,qz,7,e);
///        const Real_type O33 = symmetric ? DPA_d(qx,qy,qz,5,e) : DPA_d(qx,qy,qz,8,e);
///        const Real_type gX = u;
///        const Real_type gY = v;
///        const Real_type gZ = w;
///        QQQ0[qz][qy][qx] = (O11*gX) + (O12*gY) + (O13*gZ);
///        QQQ1[qz][qy][qx] = (O21*gX) + (O22*gY) + (O23*gZ);
///        QQQ2[qz][qy][qx] = (O31*gX) + (O32*gY) + (O33*gZ);
///      }
///    }
///  }
///
///  for(Index_type dy=0; dy<diff::D1D; ++dy)
///  {
///    for(Index_type qx=0; qx<diff::Q1D; ++qx)
///    {
///      Bt[dy][qx] = DPA_b(qx,dy);
///      Gt[dy][qx] = DPA_g(qx,dy);
///    }
///  }
///
///  for(Index_type qz=0; qz<diff::Q1D; ++qz)
///  {
///    for(Index_type qy=0; qy<diff::Q1D; ++qy)
///    {
///      for(Index_type dx=0; dx<diff::D1D; ++dx)
///      {
///        Real_type u = 0.0, v = 0.0, w = 0.0;
///        RAJAPERF_UNROLL(MQ1)
///        for (Index_type qx = 0; qx < diff::Q1D; ++qx)
///        {
///          u += QQQ0[qz][qy][qx] * Gt[dx][qx];
///          v += QQQ1[qz][qy][qx] * Bt[dx][qx];
///          w += QQQ2[qz][qy][qx] * Bt[dx][qx];
///        }
///        QQD0[qz][qy][dx] = u;
///        QQD1[qz][qy][dx] = v;
///        QQD2[qz][qy][dx] = w;
///      }
///    }
///  }
///
///  for(Index_type qz=0; qz<diff::Q1D; ++qz)
///  {
///    for(Index_type dy=0; dy<diff::D1D; ++dy)
///    {
///      for(Index_type dx=0; dx<diff::D1D; ++dx)
///      {
///        Real_type u = 0.0, v = 0.0, w = 0.0;
///        RAJAPERF_UNROLL(diff::Q1D)
///        for (Index_type qy = 0; qy < diff::Q1D; ++qy)
///        {
///          u += QQD0[qz][qy][dx] * Bt[dy][qy];
///          v += QQD1[qz][qy][dx] * Gt[dy][qy];
///          w += QQD2[qz][qy][dx] * Bt[dy][qy];
///        }
///        QDD0[qz][dy][dx] = u;
///        QDD1[qz][dy][dx] = v;
///        QDD2[qz][dy][dx] = w;
///      }
///    }
///  }
///
///  for(Index_type dz=0; dz<diff::D1D; ++dz)
///  {
///    for(Index_type dy=0; dy<diff::D1D; ++dy)
///    {
///      for(Index_type dx=0; dx<diff::D1D; ++dx)
///      {
///        Real_type u = 0.0, v = 0.0, w = 0.0;
///        RAJAPERF_UNROLL(MQ1)
///        for (Index_type qz = 0; qz < diff::Q1D; ++qz)
///        {
///          u += QDD0[qz][dy][dx] * Bt[dz][qz];
///          v += QDD1[qz][dy][dx] * Bt[dz][qz];
///          w += QDD2[qz][dy][dx] * Gt[dz][qz];
///        }
///        DPA_Y(dx,dy,dz,e) += (u + v + w);
///       }
///     }
///   }
///
/// } // element loop
///
//clang-format on

#ifndef RAJAPerf_Apps_DIFFUSION3DPA_HPP
#define RAJAPerf_Apps_DIFFUSION3DPA_HPP

#define DIFFUSION3DPA_DATA_SETUP                                               \
  Real_ptr Basis = m_B;                                                        \
  Real_ptr dBasis = m_G;                                                       \
  Real_ptr D = m_D;                                                            \
  Real_ptr X = m_X;                                                            \
  Real_ptr Y = m_Y;                                                            \
  Index_type NE = m_NE;                                                        \
  const bool symmetric = true;

#include "FEM_MACROS.hpp"
#include "common/KernelBase.hpp"

#include "RAJA/RAJA.hpp"

// Number of Dofs/Qpts in 1D
namespace diff {
constexpr RAJA::Index_type D1D = 3;
constexpr RAJA::Index_type Q1D = 4;
constexpr RAJA::Index_type DPA_SYM = 6;
} // namespace diff
#define DPA_b(x, y) Basis[x + diff::Q1D * y]
#define DPA_g(x, y) dBasis[x + diff::Q1D * y]
#define DPA_X(dx, dy, dz, e)                                                   \
  X[dx + diff::D1D * dy + diff::D1D * diff::D1D * dz +                         \
    diff::D1D * diff::D1D * diff::D1D * e]
#define DPA_Y(dx, dy, dz, e)                                                   \
  Y[dx + diff::D1D * dy + diff::D1D * diff::D1D * dz +                         \
    diff::D1D * diff::D1D * diff::D1D * e]
#define DPA_d(qx, qy, qz, s, e)                                                \
  D[qx + diff::Q1D * qy + diff::Q1D * diff::Q1D * qz +                         \
    diff::Q1D * diff::Q1D * diff::Q1D * s +                                    \
    diff::Q1D * diff::Q1D * diff::Q1D * diff::DPA_SYM * e]

// Half of B and G are stored in shared to get B, Bt, G and Gt.
// Indices computation for SmemPADiffusionApply3D.
#define DPA_qi(q, d, Q) (((q) <= (d)) ? (q) : (Q) - 1 - (q))
#define DPA_dj(q, d, D) (((q) <= (d)) ? (d) : (D) - 1 - (d))
#define DPA_qk(q, d, Q) (((q) <= (d)) ? (Q) - 1 - (q) : (q))
#define DPA_dl(q, d, D) (((q) <= (d)) ? (D) - 1 - (d) : (d))
#define DPA_sign(q, d) (((q) <= (d)) ? -1.0 : 1.0)

#define DIFFUSION3DPA_0_GPU                                                    \
  constexpr Index_type MQ1 = diff::Q1D;                                        \
  constexpr Index_type MD1 = diff::D1D;                                        \
  constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;                          \
  RAJA_TEAM_SHARED Real_type sBG[MQ1 * MD1];                                   \
  Real_type(*B)[MD1] = (Real_type(*)[MD1])sBG;                                 \
  Real_type(*G)[MD1] = (Real_type(*)[MD1])sBG;                                 \
  Real_type(*Bt)[MQ1] = (Real_type(*)[MQ1])sBG;                                \
  Real_type(*Gt)[MQ1] = (Real_type(*)[MQ1])sBG;                                \
  RAJA_TEAM_SHARED Real_type sm0[3][MDQ * MDQ * MDQ];                          \
  RAJA_TEAM_SHARED Real_type sm1[3][MDQ * MDQ * MDQ];                          \
  Real_type(*s_X)[MD1][MD1] = (Real_type(*)[MD1][MD1])(sm0 + 2);               \
  Real_type(*DDQ0)[MD1][MQ1] = (Real_type(*)[MD1][MQ1])(sm0 + 0);              \
  Real_type(*DDQ1)[MD1][MQ1] = (Real_type(*)[MD1][MQ1])(sm0 + 1);              \
  Real_type(*DQQ0)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])(sm1 + 0);              \
  Real_type(*DQQ1)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])(sm1 + 1);              \
  Real_type(*DQQ2)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])(sm1 + 2);              \
  Real_type(*QQQ0)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])(sm0 + 0);              \
  Real_type(*QQQ1)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])(sm0 + 1);              \
  Real_type(*QQQ2)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])(sm0 + 2);              \
  Real_type(*QQD0)[MQ1][MD1] = (Real_type(*)[MQ1][MD1])(sm1 + 0);              \
  Real_type(*QQD1)[MQ1][MD1] = (Real_type(*)[MQ1][MD1])(sm1 + 1);              \
  Real_type(*QQD2)[MQ1][MD1] = (Real_type(*)[MQ1][MD1])(sm1 + 2);              \
  Real_type(*QDD0)[MD1][MD1] = (Real_type(*)[MD1][MD1])(sm0 + 0);              \
  Real_type(*QDD1)[MD1][MD1] = (Real_type(*)[MD1][MD1])(sm0 + 1);              \
  Real_type(*QDD2)[MD1][MD1] = (Real_type(*)[MD1][MD1])(sm0 + 2);

#define DIFFUSION3DPA_0_CPU                                                    \
  constexpr Index_type MQ1 = diff::Q1D;                                        \
  constexpr Index_type MD1 = diff::D1D;                                        \
  constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;                          \
  Real_type sBG[MQ1 * MD1];                                                    \
  Real_type(*B)[MD1] = (Real_type(*)[MD1])sBG;                                 \
  Real_type(*G)[MD1] = (Real_type(*)[MD1])sBG;                                 \
  Real_type(*Bt)[MQ1] = (Real_type(*)[MQ1])sBG;                                \
  Real_type(*Gt)[MQ1] = (Real_type(*)[MQ1])sBG;                                \
  Real_type sm0[3][MDQ * MDQ * MDQ];                                           \
  Real_type sm1[3][MDQ * MDQ * MDQ];                                           \
  Real_type(*s_X)[MD1][MD1] = (Real_type(*)[MD1][MD1])(sm0 + 2);               \
  Real_type(*DDQ0)[MD1][MQ1] = (Real_type(*)[MD1][MQ1])(sm0 + 0);              \
  Real_type(*DDQ1)[MD1][MQ1] = (Real_type(*)[MD1][MQ1])(sm0 + 1);              \
  Real_type(*DQQ0)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])(sm1 + 0);              \
  Real_type(*DQQ1)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])(sm1 + 1);              \
  Real_type(*DQQ2)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])(sm1 + 2);              \
  Real_type(*QQQ0)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])(sm0 + 0);              \
  Real_type(*QQQ1)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])(sm0 + 1);              \
  Real_type(*QQQ2)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])(sm0 + 2);              \
  Real_type(*QQD0)[MQ1][MD1] = (Real_type(*)[MQ1][MD1])(sm1 + 0);              \
  Real_type(*QQD1)[MQ1][MD1] = (Real_type(*)[MQ1][MD1])(sm1 + 1);              \
  Real_type(*QQD2)[MQ1][MD1] = (Real_type(*)[MQ1][MD1])(sm1 + 2);              \
  Real_type(*QDD0)[MD1][MD1] = (Real_type(*)[MD1][MD1])(sm0 + 0);              \
  Real_type(*QDD1)[MD1][MD1] = (Real_type(*)[MD1][MD1])(sm0 + 1);              \
  Real_type(*QDD2)[MD1][MD1] = (Real_type(*)[MD1][MD1])(sm0 + 2);

#define DIFFUSION3DPA_1 s_X[dz][dy][dx] = DPA_X(dx, dy, dz, e);

#define DIFFUSION3DPA_2                                                        \
  B[qx][dy] = DPA_b(qx, dy);                                                   \
  G[qx][dy] = DPA_g(qx, dy);

#define DIFFUSION3DPA_3                                                        \
  Real_type u = 0.0, v = 0.0;                                                  \
  RAJAPERF_UNROLL(MD1)                                                         \
  for (Index_type dx = 0; dx < diff::D1D; ++dx) {                              \
    const Real_type coords = s_X[dz][dy][dx];                                  \
    u += coords * B[qx][dx];                                                   \
    v += coords * G[qx][dx];                                                   \
  }                                                                            \
  DDQ0[dz][dy][qx] = u;                                                        \
  DDQ1[dz][dy][qx] = v;

#define DIFFUSION3DPA_4                                                        \
  Real_type u = 0.0, v = 0.0, w = 0.0;                                         \
  RAJAPERF_UNROLL(MD1)                                                         \
  for (Index_type dy = 0; dy < diff::D1D; ++dy) {                              \
    u += DDQ1[dz][dy][qx] * B[qy][dy];                                         \
    v += DDQ0[dz][dy][qx] * G[qy][dy];                                         \
    w += DDQ0[dz][dy][qx] * B[qy][dy];                                         \
  }                                                                            \
  DQQ0[dz][qy][qx] = u;                                                        \
  DQQ1[dz][qy][qx] = v;                                                        \
  DQQ2[dz][qy][qx] = w;

#define DIFFUSION3DPA_5                                                        \
  Real_type u = 0.0, v = 0.0, w = 0.0;                                         \
  RAJAPERF_UNROLL(MD1)                                                         \
  for (Index_type dz = 0; dz < diff::D1D; ++dz) {                              \
    u += DQQ0[dz][qy][qx] * B[qz][dz];                                         \
    v += DQQ1[dz][qy][qx] * B[qz][dz];                                         \
    w += DQQ2[dz][qy][qx] * G[qz][dz];                                         \
  }                                                                            \
  const Real_type O11 = DPA_d(qx, qy, qz, 0, e);                               \
  const Real_type O12 = DPA_d(qx, qy, qz, 1, e);                               \
  const Real_type O13 = DPA_d(qx, qy, qz, 2, e);                               \
  const Real_type O21 = symmetric ? O12 : DPA_d(qx, qy, qz, 3, e);             \
  const Real_type O22 =                                                        \
      symmetric ? DPA_d(qx, qy, qz, 3, e) : DPA_d(qx, qy, qz, 4, e);           \
  const Real_type O23 =                                                        \
      symmetric ? DPA_d(qx, qy, qz, 4, e) : DPA_d(qx, qy, qz, 5, e);           \
  const Real_type O31 = symmetric ? O13 : DPA_d(qx, qy, qz, 6, e);             \
  const Real_type O32 = symmetric ? O23 : DPA_d(qx, qy, qz, 7, e);             \
  const Real_type O33 =                                                        \
      symmetric ? DPA_d(qx, qy, qz, 5, e) : DPA_d(qx, qy, qz, 8, e);           \
  const Real_type gX = u;                                                      \
  const Real_type gY = v;                                                      \
  const Real_type gZ = w;                                                      \
  QQQ0[qz][qy][qx] = (O11 * gX) + (O12 * gY) + (O13 * gZ);                     \
  QQQ1[qz][qy][qx] = (O21 * gX) + (O22 * gY) + (O23 * gZ);                     \
  QQQ2[qz][qy][qx] = (O31 * gX) + (O32 * gY) + (O33 * gZ);

#define DIFFUSION3DPA_6                                                        \
  Bt[dy][qx] = DPA_b(qx, dy);                                                  \
  Gt[dy][qx] = DPA_g(qx, dy);

#define DIFFUSION3DPA_7                                                        \
  Real_type u = 0.0, v = 0.0, w = 0.0;                                         \
  RAJAPERF_UNROLL(MQ1)                                                         \
  for (Index_type qx = 0; qx < diff::Q1D; ++qx) {                              \
    u += QQQ0[qz][qy][qx] * Gt[dx][qx];                                        \
    v += QQQ1[qz][qy][qx] * Bt[dx][qx];                                        \
    w += QQQ2[qz][qy][qx] * Bt[dx][qx];                                        \
  }                                                                            \
  QQD0[qz][qy][dx] = u;                                                        \
  QQD1[qz][qy][dx] = v;                                                        \
  QQD2[qz][qy][dx] = w;

#define DIFFUSION3DPA_8                                                        \
  Real_type u = 0.0, v = 0.0, w = 0.0;                                         \
  RAJAPERF_UNROLL(diff::Q1D)                                                   \
  for (Index_type qy = 0; qy < diff::Q1D; ++qy) {                              \
    u += QQD0[qz][qy][dx] * Bt[dy][qy];                                        \
    v += QQD1[qz][qy][dx] * Gt[dy][qy];                                        \
    w += QQD2[qz][qy][dx] * Bt[dy][qy];                                        \
  }                                                                            \
  QDD0[qz][dy][dx] = u;                                                        \
  QDD1[qz][dy][dx] = v;                                                        \
  QDD2[qz][dy][dx] = w;

#define DIFFUSION3DPA_9                                                        \
  Real_type u = 0.0, v = 0.0, w = 0.0;                                         \
  RAJAPERF_UNROLL(MQ1)                                                         \
  for (Index_type qz = 0; qz < diff::Q1D; ++qz) {                              \
    u += QDD0[qz][dy][dx] * Bt[dz][qz];                                        \
    v += QDD1[qz][dy][dx] * Bt[dz][qz];                                        \
    w += QDD2[qz][dy][dx] * Gt[dz][qz];                                        \
  }                                                                            \
  DPA_Y(dx, dy, dz, e) += (u + v + w);

//
// CPU SoA variant: batch of FEM_VL elements starting at e0, element innermost.
// Stages 2 and 6 (basis loads) are element independent and reused as is.
//
#define DIFFUSION3DPA_0_CPU_SIMD                                               \
  constexpr Index_type MQ1 = diff::Q1D;                                        \
  constexpr Index_type MD1 = diff::D1D;                                        \
  constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;                          \
  constexpr Index_type VL = FEM_VL;                                            \
  FEM_BATCH_SETUP(e0, NE);                                                     \
  Real_type sBG[MQ1 * MD1];                                                    \
  Real_type(*B)[MD1] = (Real_type(*)[MD1])sBG;                                 \
  Real_type(*G)[MD1] = (Real_type(*)[MD1])sBG;                                 \
  Real_type(*Bt)[MQ1] = (Real_type(*)[MQ1])sBG;                                \
  Real_type(*Gt)[MQ1] = (Real_type(*)[MQ1])sBG;                                \
  Real_type sm0[3][MDQ * MDQ * MDQ][VL];                                       \
  Real_type sm1[3][MDQ * MDQ * MDQ][VL];                                       \
  Real_type(*s_X)[MD1][MD1][VL] = (Real_type(*)[MD1][MD1][VL])(sm0 + 2);       \
  Real_type(*DDQ0)[MD1][MQ1][VL] = (Real_type(*)[MD1][MQ1][VL])(sm0 + 0);      \
  Real_type(*DDQ1)[MD1][MQ1][VL] = (Real_type(*)[MD1][MQ1][VL])(sm0 + 1);      \
  Real_type(*DQQ0)[MQ1][MQ1][VL] = (Real_type(*)[MQ1][MQ1][VL])(sm1 + 0);      \
  Real_type(*DQQ1)[MQ1][MQ1][VL] = (Real_type(*)[MQ1][MQ1][VL])(sm1 + 1);      \
  Real_type(*DQQ2)[MQ1][MQ1][VL] = (Real_type(*)[MQ1][MQ1][VL])(sm1 + 2);      \
  Real_type(*QQQ0)[MQ1][MQ1][VL] = (Real_type(*)[MQ1][MQ1][VL])(sm0 + 0);      \
  Real_type(*QQQ1)[MQ1][MQ1][VL] = (Real_type(*)[MQ1][MQ1][VL])(sm0 + 1);      \
  Real_type(*QQQ2)[MQ1][MQ1][VL] = (Real_type(*)[MQ1][MQ1][VL])(sm0 + 2);      \
  Real_type(*QQD0)[MQ1][MD1][VL] = (Real_type(*)[MQ1][MD1][VL])(sm1 + 0);      \
  Real_type(*QQD1)[MQ1][MD1][VL] = (Real_type(*)[MQ1][MD1][VL])(sm1 + 1);      \
  Real_type(*QQD2)[MQ1][MD1][VL] = (Real_type(*)[MQ1][MD1][VL])(sm1 + 2);      \
  Real_type(*QDD0)[MD1][MD1][VL] = (Real_type(*)[MD1][MD1][VL])(sm0 + 0);      \
  Real_type(*QDD1)[MD1][MD1][VL] = (Real_type(*)[MD1][MD1][VL])(sm0 + 1);      \
  Real_type(*QDD2)[MD1][MD1][VL] = (Real_type(*)[MD1][MD1][VL])(sm0 + 2);

// gather AoS -> SoA
#define DIFFUSION3DPA_SIMD_1                                                   \
  CPU_FOREACH_LANE(l) {                                                        \
    s_X[dz][dy][dx][l] = DPA_X(dx, dy, dz, le[l]);                             \
  }

#define DIFFUSION3DPA_SIMD_3                                                   \
  CPU_FOREACH_LANE(l) {                                                        \
    Real_type u = 0.0, v = 0.0;                                                \
    for (Index_type dx = 0; dx < diff::D1D; ++dx) {                            \
      const Real_type coords = s_X[dz][dy][dx][l];                             \
      u += coords * B[qx][dx];                                                 \
      v += coords * G[qx][dx];                                                 \
    }                                                                          \
    DDQ0[dz][dy][qx][l] = u;                                                   \
    DDQ1[dz][dy][qx][l] = v;                                                   \
  }

#define DIFFUSION3DPA_SIMD_4                                                   \
  CPU_FOREACH_LANE(l) {                                                        \
    Real_type u = 0.0, v = 0.0, w = 0.0;                                       \
    for (Index_type dy = 0; dy < diff::D1D; ++dy) {                            \
      u += DDQ1[dz][dy][qx][l] * B[qy][dy];                                    \
      v += DDQ0[dz][dy][qx][l] * G[qy][dy];                                    \
      w += DDQ0[dz][dy][qx][l] * B[qy][dy];                                    \
    }                                                                          \
    DQQ0[dz][qy][qx][l] = u;                                                   \
    DQQ1[dz][qy][qx][l] = v;                                                   \
    DQQ2[dz][qy][qx][l] = w;                                                   \
  }

#define DIFFUSION3DPA_SIMD_5                                                   \
  CPU_FOREACH_LANE(l) {                                                        \
    Real_type u = 0.0, v = 0.0, w = 0.0;                                       \
    for (Index_type dz = 0; dz < diff::D1D; ++dz) {                            \
      u += DQQ0[dz][qy][qx][l] * B[qz][dz];                                    \
      v += DQQ1[dz][qy][qx][l] * B[qz][dz];                                    \
      w += DQQ2[dz][qy][qx][l] * G[qz][dz];                                    \
    }                                                                          \
    const Index_type e = le[l];                                                \
    const Real_type O11 = DPA_d(qx, qy, qz, 0, e);                             \
    const Real_type O12 = DPA_d(qx, qy, qz, 1, e);                             \
    const Real_type O13 = DPA_d(qx, qy, qz, 2, e);                             \
    const Real_type O21 = symmetric ? O12 : DPA_d(qx, qy, qz, 3, e);           \
    const Real_type O22 =                                                      \
        symmetric ? DPA_d(qx, qy, qz, 3, e) : DPA_d(qx, qy, qz, 4, e);         \
    const Real_type O23 =                                                      \
        symmetric ? DPA_d(qx, qy, qz, 4, e) : DPA_d(qx, qy, qz, 5, e);         \
    const Real_type O31 = symmetric ? O13 : DPA_d(qx, qy, qz, 6, e);           \
    const Real_type O32 = symmetric ? O23 : DPA_d(qx, qy, qz, 7, e);           \
    const Real_type O33 =                                                      \
        symmetric ? DPA_d(qx, qy, qz, 5, e) : DPA_d(qx, qy, qz, 8, e);         \
    QQQ0[qz][qy][qx][l] = (O11 * u) + (O12 * v) + (O13 * w);                   \
    QQQ1[qz][qy][qx][l] = (O21 * u) + (O22 * v) + (O23 * w);                   \
    QQQ2[qz][qy][qx][l] = (O31 * u) + (O32 * v) + (O33 * w);                   \
  }

#define DIFFUSION3DPA_SIMD_7                                                   \
  CPU_FOREACH_LANE(l) {                                                        \
    Real_type u = 0.0, v = 0.0, w = 0.0;                                       \
    for (Index_type qx = 0; qx < diff::Q1D; ++qx) {                            \
      u += QQQ0[qz][qy][qx][l] * Gt[dx][qx];                                   \
      v += QQQ1[qz][qy][qx][l] * Bt[dx][qx];                                   \
      w += QQQ2[qz][qy][qx][l] * Bt[dx][qx];                                   \
    }                                                                          \
    QQD0[qz][qy][dx][l] = u;                                                   \
    QQD1[qz][qy][dx][l] = v;                                                   \
    QQD2[qz][qy][dx][l] = w;                                                   \
  }

#define DIFFUSION3DPA_SIMD_8                                                   \
  CPU_FOREACH_LANE(l) {                                                        \
    Real_type u = 0.0, v = 0.0, w = 0.0;                                       \
    for (Index_type qy = 0; qy < diff::Q1D; ++qy) {                            \
      u += QQD0[qz][qy][dx][l] * Bt[dy][qy];                                   \
      v += QQD1[qz][qy][dx][l] * Gt[dy][qy];                                   \
      w += QQD2[qz][qy][dx][l] * Bt[dy][qy];                                   \
    }                                                                          \
    QDD0[qz][dy][dx][l] = u;                                                   \
    QDD1[qz][dy][dx][l] = v;                                                   \
    QDD2[qz][dy][dx][l] = w;                                                   \
  }

// scatter SoA -> AoS (valid lanes only)
#define DIFFUSION3DPA_SIMD_9                                                   \
  CPU_FOREACH_LANE(l) {                                                        \
    Real_type u = 0.0, v = 0.0, w = 0.0;                                       \
    for (Index_type qz = 0; qz < diff::Q1D; ++qz) {                            \
      u += QDD0[qz][dy][dx][l] * Bt[dz][qz];                                   \
      v += QDD1[qz][dy][dx][l] * Bt[dz][qz];                                   \
      w += QDD2[qz][dy][dx][l] * Gt[dz][qz];                                   \
    }                                                                          \
    if (l < nel) {                                                             \
      const Index_type e = e0 + l;                                             \
      DPA_Y(dx, dy, dz, e) += (u + v + w);                                     \
    }                                                                          \
  }

#include "FEM_SIMD.hpp"

#if defined(RAJAPERF_FEM_AVX512)
//
// AVX-512 batch of FEM_VL elements [e0, e0+nel), stages 1-9 of the CPU SoA
// variant: X, D and Y move between AoS and SoA through 8x8 in-register
// transposes of each element's contiguous blocks instead of gathers and
// scatters. The next batch's D, X and Y are prefetched (DRAM bound).
//
RAJAPERF_FEM_NOINLINE void diffusion3dpa_batch_avx512(
    rajaperf::Real_const_ptr Basis, rajaperf::Real_const_ptr dBasis,
    rajaperf::Real_const_ptr D, rajaperf::Real_const_ptr X,
    rajaperf::Real_ptr Y, const rajaperf::Index_type e0,
    const rajaperf::Index_type nel, const rajaperf::Index_type NE)
{
  using namespace rajaperf::fem_simd;
  using rajaperf::Real_type;
  using rajaperf::Index_type;
  constexpr Index_type D1 = diff::D1D;
  constexpr Index_type Q1 = diff::Q1D;
  constexpr Index_type ND = D1 * D1 * D1;
  constexpr Index_type NQ = Q1 * Q1 * Q1;
  constexpr Index_type NS = diff::DPA_SYM;

  if (e0 + 2 * FEM_VL <= NE) {
    const char *const pd = reinterpret_cast<const char *>(D + NQ * NS * (e0 + FEM_VL));
    for (Index_type o = 0; o < NQ * NS * FEM_VL * 8; o += 64) {
      __builtin_prefetch(pd + o, 0, 3);
    }
    const char *const px = reinterpret_cast<const char *>(X + ND * (e0 + FEM_VL));
    const char *const py = reinterpret_cast<const char *>(Y + ND * (e0 + FEM_VL));
    for (Index_type o = 0; o < ND * FEM_VL * 8; o += 64) {
      __builtin_prefetch(px + o, 0, 3);
      __builtin_prefetch(py + o, 1, 3);
    }
  }

  Real_type B[Q1][D1], G[Q1][D1];
  for (Index_type d = 0; d < D1; ++d) {
    for (Index_type q = 0; q < Q1; ++q) {
      B[q][d] = DPA_b(q, d);
      G[q][d] = DPA_g(q, d);
    }
  }

  const Real_type *xrow[8], *drow[8];
  Real_type *yrow[8];
  for (Index_type l = 0; l < 8; ++l) {
    const Index_type e = e0 + ((l < nel) ? l : nel - 1);
    xrow[l] = X + ND * e;
    yrow[l] = Y + ND * e;
    drow[l] = D + NQ * NS * e;
  }

  alignas(64) vd sX[D1][D1][D1];
  RAJAPERF_FEM_STACK_ALIGN(sX);
  load_transpose<ND>(xrow, &sX[0][0][0]);

  vd DDQ0[D1][D1][Q1], DDQ1[D1][D1][Q1];
  for (Index_type dz = 0; dz < D1; ++dz) {
    for (Index_type dy = 0; dy < D1; ++dy) {
      for (Index_type qx = 0; qx < Q1; ++qx) {
        vd u = vzero(), v = vzero();
        for (Index_type dx = 0; dx < D1; ++dx) {
          u = vfma(sX[dz][dy][dx], vset1(B[qx][dx]), u);
          v = vfma(sX[dz][dy][dx], vset1(G[qx][dx]), v);
        }
        DDQ0[dz][dy][qx] = u;
        DDQ1[dz][dy][qx] = v;
      }
    }
  }

  vd DQQ0[D1][Q1][Q1], DQQ1[D1][Q1][Q1], DQQ2[D1][Q1][Q1];
  for (Index_type dz = 0; dz < D1; ++dz) {
    for (Index_type qy = 0; qy < Q1; ++qy) {
      for (Index_type qx = 0; qx < Q1; ++qx) {
        vd u = vzero(), v = vzero(), w = vzero();
        for (Index_type dy = 0; dy < D1; ++dy) {
          u = vfma(DDQ1[dz][dy][qx], vset1(B[qy][dy]), u);
          v = vfma(DDQ0[dz][dy][qx], vset1(G[qy][dy]), v);
          w = vfma(DDQ0[dz][dy][qx], vset1(B[qy][dy]), w);
        }
        DQQ0[dz][qy][qx] = u;
        DQQ1[dz][qy][qx] = v;
        DQQ2[dz][qy][qx] = w;
      }
    }
  }

  // symmetric D: O11 O12 O13 O22 O23 O33 per quadrature point
  vd QQQ0[Q1][Q1][Q1], QQQ1[Q1][Q1][Q1], QQQ2[Q1][Q1][Q1];
  for (Index_type qz = 0; qz < Q1; ++qz) {
    vd Os[NS][Q1 * Q1];
    for (Index_type s = 0; s < NS; ++s) {
      const Real_type *srow[8];
      for (Index_type l = 0; l < 8; ++l) {
        srow[l] = drow[l] + NQ * s + Q1 * Q1 * qz;
      }
      load_transpose<Q1 * Q1>(srow, Os[s]);
    }
    for (Index_type qy = 0; qy < Q1; ++qy) {
      for (Index_type qx = 0; qx < Q1; ++qx) {
        vd u = vzero(), v = vzero(), w = vzero();
        for (Index_type dz = 0; dz < D1; ++dz) {
          u = vfma(DQQ0[dz][qy][qx], vset1(B[qz][dz]), u);
          v = vfma(DQQ1[dz][qy][qx], vset1(B[qz][dz]), v);
          w = vfma(DQQ2[dz][qy][qx], vset1(G[qz][dz]), w);
        }
        const Index_type q = qx + Q1 * qy;
        QQQ0[qz][qy][qx] = vfma(Os[2][q], w, vfma(Os[1][q], v, vmul(Os[0][q], u)));
        QQQ1[qz][qy][qx] = vfma(Os[4][q], w, vfma(Os[3][q], v, vmul(Os[1][q], u)));
        QQQ2[qz][qy][qx] = vfma(Os[5][q], w, vfma(Os[4][q], v, vmul(Os[2][q], u)));
      }
    }
  }

  vd QQD0[Q1][Q1][D1], QQD1[Q1][Q1][D1], QQD2[Q1][Q1][D1];
  for (Index_type qz = 0; qz < Q1; ++qz) {
    for (Index_type qy = 0; qy < Q1; ++qy) {
      for (Index_type dx = 0; dx < D1; ++dx) {
        vd u = vzero(), v = vzero(), w = vzero();
        for (Index_type qx = 0; qx < Q1; ++qx) {
          u = vfma(QQQ0[qz][qy][qx], vset1(G[qx][dx]), u);
          v = vfma(QQQ1[qz][qy][qx], vset1(B[qx][dx]), v);
          w = vfma(QQQ2[qz][qy][qx], vset1(B[qx][dx]), w);
        }
        QQD0[qz][qy][dx] = u;
        QQD1[qz][qy][dx] = v;
        QQD2[qz][qy][dx] = w;
      }
    }
  }

  vd QDD0[Q1][D1][D1], QDD1[Q1][D1][D1], QDD2[Q1][D1][D1];
  for (Index_type qz = 0; qz < Q1; ++qz) {
    for (Index_type dy = 0; dy < D1; ++dy) {
      for (Index_type dx = 0; dx < D1; ++dx) {
        vd u = vzero(), v = vzero(), w = vzero();
        for (Index_type qy = 0; qy < Q1; ++qy) {
          u = vfma(QQD0[qz][qy][dx], vset1(B[qy][dy]), u);
          v = vfma(QQD1[qz][qy][dx], vset1(G[qy][dy]), v);
          w = vfma(QQD2[qz][qy][dx], vset1(B[qy][dy]), w);
        }
        QDD0[qz][dy][dx] = u;
        QDD1[qz][dy][dx] = v;
        QDD2[qz][dy][dx] = w;
      }
    }
  }

  alignas(64) vd y[D1][D1][D1];
  for (Index_type dz = 0; dz < D1; ++dz) {
    for (Index_type dy = 0; dy < D1; ++dy) {
      for (Index_type dx = 0; dx < D1; ++dx) {
        vd u = vzero(), v = vzero(), w = vzero();
        for (Index_type qz = 0; qz < Q1; ++qz) {
          u = vfma(QDD0[qz][dy][dx], vset1(B[qz][dz]), u);
          v = vfma(QDD1[qz][dy][dx], vset1(B[qz][dz]), v);
          w = vfma(QDD2[qz][dy][dx], vset1(G[qz][dz]), w);
        }
        y[dz][dy][dx] = vadd(vadd(u, v), w);
      }
    }
  }
  transpose_store<ND, true>(yrow, nel, &y[0][0][0]);
}
#endif

namespace rajaperf {
class RunParams;

namespace apps {

class DIFFUSION3DPA : public KernelBase {
public:
  DIFFUSION3DPA(const RunParams &params);

  ~DIFFUSION3DPA();

  void setSize(Index_type target_size, Index_type target_reps);
  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void defineSeqVariantTunings();
  void defineOpenMPVariantTunings();
  void defineCudaVariantTunings();
  void defineHipVariantTunings();
  void defineSyclVariantTunings();

  void runSeqVariant(VariantID vid);
  template <size_t tune_idx>
  void runOpenMPVariant(VariantID vid);

  template <size_t block_size> void runCudaVariantImpl(VariantID vid);
  template <size_t block_size> void runHipVariantImpl(VariantID vid);
  template <size_t work_group_size> void runSyclVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size =
      diff::Q1D * diff::Q1D * diff::Q1D;
  using gpu_block_sizes_type = integer::list_type<default_gpu_block_size>;

  Real_ptr m_B;
  Real_ptr m_Bt;
  Real_ptr m_G;
  Real_ptr m_Gt;
  Real_ptr m_D;
  Real_ptr m_X;
  Real_ptr m_Y;

  Index_type m_NE;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
