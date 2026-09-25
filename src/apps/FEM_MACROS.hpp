//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#ifndef RAJAPerf_FEM_MACROS_HPP
#define RAJAPerf_FEM_MACROS_HPP

#include "RAJA/RAJA.hpp"

#include "common/RPTypes.hpp"

#if defined(USE_RAJAPERF_UNROLL)
// If enabled uses RAJA's RAJA_UNROLL_COUNT which is always on
#define RAJAPERF_UNROLL(N) RAJA_UNROLL_COUNT(N)
#else
#define RAJAPERF_UNROLL(N)
#endif

// Need two different host/device macros due to
// how hipcc/clang works.
// See note in MAT_MAT_SHARED regarding hipcc/clang
// builds.
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
#define GPU_FOREACH_THREAD(i, k, N)                                            \
  for (Index_type i = threadIdx.k; i < N; i += blockDim.k)

#define GPU_FOREACH_THREAD_INC(i, k, N, INC)                                   \
  for (Index_type i = threadIdx.k; i < N; i += INC)


#define GPU_FOREACH_THREAD_DIRECT(i, k, N)                                     \
  if (const Index_type i = threadIdx.k; i < N)
#endif

#if defined(RAJA_ENABLE_SYCL)
#define SYCL_FOREACH_THREAD(i, k, N)                                           \
  for (Index_type i = itm.get_local_id(k); i < N; i += itm.get_local_range(k))
#define SYCL_FOREACH_THREAD_DIRECT(i, k, N)                                    \
  if (const Index_type i = itm.get_local_id(k); i < N)
#endif

#if defined(RAJA_ENABLE_SYCL)
#define SYCL_SHARED_LOOP_2D(tx, ty, Nx, Ny)                                    \
  if (itm.get_local_id(0) < 1)                                                 \
    for (Index_type ty = itm.get_local_id(1); ty < Ny;                         \
         ty += itm.get_local_range(1))                                         \
      for (Index_type tx = itm.get_local_id(2); tx < Nx;                       \
           tx += itm.get_local_range(2))

#define SYCL_SHARED_LOOP_3D(tx, ty, tz, Nx, Ny, Nz)                            \
  for (Index_type tz = itm.get_local_id(0); tz < Nz;                           \
       tz += itm.get_local_range(0))                                           \
    for (Index_type ty = itm.get_local_id(1); ty < Ny;                         \
         ty += itm.get_local_range(1))                                         \
      for (Index_type tx = itm.get_local_id(2); tx < Nx;                       \
           tx += itm.get_local_range(2))

#endif

#define CPU_FOREACH(i, k, N) for (Index_type i = 0; i < N; i++)

// CPU SoA variants process FEM_VL elements per batch (one 512-bit vector),
// element (lane) index innermost.
constexpr RAJA::Index_type FEM_VL = 64 / sizeof(rajaperf::Real_type);

#if defined(_OPENMP) && defined(__clang__)
// keep clang from fully unrolling the short lane loop before it vectorizes
#define FEM_OMP_SIMD _Pragma("omp simd") _Pragma("clang loop unroll(disable)")
#elif defined(_OPENMP)
#define FEM_OMP_SIMD _Pragma("omp simd")
#else
#define FEM_OMP_SIMD
#endif

#define CPU_FOREACH_LANE(l) FEM_OMP_SIMD for (Index_type l = 0; l < FEM_VL; l++)

// Lane -> element map for batch [e0, e0+FEM_VL); tail lanes replicate the
// last valid element so all lanes compute on valid data.
#define FEM_BATCH_SETUP(e0, NE)                                                \
  const Index_type nel = ((NE) - (e0) < FEM_VL) ? (NE) - (e0) : FEM_VL;        \
  Index_type le[FEM_VL];                                                       \
  for (Index_type l = 0; l < FEM_VL; l++) {                                    \
    le[l] = (e0) + ((l < nel) ? l : nel - 1);                                  \
  }

#define SHARED_LOOP_2D(tx, ty, Nx, Ny)                                         \
  for (Index_type ty = 0; ty < Ny; ty++)                                       \
    for (Index_type tx = 0; tx < Nx; tx++)

#define SHARED_LOOP_3D(tx, ty, tz, Nx, Ny, Nz)                                 \
  for (Index_type tz = 0; tz < Nz; tz++)                                       \
    for (Index_type ty = 0; ty < Ny; ty++)                                     \
      for (Index_type tx = 0; tx < Nx; tx++)

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
#define GPU_SHARED_DIRECT_2D(tx, ty, Nx, Ny)                                   \
  if (threadIdx.z < 1)                                                         \
    if (const Index_type ty = threadIdx.y; ty < Ny)                            \
      if (const Index_type tx = threadIdx.x; tx < Nx)

#define GPU_SHARED_DIRECT_3D(tx, ty, tz, Nx, Ny, Nz)                           \
  if (const Index_type tz = threadIdx.z; tz < Nz)                              \
    if (const Index_type ty = threadIdx.y; ty < Ny)                            \
      if (const Index_type tx = threadIdx.x; tx < Nx)

#define GPU_SHARED_LOOP_2D(tx, ty, Nx, Ny)                                     \
  if (threadIdx.z < 1)                                                         \
    for (Index_type ty = threadIdx.y; ty < Ny; ty += blockDim.y)               \
      for (Index_type tx = threadIdx.x; tx < Nx; tx += blockDim.x)

#define GPU_SHARED_LOOP_3D(tx, ty, tz, Nx, Ny, Nz)                             \
  for (Index_type tz = threadIdx.z; tz < Nz; tz += blockDim.z)                 \
    for (Index_type ty = threadIdx.y; ty < Ny; ty += blockDim.y)               \
      for (Index_type tx = threadIdx.x; tx < Nx; tx += blockDim.x)

#define GPU_SHARED_LOOP_2D_INC(tx, ty, Nx, Ny, runtime_blocks_size)            \
  if (threadIdx.z < 1)                                                         \
    for (Index_type ty = threadIdx.y; ty < Ny; ty += runtime_blocks_size)      \
      for (Index_type tx = threadIdx.x; tx < Nx; tx += runtime_blocks_size)

#define GPU_SHARED_LOOP_3D_INC(tx, ty, tz, Nx, Ny, Nz, runtime_blocks_size)    \
  for (Index_type tz = threadIdx.z; tz < Nz; tz += runtime_blocks_size)        \
    for (Index_type ty = threadIdx.y; ty < Ny; ty += runtime_blocks_size)      \
      for (Index_type tx = threadIdx.x; tx < Nx; tx += runtime_blocks_size)

#define GPU_SHARED_2D_APPLY(SHARED_2D_MACRO, tx, ty, Nx, Ny)                   \
  SHARED_2D_MACRO(tx, ty, Nx, Ny)

#define GPU_SHARED_3D_APPLY(SHARED_3D_MACRO, tx, ty, tz, Nx, Ny, Nz)           \
  SHARED_3D_MACRO(tx, ty, tz, Nx, Ny, Nz)

#endif

#endif // closing endif for header file include guard
