
 
#ifndef GMX_SIMD_IMPL_NSIMD_SIMD4_FLOAT_DEFINED_H
#define GMX_SIMD_IMPL_NSIMD_SIMD4_FLOAT_DEFINED_H

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/cxx_adv_api_functions.hpp>
#include <nsimd/nsimd.h>

#include "impl_nsimd_simd4_double.h"
#include "gromacs/math/utilities.h"

#if (defined(NSIMD_AVX2) || defined(NSIMD_AVX))

static inline double gmx_simdcall
dotProduct(Simd4Double a, Simd4Double b)
{
    __m128d tmp1, tmp2;

    a.simdInternal_  = _mm256_mul_pd(a.simdInternal_.native_register(), b.simdInternal_.native_register());
    tmp1             = _mm256_castpd256_pd128(a.simdInternal_.native_register());
    tmp2             = _mm256_extractf128_pd(a.simdInternal_.native_register(), 0x1);

    tmp1 = _mm_add_pd(tmp1, _mm_permute_pd(tmp1, _MM_SHUFFLE2(0, 1)));
    tmp1 = _mm_add_pd(tmp1, tmp2);
    return *reinterpret_cast<double *>(&tmp1);
}

static inline void gmx_simdcall
transpose(Simd4Double * v0, Simd4Double * v1,
          Simd4Double * v2, Simd4Double * v3)
{
    __m256d t1, t2, t3, t4;
    t1                = _mm256_unpacklo_pd(v0->simdInternal_.native_register(), v1->simdInternal_.native_register());
    t2                = _mm256_unpackhi_pd(v0->simdInternal_.native_register(), v1->simdInternal_.native_register());
    t3                = _mm256_unpacklo_pd(v2->simdInternal_.native_register(), v3->simdInternal_.native_register());
    t4                = _mm256_unpackhi_pd(v2->simdInternal_.native_register(), v3->simdInternal_.native_register());
    v0->simdInternal_ = _mm256_permute2f128_pd(t1, t3, 0x20);
    v1->simdInternal_ = _mm256_permute2f128_pd(t2, t4, 0x20);
    v2->simdInternal_ = _mm256_permute2f128_pd(t1, t3, 0x31);
    v3->simdInternal_ = _mm256_permute2f128_pd(t2, t4, 0x31);
}

#else

#endif

#endif