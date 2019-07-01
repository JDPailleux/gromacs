#ifndef GMX_IMPL_NSIMD_FLOAT_DEFINED_H
#define GMX_IMPL_NSIMD_FLOAT_DEFINED_H


#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/cxx_adv_api_functions.hpp>
#include <nsimd/nsimd.h>

#include "impl_nsimd_simd_float.h"
#include "gromacs/math/utilities.h"

#if (defined(NSIMD_AVX2) || defined(NSIMD_AVX))

static inline SimdFloat gmx_simdcall
frexp(SimdFloat value, SimdFInt32 * exponent)
{
    const __m256  exponentMask      = _mm256_castsi256_ps(_mm256_set1_epi32(0x7F800000));
    const __m256  mantissaMask      = _mm256_castsi256_ps(_mm256_set1_epi32(0x807FFFFF));
    const __m256  half              = _mm256_set1_ps(0.5);
    const __m128i exponentBias      = _mm_set1_epi32(126);  // add 1 to make our definition identical to frexp()
    __m256i       iExponent;
    __m128i       iExponentLow, iExponentHigh;

    iExponent               = _mm256_castps_si256(_mm256_and_ps(value.simdInternal_.native_register(), exponentMask));
    iExponentHigh           = _mm256_extractf128_si256(iExponent, 0x1);
    iExponentLow            = _mm256_castsi256_si128(iExponent);
    iExponentLow            = _mm_srli_epi32(iExponentLow, 23);
    iExponentHigh           = _mm_srli_epi32(iExponentHigh, 23);
    iExponentLow            = _mm_sub_epi32(iExponentLow, exponentBias);
    iExponentHigh           = _mm_sub_epi32(iExponentHigh, exponentBias);
    iExponent               = _mm256_castsi128_si256(iExponentLow);
    exponent->simdInternal_ = _mm256_insertf128_si256(iExponent, iExponentHigh, 0x1);

    return {
               _mm256_or_ps(_mm256_and_ps(value.simdInternal_.native_register(), mantissaMask), half)
    };

}

template <MathOptimization opt = MathOptimization::Safe>
static inline SimdFloat gmx_simdcall
ldexp(SimdFloat value, SimdFInt32 exponent)
{
    const __m128i exponentBias      = _mm_set1_epi32(127);
    __m256i       iExponent;
    __m128i       iExponentLow, iExponentHigh;

    iExponentHigh = _mm256_extractf128_si256(exponent.simdInternal_.native_register(), 0x1);
    iExponentLow  = _mm256_castsi256_si128(exponent.simdInternal_.native_register());

    iExponentLow  = _mm_add_epi32(iExponentLow, exponentBias);
    iExponentHigh = _mm_add_epi32(iExponentHigh, exponentBias);

    if (opt == MathOptimization::Safe)
    {
        // Make sure biased argument is not negative
        iExponentLow  = _mm_max_epi32(iExponentLow, _mm_setzero_si128());
        iExponentHigh = _mm_max_epi32(iExponentHigh, _mm_setzero_si128());
    }

    iExponentLow  = _mm_slli_epi32(iExponentLow, 23);
    iExponentHigh = _mm_slli_epi32(iExponentHigh, 23);
    iExponent     = _mm256_castsi128_si256(iExponentLow);
    iExponent     = _mm256_insertf128_si256(iExponent, iExponentHigh, 0x1);
    return {
               _mm256_mul_ps(value.simdInternal_.native_register(), _mm256_castsi256_ps(iExponent))
    };
}

static inline float gmx_simdcall
reduce(SimdFloat a)
{
    __m128 t0;
    t0 = _mm_add_ps(_mm256_castps256_ps128(a.simdInternal_.native_register()), _mm256_extractf128_ps(a.simdInternal_.native_register(), 0x1));
    t0 = _mm_add_ps(t0, _mm_permute_ps(t0, _MM_SHUFFLE(1, 0, 3, 2)));
    t0 = _mm_add_ss(t0, _mm_permute_ps(t0, _MM_SHUFFLE(0, 3, 2, 1)));
    return *reinterpret_cast<float *>(&t0);
}

static inline SimdFInt32 gmx_simdcall
cvttR2I(SimdFloat a)
{
    return {
               _mm256_cvttps_epi32(a.simdInternal_.native_register())
    };
}

#endif

#endif