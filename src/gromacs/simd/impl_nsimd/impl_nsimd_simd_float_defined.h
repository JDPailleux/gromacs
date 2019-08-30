#ifndef GMX_IMPL_NSIMD_FLOAT_DEFINED_H
#define GMX_IMPL_NSIMD_FLOAT_DEFINED_H
 
#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/cxx_adv_api_functions.hpp>
#include <nsimd/nsimd.h>

#include "impl_nsimd_general.h"
#include "impl_nsimd_simd_float.h"

#if (defined(NSIMD_SSE2) || defined(NSIMD_SSE42))

static inline SimdFloat gmx_simdcall
frexp(SimdFloat value, SimdFInt32 * exponent)
{
    const __m128  exponentMask   = _mm_castsi128_ps(_mm_set1_epi32(0x7F800000));
    const __m128  mantissaMask   = _mm_castsi128_ps(_mm_set1_epi32(0x807FFFFF));
    const __m128i exponentBias   = _mm_set1_epi32(126); // add 1 to make our definition identical to frexp()
    const __m128  half           = _mm_set1_ps(0.5f);
    __m128i       iExponent;

    iExponent               = _mm_castps_si128(_mm_and_ps(value.simdInternal_.native_register(), exponentMask));
    iExponent               = _mm_sub_epi32(_mm_srli_epi32(iExponent, 23), exponentBias);
    exponent->simdInternal_ = iExponent;

    return {
               _mm_or_ps( _mm_and_ps(value.simdInternal_.native_register(), mantissaMask), half)
    };
}

static inline float gmx_simdcall
reduce(SimdFloat a)
{
    // Shuffle has latency 1/throughput 1, followed by add with latency 3, t-put 1.
    // This is likely faster than using _mm_hadd_ps, which has latency 5, t-put 2.
    a.simdInternal_ = _mm_add_ps(a.simdInternal_.native_register(), _mm_shuffle_ps(a.simdInternal_.native_register(), a.simdInternal_.native_register(), _MM_SHUFFLE(1, 0, 3, 2)));
    a.simdInternal_ = _mm_add_ss(a.simdInternal_.native_register(), _mm_shuffle_ps(a.simdInternal_.native_register(), a.simdInternal_.native_register(), _MM_SHUFFLE(0, 3, 2, 1)));
    return *reinterpret_cast<float *>(&a);
}

static inline SimdFInt32 gmx_simdcall
cvttR2I(SimdFloat a)
{
    return {
               _mm_cvttps_epi32(a.simdInternal_.native_register())
    };
}

#if defined(NSIMD_SSE42)

template <MathOptimization opt = MathOptimization::Safe>
static inline SimdFloat gmx_simdcall
ldexp(SimdFloat value, SimdFInt32 exponent)
{
    const __m128i exponentBias = _mm_set1_epi32(127);
    __m128i       iExponent;

    iExponent = _mm_add_epi32(exponent.simdInternal_.native_register(), exponentBias);

    if (opt == MathOptimization::Safe)
    {
        // Make sure biased argument is not negative
        iExponent = _mm_max_epi32(iExponent, _mm_setzero_si128());
    }

    iExponent = _mm_slli_epi32( iExponent, 23);

    return {
               _mm_mul_ps(value.simdInternal_.native_register(), _mm_castsi128_ps(iExponent))
    };
}

template<int index>
static inline std::int32_t gmx_simdcall
extract(SimdFInt32 a)
{
    return _mm_extract_epi32(a.simdInternal_.native_register(), index);
}
#else

template <MathOptimization opt = MathOptimization::Safe>
static inline SimdFloat gmx_simdcall
ldexp(SimdFloat value, SimdFInt32 exponent)
{
    const __m128i exponentBias = _mm_set1_epi32(127);
    __m128i       iExponent;

    iExponent = _mm_add_epi32(exponent.simdInternal_.native_register(), exponentBias);

    if (opt == MathOptimization::Safe)
    {
        // Make sure biased argument is not negative
        iExponent = _mm_and_si128(iExponent, _mm_cmpgt_epi32(iExponent, _mm_setzero_si128()));
    }

    iExponent = _mm_slli_epi32( iExponent, 23);

    return {
               _mm_mul_ps(value.simdInternal_.native_register(), _mm_castsi128_ps(iExponent))
    };
}

template<int index>
static inline std::int32_t gmx_simdcall
extract(SimdFInt32 a)
{
    return _mm_cvtsi128_si32( _mm_srli_si128(a.simdInternal_.native_register(), 4 * index) );
}
#endif

#elif (defined(NSIMD_AVX2) || defined(NSIMD_AVX))

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

template<int index>
static inline std::int32_t gmx_simdcall
extract(SimdFInt32 a)
{
    return _mm_extract_epi32(_mm256_extractf128_si256(a.simdInternal_.native_register(), index>>2), index & 0x3);
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


#elif (defined(NSIMD_AVX512_KNL) || defined(NSIMD_AVX512_SKYLAKE))

static inline SimdFloat gmx_simdcall
frexp(SimdFloat value, SimdFInt32 * exponent)
{
    __m512  rExponent = _mm512_getexp_ps(value.simdInternal_.native_register());
    __m512i iExponent =  _mm512_cvtps_epi32(rExponent);

    exponent->simdInternal_ = _mm512_add_epi32(iExponent, _mm512_set1_epi32(1));

    return {
               _mm512_getmant_ps(value.simdInternal_.native_register(), _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src)
    };
}

template <MathOptimization opt = MathOptimization::Safe>
static inline SimdFloat gmx_simdcall
ldexp(SimdFloat value, SimdFInt32 exponent)
{
    const __m512i exponentBias = _mm512_set1_epi32(127);
    __m512i       iExponent    =  _mm512_add_epi32(exponent.simdInternal_.native_register(), exponentBias);

    if (opt == MathOptimization::Safe)
    {
        // Make sure biased argument is not negative
        iExponent = _mm512_max_epi32(iExponent, _mm512_setzero_epi32());
    }

    iExponent = _mm512_slli_epi32(iExponent, 23);

    return {
               _mm512_mul_ps(value.simdInternal_.native_register(), _mm512_castsi512_ps(iExponent))
    };
}

static inline float gmx_simdcall
reduce(SimdFloat a)
{
    __m512 x = a.simdInternal_.native_register();
    x = _mm512_add_ps(x, _mm512_shuffle_f32x4(x, x, 0xEE));
    x = _mm512_add_ps(x, _mm512_shuffle_f32x4(x, x, 0x11));
    x = _mm512_add_ps(x, _mm512_permute_ps(x, 0xEE));
    x = _mm512_add_ps(x, _mm512_permute_ps(x, 0x11));
    return *reinterpret_cast<float *>(&x);
}

static inline SimdFInt32 gmx_simdcall
cvttR2I(SimdFloat a)
{
    return {
               _mm512_cvttps_epi32(a.simdInternal_.native_register())
    };
}

static inline SimdFloat gmx_simdcall
copysign(SimdFloat a, SimdFloat b)
{
    return {
               _mm512_castsi512_ps(_mm512_ternarylogic_epi32(
                                           _mm512_castps_si512(a.simdInternal_.native_register()),
                                           _mm512_castps_si512(b.simdInternal_.native_register()),
                                           _mm512_set1_epi32(INT32_MIN), 0xD8))
    };
}
 
#elif (defined(NSIMD_AARCH64) || defined(NSIMD_ARM_NEON))
static inline SimdFloat gmx_simdcall
frexp(SimdFloat value, SimdFInt32 * exponent)
{
    const int32x4_t    exponentMask   = vdupq_n_s32(0x7F800000);
    const int32x4_t    mantissaMask   = vdupq_n_s32(0x807FFFFF);
    const int32x4_t    exponentBias   = vdupq_n_s32(126); // add 1 to make our definition identical to frexp()
    const float32x4_t  half           = vdupq_n_f32(0.5f);
    int32x4_t          iExponent;

    iExponent               = vandq_s32(vreinterpretq_s32_f32(value.simdInternal_.native_register()), exponentMask);
    iExponent               = vsubq_s32(vshrq_n_s32(iExponent, 23), exponentBias);
    exponent->simdInternal_ = iExponent;

    return {
               vreinterpretq_f32_s32(vorrq_s32(vandq_s32(vreinterpretq_s32_f32(value.simdInternal_.native_register()),
                                                         mantissaMask),
                                               vreinterpretq_s32_f32(half)))
    };
}

template <MathOptimization opt = MathOptimization::Safe>
static inline SimdFloat gmx_simdcall
ldexp(SimdFloat value, SimdFInt32 exponent)
{
    const int32x4_t exponentBias = vdupq_n_s32(127);
    int32x4_t       iExponent    = vaddq_s32(exponent.simdInternal_.native_register(), exponentBias);

    if (opt == MathOptimization::Safe)
    {
        // Make sure biased argument is not negative
        iExponent = vmaxq_s32(iExponent, vdupq_n_s32(0));
    }

    iExponent = vshlq_n_s32( iExponent, 23);

    return {
               vmulq_f32(value.simdInternal_.native_register(), vreinterpretq_f32_s32(iExponent))
    };
}

#if defined(NSIMD_ARM_NEON)
static inline float gmx_simdcall
reduce(SimdFloat a)
{
    float32x4_t x = a.simdInternal_.native_register();
    float32x4_t y = vextq_f32(x, x, 2);

    x = vaddq_f32(x, y);
    y = vextq_f32(x, x, 1);
    x = vaddq_f32(x, y);
    return vgetq_lane_f32(x, 0);
}

#else 
static inline float gmx_simdcall
reduce(SimdFloat a)
{
    float32x4_t b = a.simdInternal_.native_register();
    b = vpaddq_f32(b, b);
    b = vpaddq_f32(b, b);
    return vgetq_lane_f32(b, 0);
}
#endif

template<int index> gmx_simdcall
static inline std::int32_t
extract(SimdFInt32 a)
{
    return vgetq_lane_s32(a.simdInternal_.native_register(), index);
    // return nsimd::get_lane(a.simdInternal_, index);
}

static inline SimdFInt32 gmx_simdcall
cvttR2I(SimdFloat a)
{
    return {
               vcvtq_s32_f32(a.simdInternal_.native_register())
               // nsimd::cvt<nsimd::pack<int>>(a.simdInternal_)
    };
}

#if GMX_SIMD_HAVE_NATIVE_RSQRT_ITER_FLOAT
static inline SimdFloat gmx_simdcall
rsqrtIter(SimdFloat lu, SimdFloat x)
{
    return {
               vmulq_f32(lu.simdInternal_.native_register(), vrsqrtsq_f32(vmulq_f32(lu.simdInternal_.native_register(), lu.simdInternal_.native_register()), x.simdInternal_.native_register()))
    };
}
#endif

static inline SimdFloat gmx_simdcall
rcpIter(SimdFloat lu, SimdFloat x)
{
    return {
               vmulq_f32(lu.simdInternal_.native_register(), vrecpsq_f32(lu.simdInternal_.native_register(), x.simdInternal_.native_register()))
    };
}


#endif

#endif
