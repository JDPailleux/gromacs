#ifndef GMX_SIMD_IMPL_NSIMD_SIMD_DOUBLE_DEFINED_H
#define GMX_SIMD_IMPL_NSIMD_SIMD_DOUBLE_DEFINED_H

#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/cxx_adv_api_functions.hpp>
#include <nsimd/nsimd.h>

#include "impl_nsimd_general.h"

#if (defined(NSIMD_SSE2) || defined(NSIMD_SSE42))

class SimdDInt32
{
    public:
        SimdDInt32() {}

        SimdDInt32(std::int32_t i) : simdInternal_(nsimd::set1<nsimd::pack<int> >(i)) {}

        // Internal utility constructor to simplify return statements
        SimdDInt32(nsimd::pack<int> simd) : simdInternal_(simd) {}

        nsimd::pack<int> simdInternal_;
};

class SimdDIBool
{
    public:
        SimdDIBool() {}

        SimdDIBool(bool b) : simdInternal_(nsimd::set1<nsimd::pack<int> >(b ? 0x7FFFFFFF : 0)) {}

        // Internal utility constructor to simplify return statements
        SimdDIBool(nsimd::pack<int> simd) : simdInternal_(simd) {}

        nsimd::pack<int> simdInternal_;
};


static inline SimdDInt32 gmx_simdcall
simdLoad(const std::int32_t * m, SimdDInt32Tag)
{
    assert(std::size_t(m) % 8 == 0);
    return {
               _mm_loadl_epi64(reinterpret_cast<const __m128i *>(m))
    };
}

static inline void gmx_simdcall
store(std::int32_t * m, SimdDInt32 a)
{
    assert(std::size_t(m) % 8 == 0);
    _mm_storel_epi64(reinterpret_cast<__m128i *>(m), a.simdInternal_.native_register());
}

static inline SimdDInt32 gmx_simdcall
simdLoadU(const std::int32_t *m, SimdDInt32Tag)
{
    return {
               _mm_loadl_epi64(reinterpret_cast<const __m128i *>(m))
    };
}

static inline void gmx_simdcall
storeU(std::int32_t * m, SimdDInt32 a)
{
    _mm_storel_epi64(reinterpret_cast<__m128i *>(m), a.simdInternal_.native_register());
}

static inline SimdDInt32 gmx_simdcall
setZeroDI()
{
    return {
               nsimd::set1<nsimd::pack<int> >(0)
    };
}

template<int index>
static inline std::int32_t gmx_simdcall
extract(SimdDInt32 a)
{
    return _mm_cvtsi128_si32( _mm_srli_si128(a.simdInternal_.native_register(), 4 * index) );
}

static inline double gmx_simdcall
reduce(SimdDouble a)
{
    __m128d b = _mm_add_sd(a.simdInternal_.native_register(), _mm_shuffle_pd(a.simdInternal_.native_register(), a.simdInternal_.native_register(), _MM_SHUFFLE2(1, 1)));
    return *reinterpret_cast<double *>(&b);
}

static inline SimdDBool gmx_simdcall
testBits(SimdDouble a)
{
    __m128i ia  = _mm_castpd_si128(a.simdInternal_.native_register());
    __m128i res = _mm_andnot_si128( _mm_cmpeq_epi32(ia, _mm_setzero_si128()), _mm_cmpeq_epi32(ia, ia));

    // set each 64-bit element if low or high 32-bit part is set
    res = _mm_or_si128(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(2, 3, 0, 1)));

    return {
               _mm_castsi128_pd(res)
    };
}

static inline SimdDouble
frexp(SimdDouble value, SimdDInt32 * exponent)
{
    // Don't use _mm_set1_epi64x() - on MSVC it is only supported for 64-bit builds
    const __m128d exponentMask = _mm_castsi128_pd( _mm_set_epi32(0x7FF00000, 0x00000000, 0x7FF00000, 0x00000000) );
    const __m128d mantissaMask = _mm_castsi128_pd( _mm_set_epi32(0x800FFFFF, 0xFFFFFFFF, 0x800FFFFF, 0xFFFFFFFF) );
    const __m128i exponentBias = _mm_set1_epi32(1022); // add 1 to make our definition identical to frexp()
    const __m128d half         = _mm_set1_pd(0.5);
    __m128i       iExponent;

    iExponent               = _mm_castpd_si128(_mm_and_pd(value.simdInternal_.native_register(), exponentMask));
    iExponent               = _mm_sub_epi32(_mm_srli_epi64(iExponent, 52), exponentBias);
    iExponent               = _mm_shuffle_epi32(iExponent, _MM_SHUFFLE(3, 1, 2, 0) );
    exponent->simdInternal_ = iExponent;

    return {
               _mm_or_pd(_mm_and_pd(value.simdInternal_.native_register(), mantissaMask), half)
    };
}

#if defined(NSIMD_SSE2)
template <MathOptimization opt = MathOptimization::Safe>
static inline SimdDouble
ldexp(SimdDouble value, SimdDInt32 exponent)
{
    const __m128i  exponentBias = _mm_set1_epi32(1023);
    __m128i        iExponent    = _mm_add_epi32(exponent.simdInternal_.native_register(), exponentBias);

    if (opt == MathOptimization::Safe)
    {
        // Make sure biased argument is not negative
        iExponent = _mm_and_si128(iExponent, _mm_cmpgt_epi32(iExponent, _mm_setzero_si128()));
    }

    // After conversion integers will be in slot 0,1. Move them to 0,2 so
    // we can do a 64-bit shift and get them to the dp exponents.
    iExponent = _mm_shuffle_epi32(iExponent, _MM_SHUFFLE(3, 1, 2, 0));
    iExponent = _mm_slli_epi64(iExponent, 52);

    return {
               _mm_mul_pd(value.simdInternal_.native_register(), _mm_castsi128_pd(iExponent))
    };
}

static inline SimdDouble gmx_simdcall
blend(SimdDouble a, SimdDouble b, SimdDBool sel)
{
    return {
               _mm_or_pd(_mm_andnot_pd(sel.simdInternal_.native_register(), a.simdInternal_.native_register()), _mm_and_pd(sel.simdInternal_.native_register(), b.simdInternal_.native_register()))
    };
}
#else

template <MathOptimization opt = MathOptimization::Safe>
static inline SimdDouble
ldexp(SimdDouble value, SimdDInt32 exponent)
{
    const __m128i  exponentBias = _mm_set1_epi32(1023);
    __m128i        iExponent    = _mm_add_epi32(exponent.simdInternal_.native_register(), exponentBias);

    if (opt == MathOptimization::Safe)
    {
        // Make sure biased argument is not negative
        iExponent = _mm_max_epi32(iExponent, _mm_setzero_si128());
    }

    // After conversion integers will be in slot 0,1. Move them to 0,2 so
    // we can do a 64-bit shift and get them to the dp exponents.
    iExponent = _mm_shuffle_epi32(iExponent, _MM_SHUFFLE(3, 1, 2, 0));
    iExponent = _mm_slli_epi64(iExponent, 52);

    return {
               _mm_mul_pd(value.simdInternal_.native_register(), _mm_castsi128_pd(iExponent))
    };
}

static inline SimdDouble gmx_simdcall
blend(SimdDouble a, SimdDouble b, SimdDBool sel)
{
    return {
               _mm_blendv_pd(a.simdInternal_.native_register(), b.simdInternal_.native_register(), sel.simdInternal_.native_register())
    };
}
#endif

static inline SimdDInt32 gmx_simdcall
operator&(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm_and_si128(a.simdInternal_.native_register(), b.simdInternal_.native_register())
    };
}

static inline SimdDInt32 gmx_simdcall
andNot(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm_andnot_si128(a.simdInternal_.native_register(), b.simdInternal_.native_register())
    };
}

static inline SimdDInt32 gmx_simdcall
operator|(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm_or_si128(a.simdInternal_.native_register(), b.simdInternal_.native_register())
    };
}

static inline SimdDInt32 gmx_simdcall
operator^(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm_xor_si128(a.simdInternal_.native_register(), b.simdInternal_.native_register())
    };
}

static inline SimdDInt32 gmx_simdcall
operator+(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm_add_epi32(a.simdInternal_.native_register(), b.simdInternal_.native_register())
    };
}

static inline SimdDInt32 gmx_simdcall
operator-(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm_sub_epi32(a.simdInternal_.native_register(), b.simdInternal_.native_register())
    };
}

// Override for SSE4.1 and higher
#if GMX_SIMD_X86_SSE2
static inline SimdDInt32 gmx_simdcall
operator*(SimdDInt32 a, SimdDInt32 b)
{

    __m128i tmpA = _mm_unpacklo_epi32(a.simdInternal_.native_register(), _mm_setzero_si128()); // 0 a[1] 0 a[0]
    __m128i tmpB = _mm_unpacklo_epi32(b.simdInternal_.native_register(), _mm_setzero_si128()); // 0 b[1] 0 b[0]

    __m128i tmpC  = _mm_mul_epu32(tmpA, tmpB);                               // 0 a[1]*b[1] 0 a[0]*b[0]

    return {
               _mm_shuffle_epi32(tmpC, _MM_SHUFFLE(3, 1, 2, 0))
    };
}
#endif

#if GMX_SIMD_X86_SSE4_1
static inline SimdDInt32 gmx_simdcall
operator*(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm_mullo_epi32(a.simdInternal_.native_register(), b.simdInternal_.native_register())
    };
}
#endif

static inline SimdDIBool gmx_simdcall
operator==(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm_cmpeq_epi32(a.simdInternal_.native_register(), b.simdInternal_.native_register())
    };
}

static inline SimdDIBool gmx_simdcall
testBits(SimdDInt32 a)
{
    __m128i x   = a.simdInternal_.native_register();
    __m128i res = _mm_andnot_si128( _mm_cmpeq_epi32(x, _mm_setzero_si128()), _mm_cmpeq_epi32(x, x));

    return {
               res
    };
}


static inline SimdDIBool gmx_simdcall
operator<(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm_cmplt_epi32(a.simdInternal_.native_register(), b.simdInternal_.native_register())
    };
}

static inline SimdDIBool gmx_simdcall
operator&&(SimdDIBool a, SimdDIBool b)
{
    return {
               _mm_and_si128(a.simdInternal_.native_register(), b.simdInternal_.native_register())
    };
}

static inline SimdDIBool gmx_simdcall
operator||(SimdDIBool a, SimdDIBool b)
{
    return {
               _mm_or_si128(a.simdInternal_.native_register(), b.simdInternal_.native_register())
    };
}

static inline bool gmx_simdcall
anyTrue(SimdDIBool a)
{
    return _mm_movemask_epi8(_mm_shuffle_epi32(a.simdInternal_.native_register(), _MM_SHUFFLE(1, 0, 1, 0))) != 0;
}

static inline bool gmx_simdcall
anyTrue(SimdDBool a) { return _mm_movemask_pd(a.simdInternal_.native_register()) != 0; }

static inline SimdDInt32 gmx_simdcall
selectByMask(SimdDInt32 a, SimdDIBool mask)
{
    return {
              _mm_and_si128(mask.simdInternal_.native_register(), a.simdInternal_.native_register())
    };
}

static inline SimdDInt32 gmx_simdcall
selectByNotMask(SimdDInt32 a, SimdDIBool mask)
{
    return {
               _mm_andnot_si128(mask.simdInternal_.native_register(), a.simdInternal_.native_register())
    };
}

static inline SimdDInt32 gmx_simdcall
blend(SimdDInt32 a, SimdDInt32 b, SimdDIBool sel)
{
    return {
                _mm_or_si128(_mm_andnot_si128(sel.simdInternal_.native_register(), a.simdInternal_.native_register()), _mm_and_si128(sel.simdInternal_.native_register(), b.simdInternal_.native_register()))
    };
}

static inline SimdDInt32 gmx_simdcall
cvtR2I(SimdDouble a)
{
    return {
               _mm_cvtpd_epi32(a.simdInternal_.native_register())
    };
}

static inline SimdDInt32 gmx_simdcall
cvttR2I(SimdDouble a)
{
    return {
               _mm_cvttpd_epi32(a.simdInternal_.native_register())
    };
}

static inline SimdDouble gmx_simdcall
cvtI2R(SimdDInt32 a)
{
    return {
               _mm_cvtepi32_pd(a.simdInternal_.native_register())
    };
}

static inline SimdDIBool gmx_simdcall
cvtB2IB(SimdDBool a)
{
    return {
               _mm_shuffle_epi32(_mm_castpd_si128(a.simdInternal_.native_register()), _MM_SHUFFLE(2, 0, 2, 0))
    };
}

static inline SimdDBool gmx_simdcall
cvtIB2B(SimdDIBool a)
{
    return {
               _mm_castsi128_pd(_mm_shuffle_epi32(a.simdInternal_.native_register(), _MM_SHUFFLE(1, 1, 0, 0)))
    };
}

static inline void gmx_simdcall
cvtF2DD(SimdFloat f, SimdDouble *d0, SimdDouble *d1)
{
    d0->simdInternal_ = _mm_cvtps_pd(f.simdInternal_.native_register());
    d1->simdInternal_ = _mm_cvtps_pd(_mm_movehl_ps(f.simdInternal_.native_register(), f.simdInternal_.native_register()));
}

static inline SimdFloat gmx_simdcall
cvtDD2F(SimdDouble d0, SimdDouble d1)
{
    return {
              _mm_movelh_ps(_mm_cvtpd_ps(d0.simdInternal_.native_register()), _mm_cvtpd_ps(d1.simdInternal_.native_register()))
    };
}

static inline SimdDBool gmx_simdcall
operator==(SimdDouble a, SimdDouble b)
{
    return {
               _mm_cmpeq_pd(a.simdInternal_.native_register(), b.simdInternal_.native_register())
    };
}

static inline SimdDBool gmx_simdcall
operator!=(SimdDouble a, SimdDouble b)
{
    return {
               _mm_cmpneq_pd(a.simdInternal_.native_register(), b.simdInternal_.native_register())
    };
}

static inline SimdDBool gmx_simdcall
operator<(SimdDouble a, SimdDouble b)
{
    return {
               _mm_cmplt_pd(a.simdInternal_.native_register(), b.simdInternal_.native_register())
    };
}

static inline SimdDBool gmx_simdcall
operator<=(SimdDouble a, SimdDouble b)
{
    return {
               _mm_cmple_pd(a.simdInternal_.native_register(), b.simdInternal_.native_register())
    };
}

#elif (defined(NSIMD_AVX2) || defined(NSIMD_AVX))
class SimdDInt32 // Original class because simdInternal_ is in an invalid register
{
    public:
        SimdDInt32() {}

        SimdDInt32(std::int32_t i) : simdInternal_(_mm_set1_epi32(i)) {}

        // Internal utility constructor to simplify return statements
        SimdDInt32(__m128i simd) : simdInternal_(simd) {}

        __m128i  simdInternal_;
};

class SimdDIBool
{
    public:
        SimdDIBool() {}

        SimdDIBool(bool b) : simdInternal_(_mm_set1_epi32( b ? 0xFFFFFFFF : 0)) {}

        // Internal utility constructor to simplify return statements
        SimdDIBool(__m128i simd) : simdInternal_(simd) {}

        __m128i  simdInternal_;
};

static inline SimdDInt32 gmx_simdcall
simdLoad(const std::int32_t * m, SimdDInt32Tag /*unused*/)
{
    assert(std::size_t(m) % 16 == 0);
    return {
               _mm_load_si128(reinterpret_cast<const __m128i *>(m))
    };
}

static inline void gmx_simdcall
store(std::int32_t * m, SimdDInt32 a)
{
    assert(std::size_t(m) % 16 == 0);
    _mm_store_si128(reinterpret_cast<__m128i *>(m), a.simdInternal_);
}

static inline SimdDInt32 gmx_simdcall
simdLoadU(const std::int32_t *m, SimdDInt32Tag /*unused*/)
{
    return {
               _mm_loadu_si128(reinterpret_cast<const __m128i *>(m))
    };
}

static inline void gmx_simdcall
storeU(std::int32_t * m, SimdDInt32 a)
{
    _mm_storeu_si128(reinterpret_cast<__m128i *>(m), a.simdInternal_);
}

static inline SimdDInt32 gmx_simdcall
setZeroDI()
{
    return {
               _mm_setzero_si128()
    };
}

template<int index>
static inline std::int32_t gmx_simdcall
extract(SimdDInt32 a)
{
    return _mm_extract_epi32(a.simdInternal_, index);
}

static inline double gmx_simdcall
reduce(SimdDouble a)
{
    __m128d a0, a1;
    a.simdInternal_ = _mm256_add_pd(a.simdInternal_.native_register(), _mm256_permute_pd(a.simdInternal_.native_register(), 0b0101 ));
    a0              = _mm256_castpd256_pd128(a.simdInternal_.native_register());
    a1              = _mm256_extractf128_pd(a.simdInternal_.native_register(), 0x1);
    a0              = _mm_add_sd(a0, a1);

    return *reinterpret_cast<double *>(&a0);
}

static inline SimdDIBool gmx_simdcall
testBits(SimdDInt32 a)
{
    __m128i x   = a.simdInternal_;
    __m128i res = _mm_andnot_si128( _mm_cmpeq_epi32(x, _mm_setzero_si128()), _mm_cmpeq_epi32(x, x));

    return {
               res
    };
}

static inline SimdDBool gmx_simdcall
testBits(SimdDouble a)
{
    // Do an or of the low/high 32 bits of each double (so the data is replicated),
    // and then use the same algorithm as we use for single precision.
    __m256 tst = _mm256_castpd_ps(a.simdInternal_.native_register());

    tst = _mm256_or_ps(tst, _mm256_permute_ps(tst, _MM_SHUFFLE(2, 3, 0, 1)));
    tst = _mm256_cvtepi32_ps(_mm256_castps_si256(tst));

    return {
               _mm256_castps_pd(_mm256_cmp_ps(tst, _mm256_setzero_ps(), _CMP_NEQ_OQ))
    };
}

static inline bool gmx_simdcall
anyTrue(SimdDIBool a) { return _mm_movemask_epi8(a.simdInternal_) != 0; }

static inline bool gmx_simdcall
anyTrue(SimdDBool a) { return _mm256_movemask_pd(a.simdInternal_.native_register()) != 0; }

static inline SimdDInt32 gmx_simdcall
operator&(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm_and_si128(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
andNot(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm_andnot_si128(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
operator|(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm_or_si128(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
operator^(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm_xor_si128(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
operator+(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm_add_epi32(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
operator-(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm_sub_epi32(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
operator*(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm_mullo_epi32(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDIBool gmx_simdcall
operator==(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm_cmpeq_epi32(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDIBool gmx_simdcall
operator<(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm_cmplt_epi32(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDIBool gmx_simdcall
operator&&(SimdDIBool a, SimdDIBool b)
{
    return {
               _mm_and_si128(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDIBool gmx_simdcall
operator||(SimdDIBool a, SimdDIBool b)
{
    return {
               _mm_or_si128(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
blend(SimdDInt32 a, SimdDInt32 b, SimdDIBool sel)
{
    return {
               _mm_blendv_epi8(a.simdInternal_, b.simdInternal_, sel.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
cvtR2I(SimdDouble a)
{
    return {
               _mm256_cvtpd_epi32(a.simdInternal_.native_register())
    };
}
 
static inline SimdDInt32 gmx_simdcall
cvttR2I(SimdDouble a)
{
    return {
               _mm256_cvttpd_epi32(a.simdInternal_.native_register())
    };
}

static inline SimdDouble gmx_simdcall
cvtI2R(SimdDInt32 a)
{
    return {
               _mm256_cvtepi32_pd(a.simdInternal_)
    };
}

static inline SimdDIBool gmx_simdcall
cvtB2IB(SimdDBool a)
{
    __m128i a1 = _mm256_extractf128_si256(_mm256_castpd_si256(a.simdInternal_.native_register()), 0x1);
    __m128i a0 = _mm256_castsi256_si128(_mm256_castpd_si256(a.simdInternal_.native_register()));
    a0 = _mm_shuffle_epi32(a0, _MM_SHUFFLE(2, 0, 2, 0));
    a1 = _mm_shuffle_epi32(a1, _MM_SHUFFLE(2, 0, 2, 0));

    return {
               _mm_blend_epi16(a0, a1, 0xF0)
    };
}

static inline SimdDBool gmx_simdcall
cvtIB2B(SimdDIBool a)
{
    __m128d lo = _mm_castsi128_pd(_mm_unpacklo_epi32(a.simdInternal_, a.simdInternal_));
    __m128d hi = _mm_castsi128_pd(_mm_unpackhi_epi32(a.simdInternal_, a.simdInternal_));

    return {
               _mm256_insertf128_pd(_mm256_castpd128_pd256(lo), hi, 0x1)
    };
}

static inline void gmx_simdcall
cvtF2DD(SimdFloat f, SimdDouble *d0, SimdDouble *d1)
{
    d0->simdInternal_ = _mm256_cvtps_pd(_mm256_castps256_ps128(f.simdInternal_.native_register()));
    d1->simdInternal_ = _mm256_cvtps_pd(_mm256_extractf128_ps(f.simdInternal_.native_register(), 0x1));
}

static inline SimdFloat gmx_simdcall
cvtDD2F(SimdDouble d0, SimdDouble d1)
{
    __m128 f0 = _mm256_cvtpd_ps(d0.simdInternal_.native_register());
    __m128 f1 = _mm256_cvtpd_ps(d1.simdInternal_.native_register());
    return {
               _mm256_insertf128_ps(_mm256_castps128_ps256(f0), f1, 0x1)
    };
}

static inline SimdDouble
frexp(SimdDouble value, SimdDInt32 * exponent)
{
    const __m256d exponentMask      = _mm256_castsi256_pd( _mm256_set1_epi64x(0x7FF0000000000000LL));
    const __m256d mantissaMask      = _mm256_castsi256_pd( _mm256_set1_epi64x(0x800FFFFFFFFFFFFFLL));
    const __m256d half              = _mm256_set1_pd(0.5);
    const __m128i exponentBias      = _mm_set1_epi32(1022); // add 1 to make our definition identical to frexp()
    __m256i       iExponent;
    __m128i       iExponentLow, iExponentHigh;

    iExponent               = _mm256_castpd_si256(_mm256_and_pd(value.simdInternal_.native_register(), exponentMask));
    iExponentHigh           = _mm256_extractf128_si256(iExponent, 0x1);
    iExponentLow            = _mm256_castsi256_si128(iExponent);
    iExponentLow            = _mm_srli_epi64(iExponentLow, 52);
    iExponentHigh           = _mm_srli_epi64(iExponentHigh, 52);
    iExponentLow            = _mm_shuffle_epi32(iExponentLow, _MM_SHUFFLE(1, 1, 2, 0));
    iExponentHigh           = _mm_shuffle_epi32(iExponentHigh, _MM_SHUFFLE(2, 0, 1, 1));
    iExponentLow            = _mm_or_si128(iExponentLow, iExponentHigh);
    exponent->simdInternal_ = _mm_sub_epi32(iExponentLow, exponentBias);

    return {
               _mm256_or_pd(_mm256_and_pd(value.simdInternal_.native_register(), mantissaMask), half)
    };
}

template <MathOptimization opt = MathOptimization::Safe>
static inline SimdDouble
ldexp(SimdDouble value, SimdDInt32 exponent)
{
    const __m128i exponentBias = _mm_set1_epi32(1023);
    __m128i       iExponentLow, iExponentHigh;
    __m256d       fExponent;

    iExponentLow  = _mm_add_epi32(exponent.simdInternal_, exponentBias);

    if (opt == MathOptimization::Safe)
    {
        // Make sure biased argument is not negative
        iExponentLow  = _mm_max_epi32(iExponentLow, _mm_setzero_si128());
    }

    iExponentHigh = _mm_shuffle_epi32(iExponentLow, _MM_SHUFFLE(3, 3, 2, 2));
    iExponentLow  = _mm_shuffle_epi32(iExponentLow, _MM_SHUFFLE(1, 1, 0, 0));
    iExponentHigh = _mm_slli_epi64(iExponentHigh, 52);
    iExponentLow  = _mm_slli_epi64(iExponentLow, 52);
    fExponent     = _mm256_castsi256_pd(_mm256_insertf128_si256(_mm256_castsi128_si256(iExponentLow), iExponentHigh, 0x1));
    return {
               _mm256_mul_pd(value.simdInternal_.native_register(), fExponent)
    };
}

static inline SimdDBool gmx_simdcall
operator==(SimdDouble a, SimdDouble b)
{
    return {
               _mm256_cmp_pd(a.simdInternal_.native_register(), b.simdInternal_.native_register(), _CMP_EQ_OQ)
    };
}

static inline SimdDBool gmx_simdcall
operator!=(SimdDouble a, SimdDouble b)
{
    return {
               _mm256_cmp_pd(a.simdInternal_.native_register(), b.simdInternal_.native_register(), _CMP_NEQ_OQ)
    };
}

static inline SimdDBool gmx_simdcall
operator<(SimdDouble a, SimdDouble b)
{
    return {
               _mm256_cmp_pd(a.simdInternal_.native_register(), b.simdInternal_.native_register(), _CMP_LT_OQ)
    };
}

static inline SimdDBool gmx_simdcall
operator<=(SimdDouble a, SimdDouble b)
{
    return {
               _mm256_cmp_pd(a.simdInternal_.native_register(), b.simdInternal_.native_register(), _CMP_LE_OQ)
    };
}

static inline SimdDInt32 gmx_simdcall
selectByMask(SimdDInt32 a, SimdDIBool mask)
{
    return {
               _mm_and_si128(a.simdInternal_, mask.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
selectByNotMask(SimdDInt32 a, SimdDIBool mask)
{
    return {
               _mm_andnot_si128(mask.simdInternal_, a.simdInternal_)
    };
}

static inline SimdDouble gmx_simdcall
blend(SimdDouble a, SimdDouble b, SimdDBool sel)
{
    return {
               _mm256_blendv_pd(a.simdInternal_.native_register(), b.simdInternal_.native_register(), sel.simdInternal_.native_register())
    };
}
#elif (defined(NSIMD_AVX512_KNL) || defined(NSIMD_AVX512_SKYLAKE))

class SimdDInt32
{
    public:
        SimdDInt32() {}

        SimdDInt32(std::int32_t i) : simdInternal_(_mm256_set1_epi32(i)) {}

        // Internal utility constructor to simplify return statements
        SimdDInt32(__m256i simd) : simdInternal_(simd) {}

        __m256i  simdInternal_;
};

class SimdDIBool
{
    public:
        SimdDIBool() {}

        // Internal utility constructor to simplify return statements
        SimdDIBool(__mmask16 simd) : simdInternal_(simd) {}

        __mmask16  simdInternal_;
};

static inline SimdDInt32 gmx_simdcall
simdLoad(const std::int32_t * m, SimdDInt32Tag)
{
    assert(std::size_t(m) % 32 == 0);
    return {
               _mm256_load_si256(reinterpret_cast<const __m256i *>(m))
    };
}

static inline void gmx_simdcall
store(std::int32_t * m, SimdDInt32 a)
{
    assert(std::size_t(m) % 32 == 0);
    _mm256_store_si256(reinterpret_cast<__m256i *>(m), a.simdInternal_);
}

static inline SimdDInt32 gmx_simdcall
simdLoadU(const std::int32_t *m, SimdDInt32Tag)
{
    return {
               _mm256_loadu_si256(reinterpret_cast<const __m256i *>(m))
    };
}

static inline void gmx_simdcall
storeU(std::int32_t * m, SimdDInt32 a)
{
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(m), a.simdInternal_);
}

static inline SimdDInt32 gmx_simdcall
setZeroDI()
{
    return {
               _mm256_setzero_si256()
    };
}

static inline double gmx_simdcall
reduce(SimdDouble a)
{
    __m512d x = a.simdInternal_.native_register();
    x = _mm512_add_pd(x, _mm512_shuffle_f64x2(x, x, 0xEE));
    x = _mm512_add_pd(x, _mm512_shuffle_f64x2(x, x, 0x11));
    x = _mm512_add_pd(x, _mm512_permute_pd(x, 0x01));
    return *reinterpret_cast<double *>(&x);
}

static inline SimdDBool gmx_simdcall
testBits(SimdDouble a)
{
    return {
               _mm512_test_epi64_mask(_mm512_castpd_si512(a.simdInternal_.native_register()), _mm512_castpd_si512(a.simdInternal_.native_register()))
    };
}

static inline SimdDIBool gmx_simdcall
testBits(SimdDInt32 a)
{
    return {
               _mm512_mask_test_epi32_mask(avx512Int2Mask(0xFF), _mm512_castsi256_si512(a.simdInternal_), _mm512_castsi256_si512(a.simdInternal_))
    };
}

static inline SimdDInt32 gmx_simdcall
operator&(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm256_and_si256(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
andNot(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm256_andnot_si256(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
operator|(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm256_or_si256(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
operator^(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm256_xor_si256(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
operator+(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm256_add_epi32(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
operator-(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm256_sub_epi32(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
operator*(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm256_mullo_epi32(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDIBool gmx_simdcall
operator==(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm512_mask_cmp_epi32_mask(avx512Int2Mask(0xFF), _mm512_castsi256_si512(a.simdInternal_), _mm512_castsi256_si512(b.simdInternal_), _MM_CMPINT_EQ)
    };
}

static inline SimdDIBool gmx_simdcall
operator<(SimdDInt32 a, SimdDInt32 b)
{
    return {
               _mm512_mask_cmp_epi32_mask(avx512Int2Mask(0xFF), _mm512_castsi256_si512(a.simdInternal_), _mm512_castsi256_si512(b.simdInternal_), _MM_CMPINT_LT)
    };
}

static inline SimdDIBool gmx_simdcall
operator&&(SimdDIBool a, SimdDIBool b)
{
    return {
               _mm512_kand(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDIBool gmx_simdcall
operator||(SimdDIBool a, SimdDIBool b)
{
    return {
               _mm512_kor(a.simdInternal_, b.simdInternal_)
    };
}

static inline bool gmx_simdcall
anyTrue(SimdDIBool a)
{
    return ( avx512Mask2Int(a.simdInternal_) & 0xFF) != 0;
}

static inline bool gmx_simdcall
anyTrue(SimdDBool a)
{
    return ( avx512Mask2Int(a.simdInternal_) != 0);
}

static inline SimdDBool gmx_simdcall
operator==(SimdDouble a, SimdDouble b)
{
    return {
               _mm512_cmp_pd_mask(a.simdInternal_.native_register(), b.simdInternal_.native_register(), _CMP_EQ_OQ)
    };
}

static inline SimdDBool gmx_simdcall
operator!=(SimdDouble a, SimdDouble b)
{
    return {
               _mm512_cmp_pd_mask(a.simdInternal_.native_register(), b.simdInternal_.native_register(), _CMP_NEQ_OQ)
    };
}

static inline SimdDBool gmx_simdcall
operator<(SimdDouble a, SimdDouble b)
{
    return {
               _mm512_cmp_pd_mask(a.simdInternal_.native_register(), b.simdInternal_.native_register(), _CMP_LT_OQ)
    };
}

static inline SimdDBool gmx_simdcall
operator<=(SimdDouble a, SimdDouble b)
{
    return {
               _mm512_cmp_pd_mask(a.simdInternal_.native_register(), b.simdInternal_.native_register(), _CMP_LE_OQ)
    };
}


static inline SimdDouble
frexp(SimdDouble value, SimdDInt32 * exponent)
{
    __m512d rExponent = _mm512_getexp_pd(value.simdInternal_.native_register());
    __m256i iExponent = _mm512_cvtpd_epi32(rExponent);

    exponent->simdInternal_ = _mm256_add_epi32(iExponent, _mm256_set1_epi32(1));

    return {
               _mm512_getmant_pd(value.simdInternal_.native_register(), _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src)
    };
}

template <MathOptimization opt = MathOptimization::Safe>
static inline SimdDouble
ldexp(SimdDouble value, SimdDInt32 exponent)
{
    const __m256i exponentBias = _mm256_set1_epi32(1023);
    __m256i       iExponent    = _mm256_add_epi32(exponent.simdInternal_, exponentBias);
    __m512i       iExponent512;

    if (opt == MathOptimization::Safe)
    {
        // Make sure biased argument is not negative
        iExponent = _mm256_max_epi32(iExponent, _mm256_setzero_si256());
    }

    iExponent512 = _mm512_permutexvar_epi32(_mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), _mm512_castsi256_si512(iExponent));
    iExponent512 = _mm512_mask_slli_epi32(_mm512_setzero_epi32(), avx512Int2Mask(0xAAAA), iExponent512, 20);
    return {
	    _mm512_mul_pd(_mm512_castsi512_pd(iExponent512), value.simdInternal_.native_register())
	};
}



static inline SimdDouble gmx_simdcall
copysign(SimdDouble a, SimdDouble b)
{
    return {
               _mm512_castsi512_pd(_mm512_ternarylogic_epi64(
                                           _mm512_castpd_si512(a.simdInternal_.native_register()),
                                           _mm512_castpd_si512(b.simdInternal_.native_register()),
                                           _mm512_set1_epi64(INT64_MIN), 0xD8))
    };
}

static inline SimdDInt32 gmx_simdcall
blend(SimdDInt32 a, SimdDInt32 b, SimdDIBool sel)
{
    return {
               _mm512_castsi512_si256(_mm512_mask_blend_epi32(sel.simdInternal_, _mm512_castsi256_si512(a.simdInternal_), _mm512_castsi256_si512(b.simdInternal_)))
    };
}

static inline SimdDouble gmx_simdcall
blend(SimdDouble a, SimdDouble b, SimdDBool sel)
{
    return {
               _mm512_mask_blend_pd(sel.simdInternal_, a.simdInternal_.native_register(), b.simdInternal_.native_register())
    };
}

static inline SimdDInt32 gmx_simdcall
cvtR2I(SimdDouble a)
{
    return {
               _mm512_cvtpd_epi32(a.simdInternal_.native_register())
    };
}

static inline SimdDInt32 gmx_simdcall
cvttR2I(SimdDouble a)
{
    return {
               _mm512_cvttpd_epi32(a.simdInternal_.native_register())
    };
}

static inline SimdDouble gmx_simdcall
cvtI2R(SimdDInt32 a)
{
    return {
               _mm512_cvtepi32_pd(a.simdInternal_)
    };
}

static inline SimdDIBool gmx_simdcall
cvtB2IB(SimdDBool a)
{
    return {
              a.simdInternal_
    };
}

static inline SimdDBool gmx_simdcall
cvtIB2B(SimdDIBool a)
{
    return {
               static_cast<__mmask8>(a.simdInternal_)
    };
}

static inline void gmx_simdcall
cvtF2DD(SimdFloat f, SimdDouble *d0, SimdDouble *d1)
{
    d0->simdInternal_ = _mm512_cvtps_pd(_mm512_castps512_ps256(f.simdInternal_.native_register()));
    d1->simdInternal_ = _mm512_cvtps_pd(_mm512_castps512_ps256(_mm512_shuffle_f32x4(f.simdInternal_.native_register(), f.simdInternal_.native_register(), 0xEE)));
}

static inline SimdFloat gmx_simdcall
cvtDD2F(SimdDouble d0, SimdDouble d1)
{
    __m512 f0 = _mm512_castps256_ps512(_mm512_cvtpd_ps(d0.simdInternal_.native_register()));
    __m512 f1 = _mm512_castps256_ps512(_mm512_cvtpd_ps(d1.simdInternal_.native_register()));
    return {
               _mm512_shuffle_f32x4(f0, f1, 0x44)
    };
}

//------------------

static inline SimdDBool gmx_simdcall
operator&&(SimdDBool a, SimdDBool b)
{
    return {
               static_cast<__mmask8>(_mm512_kand(a.simdInternal_, b.simdInternal_))
    };
}

static inline SimdDBool gmx_simdcall
operator||(SimdDBool a, SimdDBool b)
{
    return {
               static_cast<__mmask8>(_mm512_kor(a.simdInternal_, b.simdInternal_))
    };
}

static inline SimdDouble gmx_simdcall
selectByMask(SimdDouble a, SimdDBool m)
{
    return {
               _mm512_mask_mov_pd(_mm512_setzero_pd(), m.simdInternal_, a.simdInternal_.native_register())
    };
}

static inline SimdDouble gmx_simdcall
selectByNotMask(SimdDouble a, SimdDBool m)
{
    return {
               _mm512_mask_mov_pd(a.simdInternal_.native_register(), m.simdInternal_, _mm512_setzero_pd())
    };
}

static inline SimdDouble gmx_simdcall
maskAdd(SimdDouble a, SimdDouble b, SimdDBool m)
{
    return {
               _mm512_mask_add_pd(a.simdInternal_.native_register(), m.simdInternal_, a.simdInternal_.native_register(), b.simdInternal_.native_register())
    };
}

static inline SimdDouble gmx_simdcall
maskzMul(SimdDouble a, SimdDouble b, SimdDBool m)
{
    return {
               _mm512_maskz_mul_pd(m.simdInternal_, a.simdInternal_.native_register(), b.simdInternal_.native_register())
    };
}

static inline SimdDouble gmx_simdcall
maskzFma(SimdDouble a, SimdDouble b, SimdDouble c, SimdDBool m)
{
    return {
               _mm512_maskz_fmadd_pd(m.simdInternal_, a.simdInternal_.native_register(), b.simdInternal_.native_register(), c.simdInternal_.native_register())
    };
}


#if defined(NSIMD_AVX512_SKYLAKE)
static inline SimdDouble gmx_simdcall
maskzRsqrt(SimdDouble x, SimdDBool m)
{
    return {
               _mm512_maskz_rsqrt14_pd(m.simdInternal_, x.simdInternal_.native_register())
    };
}

static inline SimdDouble gmx_simdcall
maskzRcp(SimdDouble x, SimdDBool m)
{
    return {
               _mm512_maskz_rcp14_pd(m.simdInternal_, x.simdInternal_.native_register())
    };
}
#else 

static inline SimdDouble gmx_simdcall
maskzRsqrt(SimdDouble x, SimdDBool m)
{
    return {
               _mm512_maskz_rsqrt28_pd(m.simdInternal_, x.simdInternal_.native_register())
    };
}

static inline SimdDouble gmx_simdcall
maskzRcp(SimdDouble x, SimdDBool m)
{
    return {
               _mm512_maskz_rcp28_pd(m.simdInternal_, x.simdInternal_.native_register())
    };
}
#endif

#elif defined(NSIMD_AARCH64)

class SimdDInt32
{
    public:
        SimdDInt32() {}

        SimdDInt32(std::int32_t i) : simdInternal_(vdup_n_s32(i)) {}

        // Internal utility constructor to simplify return statements
        SimdDInt32(int32x2_t simd) : simdInternal_(simd) {}

        int32x2_t  simdInternal_;
};

class SimdDIBool
{
    public:
        SimdDIBool() {}

        SimdDIBool(bool b) : simdInternal_(vdup_n_u32( b ? 0xFFFFFFFF : 0)) {}

        // Internal utility constructor to simplify return statements
        SimdDIBool(uint32x2_t simd) : simdInternal_(simd) {}

        uint32x2_t  simdInternal_;
};

static inline SimdDInt32 gmx_simdcall
simdLoad(const std::int32_t * m, SimdDInt32Tag)
{
    assert(std::size_t(m) % 8 == 0);
    return {
               vld1_s32(m)
    };
}

static inline void gmx_simdcall
store(std::int32_t * m, SimdDInt32 a)
{
    assert(std::size_t(m) % 8 == 0);
    vst1_s32(m, a.simdInternal_);
}

static inline SimdDInt32 gmx_simdcall
simdLoadU(const std::int32_t *m, SimdDInt32Tag)
{
    return {
               vld1_s32(m)
    };
}

static inline void gmx_simdcall
storeU(std::int32_t * m, SimdDInt32 a)
{
    vst1_s32(m, a.simdInternal_);
}

s.native_register()
s.native_register()
{.native_register()
    return {
               vdup_n_s32(0)
    };
}

template<int index> gmx_simdcall
static inline std::int32_t
extract(SimdDInt32 a)
{
    return vget_lane_s32(a.simdInternal_, index);
}

static inline double gmx_simdcall
reduce(SimdDouble a)
{
    float64x2_t b = vpaddq_f64(a.simdInternal_, a.simdInternal_);
    return vgetq_lane_f64(b, 0);
}

static inline SimdDIBool gmx_simdcall
testBits(SimdDInt32 a)
{
    return {
               vtst_s32( a.simdInternal_, a.simdInternal_)
    };
}

static inline SimdDBool gmx_simdcall
testBits(SimdDouble a)
{
    return {
               vtstq_s64( int64x2_t(a.simdInternal_.native_register()), int64x2_t(a.simdInternal_.native_register()) )
    };
}

static inline bool gmx_simdcall
anyTrue(SimdDIBool a)
{
    return (vmaxv_u32(a.simdInternal_) != 0);
}

static inline bool gmx_simdcall
anyTrue(SimdDBool a)
{
    return (vmaxvq_u32((uint32x4_t)(a.simdInternal_.native_register())) != 0);
}

static inline SimdDInt32 gmx_simdcall
operator&(SimdDInt32 a, SimdDInt32 b)
{
    return {
               vand_s32(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
andNot(SimdDInt32 a, SimdDInt32 b)
{
    return {
               vbic_s32(b.simdInternal_, a.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
operator|(SimdDInt32 a, SimdDInt32 b)
{
    return {
               vorr_s32(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
operator^(SimdDInt32 a, SimdDInt32 b)
{
    return {
               veor_s32(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
operator+(SimdDInt32 a, SimdDInt32 b)
{
    return {
               vadd_s32(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
operator-(SimdDInt32 a, SimdDInt32 b)
{
    return {
               vsub_s32(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
operator*(SimdDInt32 a, SimdDInt32 b)
{
    return {
               vmul_s32(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDIBool gmx_simdcall
operator==(SimdDInt32 a, SimdDInt32 b)
{
    return {
               vceq_s32(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDIBool gmx_simdcall
operator<(SimdDInt32 a, SimdDInt32 b)
{
    return {
               vclt_s32(a.simdInternal_, b.simdInternal_)
    };
}


static inline SimdDInt32 gmx_simdcall
selectByMask(SimdDInt32 a, SimdDIBool m)
{
    return {
               vand_s32(a.simdInternal_, vreinterpret_s32_u32(m.simdInternal_))
    };
}

static inline SimdDInt32 gmx_simdcall
selectByNotMask(SimdDInt32 a, SimdDIBool m)
{
    return {
               vbic_s32(a.simdInternal_, vreinterpret_s32_u32(m.simdInternal_))
    };
}

static inline SimdDInt32 gmx_simdcall
blend(SimdDInt32 a, SimdDInt32 b, SimdDIBool sel)
{
    return {
               vbsl_s32(sel.simdInternal_, b.simdInternal_, a.simdInternal_)
    };
}

static inline SimdDouble gmx_simdcall
blend(SimdDouble a, SimdDouble b, SimdDBool sel)
{
    return {
               vbslq_f64(sel.simdInternal_, b.simdInternal_, a.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
cvtR2I(SimdDouble a)
{
    return {
               vmovn_s64(vcvtnq_s64_f64(a.simdInternal_.native_register()))
    };
}

static inline SimdDInt32 gmx_simdcall
cvttR2I(SimdDouble a)
{
    return {
               vmovn_s64(vcvtq_s64_f64(a.simdInternal_.native_register()))
    };
}

static inline SimdDouble gmx_simdcall
cvtI2R(SimdDInt32 a)
{
    return {
               vcvtq_f64_s64(vmovl_s32(a.simdInternal_))
    };
}

static inline SimdDIBool gmx_simdcall
cvtB2IB(SimdDBool a)
{
    return {
               vqmovn_u64(a.simdInternal_.native_register())
    };
}

static inline SimdDBool gmx_simdcall
cvtIB2B(SimdDIBool a)
{
    return {
               vorrq_u64(vmovl_u32(a.simdInternal_), vshlq_n_u64(vmovl_u32(a.simdInternal_), 32))
    };
}

static inline void gmx_simdcall
cvtF2DD(SimdFloat f, SimdDouble *d0, SimdDouble *d1)
{
    d0->simdInternal_ = vcvt_f64_f32(vget_low_f32(f.simdInternal_.native_register()));
    d1->simdInternal_ = vcvt_high_f64_f32(f.simdInternal_.native_register());
}

static inline SimdFloat gmx_simdcall
cvtDD2F(SimdDouble d0, SimdDouble d1)
{
    return {
               vcvt_high_f32_f64(vcvt_f32_f64(d0.simdInternal_.native_register()), d1.simdInternal_.native_register())
    };
}

static inline SimdDouble
frexp(SimdDouble value, SimdDInt32 * exponent)
{
    const float64x2_t exponentMask = float64x2_t( vdupq_n_s64(0x7FF0000000000000LL) );
    const float64x2_t mantissaMask = float64x2_t( vdupq_n_s64(0x800FFFFFFFFFFFFFLL) );

    const int64x2_t   exponentBias = vdupq_n_s64(1022); // add 1 to make our definition identical to frexp()
    const float64x2_t half         = vdupq_n_f64(0.5);
    int64x2_t         iExponent;

    iExponent               = vandq_s64( int64x2_t(value.simdInternal_.native_register()), int64x2_t(exponentMask) );
    iExponent               = vsubq_s64(vshrq_n_s64(iExponent, 52), exponentBias);
    exponent->simdInternal_ = vmovn_s64(iExponent);

    return {
               float64x2_t(vorrq_s64(vandq_s64(int64x2_t(value.simdInternal_.native_register()), int64x2_t(mantissaMask)), int64x2_t(half)))
    };
}

template <MathOptimization opt = MathOptimization::Safe>
static inline SimdDouble
ldexp(SimdDouble value, SimdDInt32 exponent)
{
    const int32x2_t exponentBias = vdup_n_s32(1023);
    int32x2_t       iExponent    = vadd_s32(exponent.simdInternal_.native_register(), exponentBias);
    int64x2_t       iExponent64;

    if (opt == MathOptimization::Safe)
    {
        // Make sure biased argument is not negative
        iExponent = vmax_s32(iExponent, vdup_n_s32(0));
    }

    iExponent64 = vmovl_s32(iExponent);
    iExponent64 = vshlq_n_s64(iExponent64, 52);

    return {
               vmulq_f64(value.simdInternal_.native_register(), float64x2_t(iExponent64))
    };
}


static inline SimdDIBool gmx_simdcall
operator&&(SimdDIBool a, SimdDIBool b)
{
    return {
               vand_u32(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDIBool gmx_simdcall
operator||(SimdDIBool a, SimdDIBool b)
{
    return {
               vorr_u32(a.simdInternal_, b.simdInternal_)
    };
}
#endif


#endif // GMX_SIMD_IMPL_NSIMD_SIMD_DOUBLE_DEFINED_H
