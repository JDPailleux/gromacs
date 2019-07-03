#ifndef GMX_SIMD_IMPL_NSIMD_SIMD_DOUBLE_DEFINED_H
#define GMX_SIMD_IMPL_NSIMD_SIMD_DOUBLE_DEFINED_H

#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/cxx_adv_api_functions.hpp>
#include <nsimd/nsimd.h>


#if (defined(NSIMD_AVX2) || defined(NSIMD_AVX))
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

#elif defined(NSIMD_AARCH64)

#else

#endif


#endif // GMX_SIMD_IMPL_NSIMD_SIMD_DOUBLE_DEFINED_H