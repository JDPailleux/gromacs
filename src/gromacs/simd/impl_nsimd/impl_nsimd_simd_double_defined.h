#ifndef GMX_SIMD_IMPL_NSIMD_SIMD_DOUBLE_DEFINED_H
#define GMX_SIMD_IMPL_NSIMD_SIMD_DOUBLE_DEFINED_H

#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/cxx_adv_api_functions.hpp>
#include <nsimd/nsimd.h>

#include "impl_nsimd_general.h"

#if defined(NSIMD_SSE2)
  typedef nsimd::pack<int, 1, nsimd::sse2> pack_t;
  typedef nsimd::packl<int, 1, nsimd::sse2> packl_t;
#elif defined(NSIMD_SSE42) || defined(NSIMD_AVX) || defined(NSIMD_AVX2)
  typedef nsimd::pack<int, 1, nsimd::sse42> pack_t;
  typedef nsimd::packl<int, 1, nsimd::sse42> packl_t;
#elif defined(NSIMD_AVX512_KNL) || defined(NSIMD_AVX512_SKYLAKE)
  typedef nsimd::pack<int, 1, nsimd::avx2> pack_t;
  typedef nsimd::packl<int, 1, nsimd::avx2> packl_t;
#elif defined(NSIMD_NEON128)
  typedef nsimd::pack<int, 1, nsimd::neon128> pack_t;
  typedef nsimd::packl<int, 1, nsimd::neon128> packl_t;
#elif defined(NSIMD_AARCH64)
  typedef nsimd::pack<int, 1, nsimd::aarch64> pack_t;
  typedef nsimd::packl<int, 1, nsimd::aarch64> packl_t;
#endif

class SimdDInt32 {
public:
  SimdDInt32() {}

  SimdDInt32(std::int32_t i) : simdInternal_(nsimd::set1<pack_t>(i)) {}

  // Internal utility constructor to simplify return statements
  SimdDInt32(pack_t v) : simdInternal_(v) {}

  pack_t simdInternal_;
};

class SimdDIBool {
public:
  SimdDIBool() {}

  SimdDIBool(bool b)
      : simdInternal_(nsimd::to_logical(nsimd::set1<pack_t>(b ? 1 : 0))) {}

  // Internal utility constructor to simplify return statements
  SimdDIBool(packl_t v) : simdInternal_(v) {}

  packl_t simdInternal_;
};

static inline SimdDInt32 gmx_simdcall simdLoad(const std::int32_t *m,
                                               SimdDInt32Tag) {
  assert(std::size_t(m) % 8 == 0);
#if defined(NSIMD_SSE2) || defined(NSIMD_SSE42)
  return {pack_t(_mm_loadl_epi64((const __m128i *)m))};
#elif defined(NSIMD_NEON128) || defined(NSIMD_AARCH64)
  return {pack_t(vcombine_s32(vld1_s32(m), vdup_n_s32(0)))};
#else
  return {nsimd::loada<pack_t>(m)};
#endif
}

static inline void gmx_simdcall store(std::int32_t *m, SimdDInt32 a) {
  assert(std::size_t(m) % 8 == 0);
#if defined(NSIMD_SSE2) || defined(NSIMD_SSE42)
  _mm_storel_epi64((__m128i *)m, a.simdInternal_.native_register());
#elif defined(NSIMD_NEON128) || defined(NSIMD_AARCH64)
  vst1_s32(m, vget_low_s32(a.simdInternal_.native_register()))};
#else
  nsimd::storea(m, a.simdInternal_);
#endif
}

static inline SimdDInt32 gmx_simdcall simdLoadU(const std::int32_t *m,
                                                SimdDInt32Tag) {
#if defined(NSIMD_SSE2) || defined(NSIMD_SSE42) || defined(NSIMD_NEON128) ||  \
    defined(NSIMD_AARCH64)
  return simdLoad(m, SimdDInt32Tag());
#else
  return {nsimd::loadu<pack_t>(m)};
#endif
}

static inline void gmx_simdcall storeU(std::int32_t *m, SimdDInt32 a) {
#if defined(NSIMD_SSE2) || defined(NSIMD_SSE42) || defined(NSIMD_NEON128) ||  \
    defined(NSIMD_AARCH64)
  store(m, SimdDInt32());
#else
  nsimd::storeu(m, a.simdInternal_);
#endif
}

static inline SimdDInt32 gmx_simdcall setZeroDI() {
  return {nsimd::set1<pack_t>(0)};
}

template <int index>
static inline std::int32_t gmx_simdcall extract(SimdDInt32 a) {
  return _mm_cvtsi128_si32(
      _mm_srli_si128(a.simdInternal_.native_register(), 4 * index));
}

static inline double gmx_simdcall reduce(SimdDouble a) {
  return nsimd::addv(a.simdInternal_);
}

static inline SimdDBool gmx_simdcall testBits(SimdDouble a) {
  return to_logical(a.simdInternal_);
}

static inline SimdDouble frexp(SimdDouble value, SimdDInt32 *exponent) {
#if defined(NSIMD_SSE2) || defined(NSIMD_SSE42)
  // Don't use _mm_set1_epi64x() - on MSVC it is only supported for 64-bit
  // builds
  const __m128d exponentMask = _mm_castsi128_pd(
      _mm_set_epi32(0x7FF00000, 0x00000000, 0x7FF00000, 0x00000000));
  const __m128d mantissaMask = _mm_castsi128_pd(
      _mm_set_epi32(0x800FFFFF, 0xFFFFFFFF, 0x800FFFFF, 0xFFFFFFFF));
  const __m128i exponentBias = _mm_set1_epi32(
      1022); // add 1 to make our definition identical to frexp()
  const __m128d half = _mm_set1_pd(0.5);
  __m128i iExponent;

  iExponent = _mm_castpd_si128(
      _mm_and_pd(value.simdInternal_.native_register(), exponentMask));
  iExponent = _mm_sub_epi32(_mm_srli_epi64(iExponent, 52), exponentBias);
  iExponent = _mm_shuffle_epi32(iExponent, _MM_SHUFFLE(3, 1, 2, 0));
  exponent->simdInternal_ = pack_t(iExponent);

  return {nsimd::pack<double>(_mm_or_pd(
      _mm_and_pd(value.simdInternal_.native_register(), mantissaMask), half))};
#elif defined(NSIMD_AVX) || defined(NSIMD_AVX2)
  const __m256d exponentMask =
      _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FF0000000000000LL));
  const __m256d mantissaMask =
      _mm256_castsi256_pd(_mm256_set1_epi64x(0x800FFFFFFFFFFFFFLL));
  const __m256d half = _mm256_set1_pd(0.5);
  const __m128i exponentBias = _mm_set1_epi32(
      1022); // add 1 to make our definition identical to frexp()
  __m256i iExponent;
  __m128i iExponentLow, iExponentHigh;

  iExponent = _mm256_castpd_si256(
      _mm256_and_pd(value.simdInternal_.native_register(), exponentMask));
  iExponentHigh = _mm256_extractf128_si256(iExponent, 0x1);
  iExponentLow = _mm256_castsi256_si128(iExponent);
  iExponentLow = _mm_srli_epi64(iExponentLow, 52);
  iExponentHigh = _mm_srli_epi64(iExponentHigh, 52);
  iExponentLow = _mm_shuffle_epi32(iExponentLow, _MM_SHUFFLE(1, 1, 2, 0));
  iExponentHigh = _mm_shuffle_epi32(iExponentHigh, _MM_SHUFFLE(2, 0, 1, 1));
  iExponentLow = _mm_or_si128(iExponentLow, iExponentHigh);
  exponent->simdInternal_ = pack_t(_mm_sub_epi32(iExponentLow, exponentBias));

  return {nsimd::pack<double>(_mm256_or_pd(
      _mm256_and_pd(value.simdInternal_.native_register(), mantissaMask),
      half))};
#elif defined(NSIMD_NEON128) || defined(NSIMD_AARCH64)
  const float64x2_t exponentMask =
      float64x2_t(vdupq_n_s64(0x7FF0000000000000LL));
  const float64x2_t mantissaMask =
      float64x2_t(vdupq_n_s64(0x800FFFFFFFFFFFFFLL));

  const int64x2_t exponentBias =
      vdupq_n_s64(1022); // add 1 to make our definition identical to frexp()
  const float64x2_t half = vdupq_n_f64(0.5);
  int64x2_t iExponent;

  iExponent = vandq_s64(int64x2_t(value.simdInternal_.native_register()),
                        int64x2_t(exponentMask));
  iExponent = vsubq_s64(vshrq_n_s64(iExponent, 52), exponentBias);
  exponent->simdInternal_ = pack_t(vmovn_s64(iExponent));

  return {nsimd::pack<double>(float64x2_t(
      vorrq_s64(vandq_s64(int64x2_t(value.simdInternal_.native_register()),
                          int64x2_t(mantissaMask)),
                int64x2_t(half))))};
#else
  __m512d rExponent = _mm512_getexp_pd(value.simdInternal_.native_register());
  __m256i iExponent = _mm512_cvtpd_epi32(rExponent);

  exponent->simdInternal_ =
      pack_t(_mm256_add_epi32(iExponent, _mm256_set1_epi32(1)));

  return {nsimd::pack<double>(
      _mm512_getmant_pd(value.simdInternal_.native_register(),
                        _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src))};
#endif
}

template <MathOptimization opt = MathOptimization::Safe>
static inline SimdDouble ldexp(SimdDouble value, SimdDInt32 exponent) {
#if defined(NSIMD_SSE2)
  const __m128i exponentBias = _mm_set1_epi32(1023);
  __m128i iExponent =
      _mm_add_epi32(exponent.simdInternal_.native_register(), exponentBias);

  if (opt == MathOptimization::Safe) {
    // Make sure biased argument is not negative
    iExponent = _mm_and_si128(iExponent,
                              _mm_cmpgt_epi32(iExponent, _mm_setzero_si128()));
  }

  // After conversion integers will be in slot 0,1. Move them to 0,2 so
  // we can do a 64-bit shift and get them to the dp exponents.
  iExponent = _mm_shuffle_epi32(iExponent, _MM_SHUFFLE(3, 1, 2, 0));
  iExponent = _mm_slli_epi64(iExponent, 52);

  return {nsimd::pack<double>(_mm_mul_pd(value.simdInternal_.native_register(),
                                         _mm_castsi128_pd(iExponent)))};
#elif defined(NSIMD_SSE42)
  const __m128i exponentBias = _mm_set1_epi32(1023);
  __m128i iExponent =
      _mm_add_epi32(exponent.simdInternal_.native_register(), exponentBias);

  if (opt == MathOptimization::Safe) {
    // Make sure biased argument is not negative
    iExponent = _mm_max_epi32(iExponent, _mm_setzero_si128());
  }

  // After conversion integers will be in slot 0,1. Move them to 0,2 so
  // we can do a 64-bit shift and get them to the dp exponents.
  iExponent = _mm_shuffle_epi32(iExponent, _MM_SHUFFLE(3, 1, 2, 0));
  iExponent = _mm_slli_epi64(iExponent, 52);

  return {nsimd::pack<double>(_mm_mul_pd(value.simdInternal_.native_register(),
                                         _mm_castsi128_pd(iExponent)))};
#elif defined(NSIMD_AVX) || defined(NSIMD_AVX2)
  const __m128i exponentBias = _mm_set1_epi32(1023);
  __m128i iExponentLow, iExponentHigh;
  __m256d fExponent;

  iExponentLow =
      _mm_add_epi32(exponent.simdInternal_.native_register(), exponentBias);

  if (opt == MathOptimization::Safe) {
    // Make sure biased argument is not negative
    iExponentLow = _mm_max_epi32(iExponentLow, _mm_setzero_si128());
  }

  iExponentHigh = _mm_shuffle_epi32(iExponentLow, _MM_SHUFFLE(3, 3, 2, 2));
  iExponentLow = _mm_shuffle_epi32(iExponentLow, _MM_SHUFFLE(1, 1, 0, 0));
  iExponentHigh = _mm_slli_epi64(iExponentHigh, 52);
  iExponentLow = _mm_slli_epi64(iExponentLow, 52);
  fExponent = _mm256_castsi256_pd(_mm256_insertf128_si256(
      _mm256_castsi128_si256(iExponentLow), iExponentHigh, 0x1));
  return {nsimd::pack<double>(
      _mm256_mul_pd(value.simdInternal_.native_register(), fExponent))};
#elif defined(NSIMD_NEON128) || defined(NSIMD_AARCH64)
  const int32x2_t exponentBias = vdup_n_s32(1023);
  int32x2_t iExponent = vadd_s32(exponent.simdInternal_, exponentBias);
  int64x2_t iExponent64;

  if (opt == MathOptimization::Safe) {
    // Make sure biased argument is not negative
    iExponent = vmax_s32(iExponent, vdup_n_s32(0));
  }

  iExponent64 = vmovl_s32(iExponent);
  iExponent64 = vshlq_n_s64(iExponent64, 52);

  return {nsimd::pack<double>(vmulq_f64(value.simdInternal_.native_register(),
                                        float64x2_t(iExponent64)))};
#else
  const __m256i exponentBias = _mm256_set1_epi32(1023);
  __m256i iExponent = _mm256_add_epi32(exponent.simdInternal_, exponentBias);
  __m512i iExponent512;

  if (opt == MathOptimization::Safe) {
    // Make sure biased argument is not negative
    iExponent = _mm256_max_epi32(iExponent, _mm256_setzero_si256());
  }

  iExponent512 = _mm512_permutexvar_epi32(
      _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0),
      _mm512_castsi256_si512(iExponent));
  iExponent512 = _mm512_mask_slli_epi32(
      _mm512_setzero_epi32(), avx512Int2Mask(0xAAAA), iExponent512, 20);
  return {nsimd::pack<double>(
      _mm512_mul_pd(_mm512_castsi512_pd(iExponent512),
                    value.simdInternal_.native_register()))};
#endif
}

static inline SimdDouble gmx_simdcall blend(SimdDouble a, SimdDouble b,
                                            SimdDBool sel) {
  return {nsimd::if_else(sel.simdInternal_, b.simdInternal_, a.simdInternal_)};
}

static inline SimdDInt32 gmx_simdcall operator&(SimdDInt32 a, SimdDInt32 b) {
  return {a.simdInternal_ & b.simdInternal_};
}

static inline SimdDInt32 gmx_simdcall andNot(SimdDInt32 a, SimdDInt32 b) {
  return {nsimd::andnotb(a.simdInternal_, b.simdInternal_)};
}

static inline SimdDInt32 gmx_simdcall operator|(SimdDInt32 a, SimdDInt32 b) {
  return {a.simdInternal_ | b.simdInternal_};
}

static inline SimdDInt32 gmx_simdcall operator^(SimdDInt32 a, SimdDInt32 b) {
  return {a.simdInternal_ ^ b.simdInternal_};
}

static inline SimdDInt32 gmx_simdcall operator+(SimdDInt32 a, SimdDInt32 b) {
  return {a.simdInternal_ + b.simdInternal_};
}

static inline SimdDInt32 gmx_simdcall operator-(SimdDInt32 a, SimdDInt32 b) {
  return {a.simdInternal_ - b.simdInternal_};
}

static inline SimdDInt32 gmx_simdcall operator*(SimdDInt32 a, SimdDInt32 b) {
  return {a.simdInternal_ * b.simdInternal_};
}

static inline SimdDIBool gmx_simdcall operator==(SimdDInt32 a, SimdDInt32 b) {
  return {a.simdInternal_ == b.simdInternal_};
}

static inline SimdDIBool gmx_simdcall testBits(SimdDInt32 a) {
  return {to_logical(a.simdInternal_)};
}

static inline SimdDIBool gmx_simdcall operator<(SimdDInt32 a, SimdDInt32 b) {
  return {a.simdInternal_ < b.simdInternal_};
}

static inline SimdDIBool gmx_simdcall operator&&(SimdDIBool a, SimdDIBool b) {
  return {a.simdInternal_ && b.simdInternal_};
}

static inline SimdDIBool gmx_simdcall operator||(SimdDIBool a, SimdDIBool b) {
  return {a.simdInternal_ || b.simdInternal_};
}

static inline bool gmx_simdcall anyTrue(SimdDIBool a) {
#if defined(NSIMD_SSE2) || defined(NSIMD_SSE42)
  return nsimd::any(
      pack_t(a.simdInternal_ & pack_t(_mm_set_epi32(-1, -1, 0, 0))));
#elif defined(NSIMD_NEON128) || defined(NSIMD_AARCH64)
  return nsimd::any(pack_t(
      a.simdInternal_ & pack_t(vcombine_s32(vdup_n_s32(-1), vdup_n_s32(0)))));
#else
  return nsimd::any(a.simdInternal_);
#endif
}

static inline bool gmx_simdcall anyTrue(SimdDBool a) {
  return nsimd::any(a.simdInternal_);
}

static inline SimdDInt32 gmx_simdcall selectByMask(SimdDInt32 a,
                                                   SimdDIBool mask) {
  return {a.simdInternal_ & nsimd::to_mask(mask.simdInternal_)};
}

static inline SimdDInt32 gmx_simdcall selectByNotMask(SimdDInt32 a,
                                                      SimdDIBool mask) {
  return {a.simdInternal_ & nsimd::to_mask(!mask.simdInternal_)};
}

static inline SimdDInt32 gmx_simdcall blend(SimdDInt32 a, SimdDInt32 b,
                                            SimdDIBool sel) {
  return {nsimd::if_else(sel.simdInternal_, b.simdInternal_, a.simdInternal_)};
}

static inline SimdDInt32 gmx_simdcall cvtR2I(SimdDouble a) {
#if defined(NSIMD_SSE2) || defined(NSIMD_SSE42)
  return {pack_t(_mm_cvtpd_epi32(a.simdInternal_.native_register()))};
#elif defined(NSIMD_AVX) || defined(NSIMD_AVX2)
  return {pack_t(_mm256_cvtpd_epi32(a.simdInternal_.native_register()))};
#elif defined(NSIMD_NEON128) || defined(NSIMD_AARCH64)
  return {
      pack_t(vmovn_s64(vcvtnq_s64_f64(a.simdInternal_.native_register())))};
#else
  return {pack_t(_mm512_cvtpd_epi32(a.simdInternal_.native_register()))};
#endif
}

static inline SimdDInt32 gmx_simdcall cvttR2I(SimdDouble a) {
#if defined(NSIMD_SSE2) || defined(NSIMD_SSE42)
  return {pack_t(_mm_cvttpd_epi32(a.simdInternal_.native_register()))};
#elif defined(NSIMD_AVX) || defined(NSIMD_AVX2)
  return {pack_t(_mm256_cvttpd_epi32(a.simdInternal_.native_register()))};
#elif defined(NSIMD_NEON128) || defined(NSIMD_AARCH64)
  return {pack_t(
      vcombine_s32(vmovn_s64(vcvtq_s64_f64(a.simdInternal_.native_register())),
                   vdup_n_s32(0)))};
#else
  return {pack_t(_mm512_cvttpd_epi32(a.simdInternal_.native_register()))};
#endif
}

static inline SimdDouble gmx_simdcall cvtI2R(SimdDInt32 a) {
#if defined(NSIMD_SSE2) || defined(NSIMD_SSE42)
  return {
      nsimd::pack<double>(_mm_cvtepi32_pd(a.simdInternal_.native_register()))};
#elif defined(NSIMD_AVX) || defined(NSIMD_AVX2)
  return {nsimd::pack<double>(
      _mm256_cvtepi32_pd(a.simdInternal_.native_register()))};
#elif defined(NSIMD_NEON128) || defined(NSIMD_AARCH64)
  return {nsimd::pack<double>(vcvtq_f64_s64(
      vmovl_s32(vget_low_s64(a.simdInternal_.native_register()))))};
#else
  return {nsimd::pack<double>(
      _mm512_cvtepi32_pd(a.simdInternal_.native_register()))};
#endif
}

static inline SimdDIBool gmx_simdcall cvtB2IB(SimdDBool a) {
#if defined(NSIMD_SSE2) || defined(NSIMD_SSE42)
  return {packl_t(
      _mm_shuffle_epi32(_mm_castpd_si128(a.simdInternal_.native_register()),
                        _MM_SHUFFLE(2, 0, 2, 0)))};
#elif defined(NSIMD_AVX) || defined(NSIMD_AVX2)
  __m128i a1 = _mm256_extractf128_si256(
      _mm256_castpd_si256(a.simdInternal_.native_register()), 0x1);
  __m128i a0 = _mm256_castsi256_si128(
      _mm256_castpd_si256(a.simdInternal_.native_register()));
  a0 = _mm_shuffle_epi32(a0, _MM_SHUFFLE(2, 0, 2, 0));
  a1 = _mm_shuffle_epi32(a1, _MM_SHUFFLE(2, 0, 2, 0));
  return {packl_t(_mm_blend_epi16(a0, a1, 0xF0))};
#elif defined(NSIMD_NEON128) || defined(NSIMD_AARCH64)
  nsimd::pack<unsigned long> a2 = nsimd::cvt<unsigned long>(a.simdInternal_);
  return {nsimd::packl<double>(
      vcombine_u32(vqmovn_u64(a2.native_register()), vdup_n_u64(0)))};
#else
  return {nsimd::packl<double>((__mmask16)a.simdInternal_)};
#endif
}

static inline SimdDBool gmx_simdcall cvtIB2B(SimdDIBool a) {
#if defined(NSIMD_SSE2) || defined(NSIMD_SSE42)
  return {packl_t(_mm_castsi128_pd(_mm_shuffle_epi32(
      a.simdInternal_.native_register(), _MM_SHUFFLE(1, 1, 0, 0))))};
#elif defined(NSIMD_AVX) || defined(NSIMD_AVX2)
  __m128d lo = _mm_castsi128_pd(_mm_unpacklo_epi32(
      a.simdInternal_.native_register(), a.simdInternal_.native_register()));
  __m128d hi = _mm_castsi128_pd(_mm_unpackhi_epi32(
      a.simdInternal_.native_register(), a.simdInternal_.native_register()));

  return {nsimd::packl<double>(
      _mm256_insertf128_pd(_mm256_castpd128_pd256(lo), hi, 0x1))};
#elif defined(NSIMD_NEON128) || defined(NSIMD_AARCH64)
  return {packl_t(vorrq_u64(vmovl_u32(a.simdInternal_),
                            vshlq_n_u64(vmovl_u32(a.simdInternal_), 32)))};
#else
  return {nsimd::packl<double>((__mmask8)a.simdInternal_)};
#endif
}

static inline void gmx_simdcall cvtF2DD(SimdFloat f, SimdDouble *d0,
                                        SimdDouble *d1) {
  nsimd::packx2<double> tmp =
      nsimd::upcvt<nsimd::packx2<double> >(f.simdInternal_);
  d0->simdInternal_ = tmp.v0;
  d1->simdInternal_ = tmp.v1;
}

static inline SimdFloat gmx_simdcall cvtDD2F(SimdDouble d0, SimdDouble d1) {
  return nsimd::downcvt<nsimd::pack<float> >(d0.simdInternal_,
                                             d1.simdInternal_);
}

static inline SimdDBool gmx_simdcall operator==(SimdDouble a, SimdDouble b) {
  return {a.simdInternal_ == b.simdInternal_};
}

static inline SimdDBool gmx_simdcall operator!=(SimdDouble a, SimdDouble b) {
  return {a.simdInternal_ != b.simdInternal_};
}

static inline SimdDBool gmx_simdcall operator<(SimdDouble a, SimdDouble b) {
  return {a.simdInternal_ < b.simdInternal_};
}

static inline SimdDBool gmx_simdcall operator<=(SimdDouble a, SimdDouble b) {
  return {a.simdInternal_ <= b.simdInternal_};
}

#endif // GMX_SIMD_IMPL_NSIMD_SIMD_DOUBLE_DEFINED_H
