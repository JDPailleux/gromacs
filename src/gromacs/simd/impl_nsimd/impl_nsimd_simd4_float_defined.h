
#ifndef GMX_IMPL_NSIMD_SIMD4_FLOAT_DEFINED
#define GMX_IMPL_NSIMD_SIMD4_FLOAT_DEFINED

#include "config.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/cxx_adv_api_functions.hpp>
#include <nsimd/nsimd.h>

#include "impl_nsimd_general.h"
#include "impl_nsimd_simd4_float_defined.h"

namespace gmx {

#if defined(NSIMD_SSE2) || defined(NSIMD_SSE42) || defined(NSIMD_NEON128) ||  \
    defined(NSIMD_AARCH64)
typedef nsimd::pack<float> pack4f_t;
typedef nsimd::packl<float> packl4f_t;
#else
typedef nsimd::pack<float, 1, nsimd::sse42> pack4f_t;
typedef nsimd::packl<float, 1, nsimd::sse42> packl4f_t;
#endif

class Simd4Float {
public:
  Simd4Float() {}

  Simd4Float(float f) : simdInternal_(nsimd::set1<pack4f_t>(f)) {}

  // Internal utility constructor to simplify return statements
  Simd4Float(pack4f_t v) : simdInternal_(v) {}

  pack4f_t simdInternal_;
};

class Simd4FBool {
public:
  Simd4FBool() {}

  //! \brief Construct from scalar bool
  Simd4FBool(bool b)
      : simdInternal_(
            nsimd::to_logical(nsimd::set1<pack4f_t>(b ? 1.0f : 0.0f))) {}

  // Internal utility constructor to simplify return statements
  Simd4FBool(packl4f_t v) : simdInternal_(v) {}

  packl4f_t simdInternal_;
};

static inline Simd4Float gmx_simdcall load4(const float *m) {
  assert(std::size_t(m) % 16 == 0);
  return {nsimd::loada<pack4f_t>(m)};
}

static inline void gmx_simdcall store4(float *m, Simd4Float a) {
  assert(std::size_t(m) % 16 == 0);
  nsimd::storea(m, a.simdInternal_);
}

static inline Simd4Float gmx_simdcall load4U(const float *m) {
  return {nsimd::loadu<pack4f_t>(m)};
}

static inline void gmx_simdcall store4U(float *m, Simd4Float a) {
  nsimd::storeu(m, a.simdInternal_);
}

static inline Simd4Float gmx_simdcall simd4SetZeroF() {
  return {nsimd::set1<pack4f_t>(0.0f)};
}

static inline Simd4Float gmx_simdcall operator&(Simd4Float a, Simd4Float b) {
  return {a.simdInternal_ & b.simdInternal_};
}

static inline Simd4Float gmx_simdcall andNot(Simd4Float a, Simd4Float b) {
  return {nsimd::andnotb(b.simdInternal_, a.simdInternal_)};
}

static inline Simd4Float gmx_simdcall operator|(Simd4Float a, Simd4Float b) {
  return {a.simdInternal_ | b.simdInternal_};
}

static inline Simd4Float gmx_simdcall operator^(Simd4Float a, Simd4Float b) {
  return {a.simdInternal_ ^ b.simdInternal_};
}

static inline Simd4Float gmx_simdcall operator+(Simd4Float a, Simd4Float b) {
  return {a.simdInternal_ + b.simdInternal_};
}

static inline Simd4Float gmx_simdcall operator-(Simd4Float a, Simd4Float b) {
  return {a.simdInternal_ - b.simdInternal_};
}

static inline Simd4Float gmx_simdcall operator-(Simd4Float x) {
  return {-x.simdInternal_};
}

static inline Simd4Float gmx_simdcall operator*(Simd4Float a, Simd4Float b) {
  return {a.simdInternal_ * b.simdInternal_};
}

static inline Simd4Float gmx_simdcall fma(Simd4Float a, Simd4Float b,
                                          Simd4Float c) {
  return {nsimd::fma(a.simdInternal_, b.simdInternal_, c.simdInternal_)};
  //return {//    _mm_add_ps(_mm_mul_ps(a.simdInternal_, b.simdInternal_),
  //        //    c.simdInternal_)
  //        _mm_fmadd_ps(a.simdInternal_, b.simdInternal_, c.simdInternal_)};
}

static inline Simd4Float gmx_simdcall fms(Simd4Float a, Simd4Float b,
                                          Simd4Float c) {
  return {nsimd::fms(a.simdInternal_, b.simdInternal_, c.simdInternal_)};
  //return {_mm_sub_ps(_mm_mul_ps(a.simdInternal_, b.simdInternal_),
  //                   c.simdInternal_)};
}

static inline Simd4Float gmx_simdcall fnma(Simd4Float a, Simd4Float b,
                                           Simd4Float c) {
  return {nsimd::fnma(a.simdInternal_, b.simdInternal_, c.simdInternal_)};
  //return {_mm_sub_ps(c.simdInternal_,
  //                   _mm_mul_ps(a.simdInternal_, b.simdInternal_))};
}

static inline Simd4Float gmx_simdcall fnms(Simd4Float a, Simd4Float b,
                                           Simd4Float c) {
  return {nsimd::fnms(a.simdInternal_, b.simdInternal_, c.simdInternal_)};
  //return {_mm_sub_ps(_mm_setzero_ps(),
  //                   _mm_add_ps(_mm_mul_ps(a.simdInternal_, b.simdInternal_),
  //                              c.simdInternal_))};
}

static inline Simd4Float gmx_simdcall rsqrt(Simd4Float x) {
  return {nsimd::rsqrt11(x.simdInternal_)};
}

static inline Simd4Float gmx_simdcall abs(Simd4Float x) {
  return {nsimd::abs(x.simdInternal_)};
}

static inline Simd4Float gmx_simdcall max(Simd4Float a, Simd4Float b) {
  return {nsimd::max(a.simdInternal_, b.simdInternal_)};
}

static inline Simd4Float gmx_simdcall min(Simd4Float a, Simd4Float b) {
  return {nsimd::min(a.simdInternal_, b.simdInternal_)};
}

static inline Simd4Float gmx_simdcall round(Simd4Float x) {
  return {nsimd::round_to_even(x.simdInternal_)};
}

static inline Simd4Float gmx_simdcall trunc(Simd4Float x) {
  return {nsimd::trunc(x.simdInternal_)};
}

static inline float gmx_simdcall dotProduct(Simd4Float a, Simd4Float b) {
#if defined(NSIMD_SSE2)
  __m128 c, d;
  c = _mm_mul_ps(a.simdInternal_.native_register(),
                 b.simdInternal_.native_register());
  d = _mm_add_ps(c, _mm_permute_ps(c, _MM_SHUFFLE(2, 1, 2, 1)));
  d = _mm_add_ps(d, _mm_permute_ps(c, _MM_SHUFFLE(3, 2, 3, 2)));
  return *reinterpret_cast<float *>(&d);
#elif defined(NSIMD_ARM)
  Simd4Float c;
  c = a * b;
  /* set 4th element to 0, then add all of them */
  c.simdInternal_ =
      pack4f_t(vsetq_lane_f32(0.0f, c.simdInternal_.native_register(), 3));
  return reduce(c);
#else
  __m128 res = _mm_dp_ps(a.simdInternal_.native_register(),
                         b.simdInternal_.native_register(), 0x71);
  return *reinterpret_cast<float *>(&res);
#endif
}

static inline void gmx_simdcall transpose(Simd4Float *v0, Simd4Float *v1,
                                          Simd4Float *v2, Simd4Float *v3) {
#ifdef NSIMD_X86
  __m128 a = v0->simdInternal_.native_register();
  __m128 b = v1->simdInternal_.native_register();
  __m128 c = v2->simdInternal_.native_register();
  __m128 d = v3->simdInternal_.native_register();
  _MM_TRANSPOSE4_PS(a, b, c, d);
  v0->simdInternal_ = pack4f_t(a);
  v1->simdInternal_ = pack4f_t(b);
  v2->simdInternal_ = pack4f_t(c);
  v3->simdInternal_ = pack4f_t(d);
#else
  float32x4x2_t t0 = vuzpq_f32(v0->simdInternal_.native_register(),
                               v2->simdInternal_.native_register());
  float32x4x2_t t1 = vuzpq_f32(v1->simdInternal_.native_register(),
                               v3->simdInternal_.native_register());
  float32x4x2_t t2 = vtrnq_f32(t0.val[0], t1.val[0]);
  float32x4x2_t t3 = vtrnq_f32(t0.val[1], t1.val[1]);
  v0->simdInternal_ = t2.val[0];
  v1->simdInternal_ = t3.val[0];
  v2->simdInternal_ = t2.val[1];
  v3->simdInternal_ = t3.val[1];
#endif
}

static inline Simd4FBool gmx_simdcall operator==(Simd4Float a, Simd4Float b) {
  return {a.simdInternal_ == b.simdInternal_};
}

static inline Simd4FBool gmx_simdcall operator!=(Simd4Float a, Simd4Float b) {
  return {a.simdInternal_ != b.simdInternal_};
}

static inline Simd4FBool gmx_simdcall operator<(Simd4Float a, Simd4Float b) {
  return {a.simdInternal_ < b.simdInternal_};
}

static inline Simd4FBool gmx_simdcall operator<=(Simd4Float a, Simd4Float b) {
  return {a.simdInternal_ <= b.simdInternal_};
}

static inline Simd4FBool gmx_simdcall operator&&(Simd4FBool a, Simd4FBool b) {
  return {a.simdInternal_ && b.simdInternal_};
}

static inline Simd4FBool gmx_simdcall operator||(Simd4FBool a, Simd4FBool b) {
  return {a.simdInternal_ || b.simdInternal_};
}

static inline bool gmx_simdcall anyTrue(Simd4FBool a) {
  return nsimd::any(a.simdInternal_);
}

static inline Simd4Float gmx_simdcall selectByMask(Simd4Float a,
                                                   Simd4FBool mask) {
  return {a.simdInternal_ & nsimd::to_mask(mask.simdInternal_)};
}

static inline Simd4Float gmx_simdcall selectByNotMask(Simd4Float a,
                                                      Simd4FBool mask) {
  return {a.simdInternal_ & nsimd::to_mask(!mask.simdInternal_)};
}

static inline Simd4Float gmx_simdcall blend(Simd4Float a, Simd4Float b,
                                            Simd4FBool sel) {
  return {nsimd::if_else(sel.simdInternal_, b.simdInternal_, a.simdInternal_)};
}

static inline float gmx_simdcall reduce(Simd4Float a) {
  return {nsimd::addv(a.simdInternal_)};
}

} // namespace gmx

#endif // GMX_IMPL_NSIMD_SIMD4_FLOAT_DEFINED

