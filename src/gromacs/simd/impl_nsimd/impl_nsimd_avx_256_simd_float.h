/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2014,2015,2016,2017,2018, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org
 */

#ifndef GMX_SIMD_IMPL_X86_AVX_256_SIMD_FLOAT_H
#define GMX_SIMD_IMPL_X86_AVX_256_SIMD_FLOAT_H

#include "config.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/cxx_adv_api_functions.hpp>
#include <nsimd/nsimd.h>

#include "gromacs/math/utilities.h"

namespace gmx
{

class SimdFloat
{
    public:
        SimdFloat() {}

        SimdFloat(float f) : simdInternal_(nsimd::set1<nsimd::pack<float> >(f)) {}

        // Internal utility constructor to simplify return statement
        SimdFloat(nsimd::pack<float> simd) : simdInternal_(simd) {}

        nsimd::pack<float> simdInternal_;
};

class SimdFInt32
{
    public:
        SimdFInt32() {}

        SimdFInt32(std::int32_t i) : simdInternal_(nsimd::set1<nsimd::pack<int> >(i)) {}

        // Internal utility constructor to simplify return statement
        SimdFInt32(nsimd::pack<int> simd) : simdInternal_(simd) {}

        nsimd::pack<int> simdInternal_;
};

class SimdFBool
{
    public:
        SimdFBool() {}

        SimdFBool(bool b) : simdInternal_(nsimd::reinterpret<nsimd::pack<float> >(nsimd::set1<nsimd::pack<int> >(b ? 4294967295U : 0))) {}

        // Internal utility constructor to simplify return statement
        SimdFBool(nsimd::pack<float> simd) : simdInternal_(simd) {}

        nsimd::pack<float> simdInternal_;
};

static inline SimdFloat gmx_simdcall
simdLoad(const float *m, SimdFloatTag  /*unused*/ = {})
{
    assert(std::size_t(m) % 32 == 0);
    return {
               nsimd::loada<nsimd::pack<float> >(m)
    };
}

static inline void gmx_simdcall
store(float *m, SimdFloat a)
{
    assert(std::size_t(m) % 32 == 0);
    nsimd::storea(m, a.simdInternal_);
}

static inline SimdFloat gmx_simdcall
simdLoadU(const float *m, SimdFloatTag  /*unused*/ = {})
{
    return {
               nsimd::loadu<nsimd::pack<float> >(m)
    };
}

static inline void gmx_simdcall
storeU(float *m, SimdFloat a)
{
    nsimd::storeu(m, a.simdInternal_);
}

static inline SimdFloat gmx_simdcall
setZeroF()
{
    return {
               nsimd::set1<nsimd::pack<float> >(0)
    };
}

static inline SimdFInt32 gmx_simdcall
simdLoad(const std::int32_t * m, SimdFInt32Tag /*unused*/)
{
    assert(std::size_t(m) % 32 == 0);
    return {
               nsimd::loada<nsimd::pack<int> >(m)
    };
}

static inline void gmx_simdcall
store(std::int32_t * m, SimdFInt32 a)
{
    assert(std::size_t(m) % 32 == 0);
    nsimd::storea(m, a.simdInternal_);
}

static inline SimdFInt32 gmx_simdcall
simdLoadU(const std::int32_t *m, SimdFInt32Tag /*unused*/)
{
    return {
               nsimd::loadu<nsimd::pack<int> >(m)
    };
}

static inline void gmx_simdcall
storeU(std::int32_t * m, SimdFInt32 a)
{
    nsimd::storeu(m, a.simdInternal_);
}

static inline SimdFInt32 gmx_simdcall
setZeroFI()
{
    return {
               nsimd::set1<nsimd::pack<int> >(0)
    };
}

template<int index>
static inline std::int32_t gmx_simdcall
extract(SimdFInt32 a)
{
    return _mm_extract_epi32(_mm256_extractf128_si256(a.simdInternal_, index>>2), index & 0x3);
}

static inline SimdFloat gmx_simdcall
operator&(SimdFloat a, SimdFloat b)
{
    return {
               a.simdInternal_ & b.simdInternal_
    };
}

static inline SimdFloat gmx_simdcall
andNot(SimdFloat a, SimdFloat b)
{
    return {
               nsimd::andnotb(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdFloat gmx_simdcall
operator|(SimdFloat a, SimdFloat b)
{
    return {
               a.simdInternal_ | b.simdInternal_
    };
}

static inline SimdFloat gmx_simdcall
operator^(SimdFloat a, SimdFloat b)
{
    return {
               a.simdInternal_ ^ b.simdInternal_
    };
}

static inline SimdFloat gmx_simdcall
operator+(SimdFloat a, SimdFloat b)
{
    return {
               a.simdInternal_ + b.simdInternal_
    };
}

static inline SimdFloat gmx_simdcall
operator-(SimdFloat a, SimdFloat b)
{
    return {
               a.simdInternal_ - b.simdInternal_
    };
}

static inline SimdFloat gmx_simdcall
operator-(SimdFloat x)
{
    return {
               x.simdInternal_ ^ nsimd::set1<nsimd::pack<float> >(-0.F)
    };
}

static inline SimdFloat gmx_simdcall
operator*(SimdFloat a, SimdFloat b)
{
    return {
               a.simdInternal_ * b.simdInternal_
    };
}

static inline SimdFloat gmx_simdcall
fma(SimdFloat a, SimdFloat b, SimdFloat c)
{
    return {
               a.simdInternal_ * b.simdInternal_ + c.simdInternal_
    };
}

static inline SimdFloat gmx_simdcall
fms(SimdFloat a, SimdFloat b, SimdFloat c)
{
    return {
               a.simdInternal_ * b.simdInternal_ - c.simdInternal_
    };
}

static inline SimdFloat gmx_simdcall
fnma(SimdFloat a, SimdFloat b, SimdFloat c)
{
    return {
               c.simdInternal_ - a.simdInternal_ * b.simdInternal_
    };
}

static inline SimdFloat gmx_simdcall
fnms(SimdFloat a, SimdFloat b, SimdFloat c)
{
    return {
               nsimd::set1<nsimd::pack<float> >(0) - a.simdInternal_ * b.simdInternal_ + c.simdInternal_
    };
}

static inline SimdFloat gmx_simdcall
rsqrt(SimdFloat x)
{
    return {
               nsimd::rsqrt11(x.simdInternal_)
    };
}

static inline SimdFloat gmx_simdcall
rcp(SimdFloat x)
{
    return {
               nsimd::rec11(x.simdInternal_)
    };
}

static inline SimdFloat gmx_simdcall
maskAdd(SimdFloat a, SimdFloat b, SimdFBool m)
{
    return {
               a.simdInternal_ + b.simdInternal_ & m.simdInternal_
    };
}

static inline SimdFloat gmx_simdcall
maskzMul(SimdFloat a, SimdFloat b, SimdFBool m)
{
    return {
               a.simdInternal_ * b.simdInternal_ & m.simdInternal_
    };
}

static inline SimdFloat
maskzFma(SimdFloat a, SimdFloat b, SimdFloat c, SimdFBool m)
{
    return {
               a.simdInternal_ * b.simdInternal_ + c.simdInternal_ & m.simdInternal_
    };
}

static inline SimdFloat
maskzRsqrt(SimdFloat x, SimdFBool m)
{
#ifndef NDEBUG
    simdInternal_ = nsimd::if_else(nsimd::set1<nsimd::pack<float> >(1.F), x.simdInternal_, m.simdInternal_);
#endif
    return {
               nsimd::rsqrt11(x.simdInternal_) & m.simdInternal_
    };
}

static inline SimdFloat
maskzRcp(SimdFloat x, SimdFBool m)
{
#ifndef NDEBUG
    simdInternal_ = nsimd::if_else(nsimd::set1<nsimd::pack<float> >(1.F), x.simdInternal_, m.simdInternal_);
#endif
    return {
               nsimd::rec11(x.simdInternal_) & m.simdInternal_
    };
}

static inline SimdFloat gmx_simdcall
abs(SimdFloat x)
{
    return {
               nsimd::andnotb(nsimd::set1<nsimd::pack<float> >(-0.F), x.simdInternal_)
    };
}

static inline SimdFloat gmx_simdcall
max(SimdFloat a, SimdFloat b)
{
    return {
               nsimd::max(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdFloat gmx_simdcall
min(SimdFloat a, SimdFloat b)
{
    return {
               nsimd::min(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdFloat gmx_simdcall
round(SimdFloat x)
{
    return {
               nsimd::round(x.simdInternal_, 0 | 0)
    };
}

static inline SimdFloat gmx_simdcall
trunc(SimdFloat x)
{
    return {
               nsimd::round(x.simdInternal_, 0 | 3)
    };
}

static inline SimdFloat gmx_simdcall
frexp(SimdFloat value, SimdFInt32 * exponent)
{
    const nsimd::pack<float> exponentMask = nsimd::reinterpret<nsimd::pack<float> >(nsimd::set1<nsimd::pack<int> >(2139095040));
    const nsimd::pack<float> mantissaMask = nsimd::reinterpret<nsimd::pack<float> >(nsimd::set1<nsimd::pack<int> >(2155872255U));
    const nsimd::pack<float> half = nsimd::set1<nsimd::pack<float> >(0.5);
    const __m128i /*Invalid register*/ exponentBias = nsimd::set1<nsimd::pack<int> >(126);  // add 1 to make our definition identical to frexp(
    nsimd::pack<int> iExponent;
    __m128i /*Invalid register*/ iExponentLow, iExponentHigh;

    iExponent = nsimd::reinterpret<nsimd::pack<int> >(value.simdInternal_ & exponentMask);
    iExponentHigh           = _mm256_extractf128_si256(iExponent, 0x1);
    iExponentLow            = _mm256_castsi256_si128(iExponent);
    iExponentLow = nsimd::cvt<__m128i /*Invalid register*/ >(nsimd::cvt<nsimd::pack<int> >(iExponentLow) >> nsimd::cvt<nsimd::pack<int> >(23));
    iExponentHigh = nsimd::cvt<__m128i /*Invalid register*/ >(nsimd::cvt<nsimd::pack<int> >(iExponentHigh) >> nsimd::cvt<nsimd::pack<int> >(23));
    iExponentLow = nsimd::cvt<__m128i /*Invalid register*/ >(nsimd::cvt<nsimd::pack<int> >(iExponentLow) - nsimd::cvt<nsimd::pack<int> >(exponentBias));
    iExponentHigh = nsimd::cvt<__m128i /*Invalid register*/ >(nsimd::cvt<nsimd::pack<int> >(iExponentHigh) - nsimd::cvt<nsimd::pack<int> >(exponentBias));
    iExponent               = _mm256_castsi128_si256(iExponentLow);
    exponent->simdInternal_ = _mm256_insertf128_si256(iExponent, iExponentHigh, 0x1);

    return {
               value.simdInternal_ & mantissaMask | half
    };

}

template <MathOptimization opt = MathOptimization::Safe>
static inline SimdFloat gmx_simdcall
ldexp(SimdFloat value, SimdFInt32 exponent)
{
    const __m128i /*Invalid register*/ exponentBias = nsimd::set1<nsimd::pack<int> >(127);
    nsimd::pack<int> iExponent;
    __m128i /*Invalid register*/ iExponentLow, iExponentHigh;

    iExponentHigh = _mm256_extractf128_si256(exponent.simdInternal_, 0x1);
    iExponentLow  = _mm256_castsi256_si128(exponent.simdInternal_);

    iExponentLow = nsimd::cvt<__m128i /*Invalid register*/ >(nsimd::cvt<nsimd::pack<int> >(iExponentLow) + nsimd::cvt<nsimd::pack<int> >(exponentBias));
    iExponentHigh = nsimd::cvt<__m128i /*Invalid register*/ >(nsimd::cvt<nsimd::pack<int> >(iExponentHigh) + nsimd::cvt<nsimd::pack<int> >(exponentBias));

    if (opt == MathOptimization::Safe)
    {
        // Make sure biased argument is not negativ
        iExponentLow = nsimd::cvt<__m128i /*Invalid register*/ >(nsimd::max(nsimd::cvt<nsimd::pack<int> >(iExponentLow), nsimd::set1<nsimd::pack<int> >(0)));
        iExponentHigh = nsimd::cvt<__m128i /*Invalid register*/ >(nsimd::max(nsimd::cvt<nsimd::pack<int> >(iExponentHigh), nsimd::set1<nsimd::pack<int> >(0)));
    }

    iExponentLow = nsimd::cvt<__m128i /*Invalid register*/ >(nsimd::cvt<nsimd::pack<int> >(iExponentLow) << nsimd::cvt<nsimd::pack<int> >(23));
    iExponentHigh = nsimd::cvt<__m128i /*Invalid register*/ >(nsimd::cvt<nsimd::pack<int> >(iExponentHigh) << nsimd::cvt<nsimd::pack<int> >(23));
    iExponent     = _mm256_castsi128_si256(iExponentLow);
    iExponent     = _mm256_insertf128_si256(iExponent, iExponentHigh, 0x1);
    return {
               value.simdInternal_ * nsimd::reinterpret<nsimd::pack<float> >(iExponent)
    };
}

static inline float gmx_simdcall
reduce(SimdFloat a)
{
    __m128 /*Invalid register*/ t0;
    t0 = _mm256_castps256_ps128(a.simdInternal_) + __builtin_ia32_vextractf128_ps256((__v8sf)(__m256)(a.simdInternal_), (int)(1));
    t0 = t0 + __builtin_ia32_vpermilps((__v4sf)(__m128)(t0), (int)((((1) << 6) | ((0) << 4) | ((3) << 2) | (2))));
    t0 = _mm_add_ss(t0, _mm_permute_ps(t0, _MM_SHUFFLE(0, 3, 2, 1)));
    return *reinterpret_cast<float *>(&t0);
}

static inline SimdFBool gmx_simdcall
operator==(SimdFloat a, SimdFloat b)
{
    return {
               nsimd::eq(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdFBool gmx_simdcall
operator!=(SimdFloat a, SimdFloat b)
{
    return {
               nsimd::neq(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdFBool gmx_simdcall
operator<(SimdFloat a, SimdFloat b)
{
    return {
               nsimd::lt(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdFBool gmx_simdcall
operator<=(SimdFloat a, SimdFloat b)
{
    return {
               nsimd::leq(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdFBool gmx_simdcall
testBits(SimdFloat a)
{
    nsimd::pack<float> tst = nsimd::cvt<nsimd::pack<float> >(nsimd::reinterpret<nsimd::pack<int> >(a.simdInternal_));

    return {
               nsimd::neq(tst, nsimd::set1<nsimd::pack<float> >(0))
    };
}

static inline SimdFBool gmx_simdcall
operator&&(SimdFBool a, SimdFBool b)
{
    return {
               a.simdInternal_ & b.simdInternal_
    };
}

static inline SimdFBool gmx_simdcall
operator||(SimdFBool a, SimdFBool b)
{
    return {
               a.simdInternal_ | b.simdInternal_
    };
}

static inline bool gmx_simdcall
anyTrue(SimdFBool a) { return nsimd::any(a.simdInternal_); }

static inline SimdFloat gmx_simdcall
selectByMask(SimdFloat a, SimdFBool mask)
{
    return {
               a.simdInternal_ & mask.simdInternal_
    };
}

static inline SimdFloat gmx_simdcall
selectByNotMask(SimdFloat a, SimdFBool mask)
{
    return {
               nsimd::andnot(mask.simdInternal_, a.simdInternal_)
    };
}

static inline SimdFloat gmx_simdcall
blend(SimdFloat a, SimdFloat b, SimdFBool sel)
{
    return {
               nsimd::if_else(a.simdInternal_, b.simdInternal_, sel.simdInternal_)
    };
}

static inline SimdFInt32 gmx_simdcall
cvtR2I(SimdFloat a)
{
    return {
               nsimd::cvt<nsimd::pack<int> >(a.simdInternal_)
    };
}

static inline SimdFInt32 gmx_simdcall
cvttR2I(SimdFloat a)
{
    return {
               _mm256_cvttps_epi32(a.simdInternal_)
    };
}

static inline SimdFloat gmx_simdcall
cvtI2R(SimdFInt32 a)
{
    return {
               nsimd::cvt<nsimd::pack<float> >(a.simdInternal_)
    };
}

}      // namespace gm

#endif // GMX_SIMD_IMPL_X86_AVX_256_SIMD_FLOAT_
