/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2014,2015,2017, by the GROMACS development team, led by
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

#ifndef GMX_SIMD_IMPL_NSIMD_SIMD_DOUBLE_H
#define GMX_SIMD_IMPL_NSIMD_SIMD_DOUBLE_H

#include "config.h"

#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/cxx_adv_api_functions.hpp>
#include <nsimd/nsimd.h>

#include "impl_nsimd_simd_float.h"
#include "gromacs/math/utilities.h"

namespace gmx
{

class SimdDouble
{
    public:
        SimdDouble() {}

        SimdDouble(double d) : simdInternal_(nsimd::set1<nsimd::pack<double> >(d)) {}

        // Internal utility constructor to simplify return statements
        SimdDouble(nsimd::pack<double> simd) : simdInternal_(simd) {}

        nsimd::pack<double> simdInternal_;
};

class SimdDInt32
{
    public:
        SimdDInt32() {}

        SimdDInt32(std::int32_t i) : simdInternal_(nsimd::set1<nsimd::pack<int> >(i)) {}

        // Internal utility constructor to simplify return statements
        SimdDInt32(nsimd::pack<int> simd) : simdInternal_(simd) {}

        nsimd::pack<int> simdInternal_;
};

class SimdDBool
{
    public:
        SimdDBool() {}

        SimdDBool(bool b) : simdInternal_(nsimd::reinterpret<nsimd::pack<double> >(nsimd::set1<nsimd::pack<int> >(b ? 4294967295U : 0))) {}

        // Internal utility constructor to simplify return statements
        SimdDBool(nsimd::pack<double> simd) : simdInternal_(simd) {}

        nsimd::pack<double> simdInternal_;
};

class SimdDIBool
{
    public:
        SimdDIBool() {}

        SimdDIBool(bool b) : simdInternal_(nsimd::set1<nsimd::pack<int> >(b ? 4294967295U : 0)) {}

        // Internal utility constructor to simplify return statements
        SimdDIBool(nsimd::pack<int> simd) : simdInternal_(simd) {}

        nsimd::pack<int> simdInternal_;
};    


static inline SimdDouble gmx_simdcall
simdLoad(const double *m, SimdDoubleTag = {})
{
    assert(std::size_t(m) % 32 == 0);
    return {
               nsimd::loada<nsimd::pack<double> >(m)
    };
}

static inline void gmx_simdcall
store(double *m, SimdDouble a)
{
    assert(std::size_t(m) % 32 == 0);
    nsimd::storea(m, a.simdInternal_);
}

static inline SimdDouble gmx_simdcall
simdLoadU(const double *m, SimdDoubleTag = {})
{
    return {
               nsimd::loadu<nsimd::pack<double> >(m)
    };
}

static inline void gmx_simdcall
storeU(double *m, SimdDouble a) { nsimd::storeu(m, a.simdInternal_); }

static inline SimdDouble gmx_simdcall
setZeroD()
{
    return {
               nsimd::set1<nsimd::pack<double> >(0)
    };
}

static inline SimdDInt32 gmx_simdcall
simdLoad(const std::int32_t * m, SimdDInt32Tag /*unused*/)
{
    assert(std::size_t(m) % 16 == 0);
    return {
               nsimd::loada<__m128i >(m)
    };
}

static inline void gmx_simdcall
store(std::int32_t * m, SimdDInt32 a)
{
    assert(std::size_t(m) % 16 == 0);
    nsimd::storea(m, a.simdInternal_);
}

static inline SimdDInt32 gmx_simdcall
simdLoadU(const std::int32_t *m, SimdDInt32Tag /*unused*/)
{
    return {
               nsimd::loadu<__m128i >(m)
    };
}

static inline void gmx_simdcall
storeU(std::int32_t * m, SimdDInt32 a)
{
    nsimd::storeu(m, a.simdInternal_);
}

static inline SimdDInt32 gmx_simdcall
setZeroDI()
{
    return {
               nsimd::set1<__m128i >(0)
    };
}

//#############################
template<int index>
static inline std::int32_t gmx_simdcall
extract(SimdDInt32 a)
{
    return _mm_extract_epi32(nsimd::cvt<__m128i>(a.simdInternal_), index);
}

//#############################

static inline SimdDouble gmx_simdcall
operator&(SimdDouble a, SimdDouble b)
{
    return {
               a.simdInternal_ & b.simdInternal_
    };
}

static inline SimdDouble gmx_simdcall
andNot(SimdDouble a, SimdDouble b)
{
    return {
               nsimd::andnotb(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDouble gmx_simdcall
operator|(SimdDouble a, SimdDouble b)
{
    return {
               a.simdInternal_ | b.simdInternal_
    };
}

static inline SimdDouble gmx_simdcall
operator^(SimdDouble a, SimdDouble b)
{
    return {
               a.simdInternal_ ^ b.simdInternal_
    };
}

static inline SimdDouble gmx_simdcall
operator+(SimdDouble a, SimdDouble b)
{
    return {
               a.simdInternal_ + b.simdInternal_
    };
}

static inline SimdDouble gmx_simdcall
operator-(SimdDouble a, SimdDouble b)
{
    return {
               a.simdInternal_ - b.simdInternal_
    };
}

static inline SimdDouble gmx_simdcall
operator-(SimdDouble x)
{
    return {
               x.simdInternal_ ^ nsimd::set1<nsimd::pack<double> >(-0.)
    };
}

static inline SimdDouble gmx_simdcall
operator*(SimdDouble a, SimdDouble b)
{
    return {
               a.simdInternal_ * b.simdInternal_
    };
}




static inline SimdDouble gmx_simdcall
rsqrt(SimdDouble x)
{
    return {
               nsimd::rsqrt11(x.simdInternal_)
    };
}

static inline SimdDouble gmx_simdcall
rcp(SimdDouble x)
{
    return {
               nsimd::rec11(x.simdInternal_)
    };
}

static inline SimdDouble gmx_simdcall
maskAdd(SimdDouble a, SimdDouble b, SimdDBool m)
{
    return {
               a.simdInternal_ + b.simdInternal_ & m.simdInternal_
    };
}

static inline SimdDouble gmx_simdcall
maskzMul(SimdDouble a, SimdDouble b, SimdDBool m)
{
    return {
               a.simdInternal_ * b.simdInternal_ & m.simdInternal_
    };
}

static inline SimdDouble
maskzFma(SimdDouble a, SimdDouble b, SimdDouble c, SimdDBool m)
{
    return {
               a.simdInternal_ * b.simdInternal_ + c.simdInternal_ & m.simdInternal_
    };
}

static inline SimdDouble
maskzRsqrt(SimdDouble x, SimdDBool m)
{
#ifndef NDEBUG
    simdInternal_ = nsimd::if_else(nsimd::set1<nsimd::pack<double> >(1.), x.simdInternal_, m.simdInternal_);
#endif
    return {
               nsimd::rsqrt11(x.simdInternal_) & m.simdInternal_
    };
}

static inline SimdDouble
maskzRcp(SimdDouble x, SimdDBool m)
{
#ifndef NDEBUG
    simdInternal_ = nsimd::if_else(nsimd::set1<nsimd::pack<double> >(1.), x.simdInternal_, m.simdInternal_);
#endif
    return {
               nsimd::rec11(x.simdInternal_) & m.simdInternal_
    };
}

static inline SimdDouble gmx_simdcall
abs(SimdDouble x)
{
    return {
               nsimd::abs(x.simdInternal_)
    };
}

static inline SimdDouble gmx_simdcall
max(SimdDouble a, SimdDouble b)
{
    return {
               nsimd::max(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDouble gmx_simdcall
min(SimdDouble a, SimdDouble b)
{
    return {
               nsimd::min(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDouble gmx_simdcall
round(SimdDouble x)
{
    return {
               nsimd::round_to_even(x.simdInternal_)
    };
}

static inline SimdDouble gmx_simdcall
trunc(SimdDouble x)
{
    return {
               nsimd::trunc(x.simdInternal_)
    };
}

//##################################
static inline double gmx_simdcall
reduce(SimdDouble a)
{
    __m128d /*Invalid register*/ a0, a1;
    simdInternal_ = a.simdInternal_ + __builtin_ia32_vpermilpd256((__v4df)(__m256d)(a.simdInternal_), (int)(5));
    a0              = _mm256_castpd256_pd128(a.simdInternal_);
    a1              = _mm256_extractf128_pd(a.simdInternal_, 0x1);
    a0              = _mm_add_sd(a0, a1);

    return *reinterpret_cast<double *>(&a0);
}
//#################################

static inline SimdDBool gmx_simdcall
operator==(SimdDouble a, SimdDouble b)
{
    return {
               a.simdInternal_ == b.simdInternal_
    };
}

static inline SimdDBool gmx_simdcall
operator!=(SimdDouble a, SimdDouble b)
{
    return {
               a.simdInternal_ != b.simdInternal_
    };
}

static inline SimdDBool gmx_simdcall
operator<(SimdDouble a, SimdDouble b)
{
    return {
               a.simdInternal_ < b.simdInternal_
    };
}

static inline SimdDBool gmx_simdcall
operator<=(SimdDouble a, SimdDouble b)
{
    return {
               a.simdInternal_ <= b.simdInternal_
    };
}

//####################################
static inline SimdDBool gmx_simdcall
testBits(SimdDouble a)
{
    // Do an or of the low/high 32 bits of each double (so the data is replicated)
    // and then use the same algorithm as we use for single precision
    nsimd::pack<float> tst = nsimd::reinterpret<nsimd::pack<float> >(a.simdInternal_);

    tst = tst | __builtin_ia32_vpermilps256((__v8sf)(__m256)(tst), (int)((((2) << 6) | ((3) << 4) | ((0) << 2) | (1))));
    tst = nsimd::cvt<nsimd::pack<float> >(nsimd::reinterpret<nsimd::pack<int> >(tst));

    return {
               nsimd::reinterpret<nsimd::pack<double> >(nsimd::neq(tst, nsimd::set1<nsimd::pack<float> >(0)))
    };
}
//#################################



static inline SimdDBool gmx_simdcall
operator&&(SimdDBool a, SimdDBool b)
{
    return {
               a.simdInternal_ && b.simdInternal_
    };
}

static inline SimdDBool gmx_simdcall
operator||(SimdDBool a, SimdDBool b)
{
    return {
               a.simdInternal_ || b.simdInternal_
    };
}

static inline bool gmx_simdcall
anyTrue(SimdDBool a) { return nsimd::any(a.simdInternal_); }

static inline SimdDouble gmx_simdcall
selectByMask(SimdDouble a, SimdDBool mask)
{
    return {
               a.simdInternal_ & mask.simdInternal_
    };
}

static inline SimdDouble gmx_simdcall
selectByNotMask(SimdDouble a, SimdDBool mask)
{
    return {
               nsimd::andnotb(mask.simdInternal_, a.simdInternal_)
    };
}

static inline SimdDouble gmx_simdcall
blend(SimdDouble a, SimdDouble b, SimdDBool sel)
{
    return {
               nsimd::if_else1(a.simdInternal_, b.simdInternal_, sel.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
operator&(SimdDInt32 a, SimdDInt32 b)
{
    return {
               a.simdInternal_ & b.simdInternal_
    };
}

static inline SimdDInt32 gmx_simdcall
andNot(SimdDInt32 a, SimdDInt32 b)
{
    return {
               nsimd::andnotb(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
operator|(SimdDInt32 a, SimdDInt32 b)
{
    return {
               a.simdInternal_ | b.simdInternal_
    };
}

static inline SimdDInt32 gmx_simdcall
operator^(SimdDInt32 a, SimdDInt32 b)
{
    return {
               a.simdInternal_ ^ b.simdInternal_
    };
}

static inline SimdDInt32 gmx_simdcall
operator+(SimdDInt32 a, SimdDInt32 b)
{
    return {
               a.simdInternal_ + b.simdInternal_
    };
}

static inline SimdDInt32 gmx_simdcall
operator-(SimdDInt32 a, SimdDInt32 b)
{
    return {
               a.simdInternal_ - b.simdInternal_
    };
}

static inline SimdDInt32 gmx_simdcall
operator*(SimdDInt32 a, SimdDInt32 b)
{
    return {
               a.simdInternal_ * b.simdInternal_
    };
}

static inline SimdDIBool gmx_simdcall
operator==(SimdDInt32 a, SimdDInt32 b)
{
    return {
               nsimd::eq(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdDIBool gmx_simdcall
operator<(SimdDInt32 a, SimdDInt32 b)
{
    return {
               nsimd::lt(a.simdInternal_, b.simdInternal_)
    };
}

//#################################
static inline SimdDIBool gmx_simdcall
testBits(SimdDInt32 a)
{
    __m128i /*Invalid register*/ x = a.simdInternal_;
    __m128i /*Invalid register*/ res =
        nsimd::andnotb(nsimd::eq(nsimd::cvt<nsimd::pack<int>>(x),
                                 nsimd::set1<nsimd::pack<int>>(0)),
                       nsimd::eq(nsimd::cvt<nsimd::pack<int>>(x),
                                 nsimd::cvt<nsimd::pack<int>>(x)));

    return {
               res
    };
}
//###################################

static inline SimdDIBool gmx_simdcall
operator&&(SimdDIBool a, SimdDIBool b)
{
    return {
               a.simdInternal_ && b.simdInternal_
    };
}

static inline SimdDIBool gmx_simdcall
operator||(SimdDIBool a, SimdDIBool b)
{
    return {
               a.simdInternal_ || b.simdInternal_
    };
}

static inline bool gmx_simdcall
anyTrue(SimdDIBool a) { return nsimd::any(a.simdInternal_); }

static inline SimdDInt32 gmx_simdcall
selectByMask(SimdDInt32 a, SimdDIBool mask)
{
    return {
               a.simdInternal_ & mask.simdInternal_
    };
}

static inline SimdDInt32 gmx_simdcall
selectByNotMask(SimdDInt32 a, SimdDIBool mask)
{
    return {
               nsimd::andnotb(mask.simdInternal_, a.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
blend(SimdDInt32 a, SimdDInt32 b, SimdDIBool sel)
{
    return {
               nsimd::if_else1(a.simdInternal_, b.simdInternal_, sel.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
cvtR2I(SimdDouble a)
{
    return {
               nsimd::cvt<nsimd::pack<int> >(a.simdInternal_)
    };
}

static inline SimdDInt32 gmx_simdcall
cvttR2I(SimdDouble a)
{
    return {
            //    _mm256_cvttpd_epi32(a.simdInternal_)
               nsimd::cvt<nsimd::pack<int> >(nsimd::trunc(a))
    };
}

static inline SimdDouble gmx_simdcall
cvtI2R(SimdDInt32 a)
{
    return {
               nsimd::cvt<nsimd::pack<double> >(a.simdInternal_)
    };
}
//###################################
//###################################

static inline SimdDIBool gmx_simdcall
cvtB2IB(SimdDBool a)
{
    __m128i /*Invalid register*/ a1 = __builtin_ia32_vextractf128_si256((__v8si)(__m256i)(_mm256_castpd_si256(a.simdInternal_)), (int)(1));
    __m128i /*Invalid register*/ a0 = _mm256_castsi256_si128(_mm256_castpd_si256(a.simdInternal_));
    a0 = _mm_shuffle_epi32(a0, _MM_SHUFFLE(2, 0, 2, 0));
    a1 = _mm_shuffle_epi32(a1, _MM_SHUFFLE(2, 0, 2, 0));

    return {
               nsimd::if_else1(a0, a1, 0xF0)
    };
}
//###################################
//###################################

static inline SimdDBool gmx_simdcall
cvtIB2B(SimdDIBool a)
{
    __m128d /*Invalid register*/ lo = nsimd::reinterpret<__m128d >(_mm_unpacko_epi32(a.simdInternal_, a.simdInternal_));
    __m128d /*Invalid register*/ hi = nsimd::reinterpret<__m128d >(_mm_unpackhi_epi32(a.simdInternal_, a.simdInternal_));

    return {
               _mm256_insertf128_pd(_mm256_castpd128_pd256(lo), hi, 0x1)
    };
}
//###################################
//###################################

static inline void gmx_simdcall
cvtF2DD(SimdFloat f, SimdDouble *d0, SimdDouble *d1)
{
    simdInternal_ = nsimd::cvt<nsimd::pack<double> >(nsimd::cvt<nsimd::pack<float> >(_mm256_castps256_ps128(f.simdInternal_)));
    simdInternal_ = nsimd::cvt<nsimd::pack<double> >(nsimd::cvt<nsimd::pack<float> >(__builtin_ia32_vextractf128_ps256((__v8sf)(__m256)(f.simdInternal_), (int)(1))));
}
//###################################
//###################################

static inline SimdFloat gmx_simdcall
cvtDD2F(SimdDouble d0, SimdDouble d1)
{
    __m128 /*Invalid register*/ f0 = nsimd::cvt<nsimd::pack<float> >(d0.simdInternal_);
    __m128 /*Invalid register*/ f1 = nsimd::cvt<nsimd::pack<float> >(d1.simdInternal_);
    return {
               _mm256_insertf128_ps(_mm256_castps128_ps256(f0), f1, 0x1)
    };
}
//###################################

//---------------------------------------------------------------------------
static inline SimdDouble gmx_simdcall
fma(SimdDouble a, SimdDouble b, SimdDouble c)
{
    return {
               nsimd::fma(a.simdInternal_, b.simdInternal_, c.simdInternal_)
    };
}

static inline SimdDouble gmx_simdcall
fms(SimdDouble a, SimdDouble b, SimdDouble c)
{
    return {
               nsimd::fms(a.simdInternal_, b.simdInternal_, c.simdInternal_)
    };
}

static inline SimdDouble gmx_simdcall
fnma(SimdDouble a, SimdDouble b, SimdDouble c)
{
    return {
               nsimd::fnma(a.simdInternal_, b.simdInternal_, c.simdInternal_)
    };
}

static inline SimdDouble gmx_simdcall
fnms(SimdDouble a, SimdDouble b, SimdDouble c)
{
    return {
               nsimd::fnms(a.simdInternal_, b.simdInternal_, c.simdInternal_)
    };
}

static inline SimdDBool gmx_simdcall
testBits(SimdDouble a)
{
    nsimd::pack<long> ia = nsimd::reinterpret<nsimd::pack<long> >(a.simdInternal_);
    nsimd::pack<long> res = nsimd::andnot(nsimd::eq(ia, nsimd::set1<nsimd::pack<long> >(0)), nsimd::eq(ia, ia));

    return {
               nsimd::reinterpret<nsimd::pack<double> >(res)
    };
}

static inline SimdDouble
frexp(SimdDouble value, SimdDInt32 * exponent)  //TRADUCTION
{
    const nsimd::pack<double> exponentMask = nsimd::reinterpret<nsimd::pack<double> >(nsimd::set1<nsimd::pack<long> >(9218868437227405312LL));
    const nsimd::pack<double> mantissaMask = nsimd::reinterpret<nsimd::pack<double> >(nsimd::set1<nsimd::pack<long> >(9227875636482146303ULL));
    const nsimd::pack<long> exponentBias = nsimd::set1<nsimd::pack<long> >(1022LL); // add 1 to make our definition identical to frexp(
    const nsimd::pack<double> half = nsimd::set1<nsimd::pack<double> >(0.5);
    nsimd::pack<long> iExponent;
    __m128i /*Invalid register*/ iExponent128;

    iExponent = nsimd::reinterpret<nsimd::pack<long> >(value.simdInternal_ & exponentMask);
    iExponent = iExponent >> nsimd::cvt<nsimd::pack<long> >(52) - exponentBias;
    iExponent = _mm256_shuffle_epi32(iExponent, _MM_SHUFFLE(3, 1, 2, 0));

    iExponent128             = _mm256_extractf128_si256(iExponent, 1);
    exponent->simdInternal_  = _mm_unpacko_epi64(_mm256_castsi256_si128(iExponent), iExponent128);

    return {
               value.simdInternal_ & mantissaMask | half
    };
}

template <MathOptimization opt = MathOptimization::Safe>
static inline SimdDouble
ldexp(SimdDouble value, SimdDInt32 exponent) //TRADUCTION
{
    const __m128i /*Invalid register*/ exponentBias = nsimd::set1<nsimd::pack<int> >(1023);
    __m128i /*Invalid register*/ iExponent = exponent.simdInternal_ + nsimd::cvt<__m128i /*Invalid register*/ >(exponentBias);

    if (opt == MathOptimization::Safe)
    {
        // Make sure biased argument is not negativ
        iExponent = nsimd::cvt<__m128i /*Invalid register*/ >(nsimd::max(nsimd::cvt<nsimd::pack<int> >(iExponent), nsimd::set1<nsimd::pack<int> >(0)));
    }

    nsimd::pack<long> iExponent256 = nsimd::cvt<nsimd::pack<long> >(iExponent) << nsimd::cvt<nsimd::pack<long> >(52);
    return {
               value.simdInternal_ * nsimd::reinterpret<nsimd::pack<double> >(iExponent256)
    };
}

}      // namespace gm

#endif // GMX_SIMD_IMPL_NSIMD_SIMD_DOUBLE_H
