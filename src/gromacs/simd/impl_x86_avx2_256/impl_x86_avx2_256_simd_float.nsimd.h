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

#ifndef GMX_SIMD_IMPL_X86_AVX2_256_SIMD_FLOAT_H
#define GMX_SIMD_IMPL_X86_AVX2_256_SIMD_FLOAT_H

#include "config.h"

#include <nsimd/nsimd.h>
#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/cxx_adv_api_functions.hpp>

#include "gromacs/math/utilities.h"
#include "gromacs/simd/impl_x86_avx_256/impl_x86_avx_256_simd_float.nsimd.h"

namespace gmx
{

class SimdFIBool
{
    public:
        SimdFIBool() {}

        SimdFIBool(bool b) : simdInternal_(nsimd::set1<nsimd::pack<int> >(b ? 4294967295U : 0)) {}

        // Internal utility constructor to simplify return statement
        SimdFIBool(nsimd::pack<int> simd) : simdInternal_(simd) {}

        nsimd::pack<int> simdInternal_;
};

static inline SimdFloat gmx_simdcall
fma(SimdFloat a, SimdFloat b, SimdFloat c)
{
    return {
               nsimd::fma(a.simdInternal_, b.simdInternal_, c.simdInternal_)
    };
}

static inline SimdFloat gmx_simdcall
fms(SimdFloat a, SimdFloat b, SimdFloat c)
{
    return {
               nsimd::fms(a.simdInternal_, b.simdInternal_, c.simdInternal_)
    };
}

static inline SimdFloat gmx_simdcall
fnma(SimdFloat a, SimdFloat b, SimdFloat c)
{
    return {
               nsimd::fnma(a.simdInternal_, b.simdInternal_, c.simdInternal_)
    };
}

static inline SimdFloat gmx_simdcall
fnms(SimdFloat a, SimdFloat b, SimdFloat c)
{
    return {
               nsimd::fnms(a.simdInternal_, b.simdInternal_, c.simdInternal_)
    };
}

static inline SimdFBool gmx_simdcall
testBits(SimdFloat a)
{
    nsimd::pack<int> ia = nsimd::reinterpret<nsimd::pack<int> >(a.simdInternal_);
    nsimd::pack<int> res = nsimd::andnot(nsimd::eq(ia, nsimd::set1<nsimd::pack<int> >(0)), nsimd::eq(ia, ia));

    return {
               nsimd::reinterpret<nsimd::pack<float> >(res)
    };
}

static inline SimdFloat gmx_simdcall
frexp(SimdFloat value, SimdFInt32 * exponent)
{
    const nsimd::pack<float> exponentMask = nsimd::reinterpret<nsimd::pack<float> >(nsimd::set1<nsimd::pack<int> >(2139095040));
    const nsimd::pack<float> mantissaMask = nsimd::reinterpret<nsimd::pack<float> >(nsimd::set1<nsimd::pack<int> >(2155872255U));
    const nsimd::pack<int> exponentBias = nsimd::set1<nsimd::pack<int> >(126); // add 1 to make our definition identical to frexp(
    const nsimd::pack<float> half = nsimd::set1<nsimd::pack<float> >(0.5);
    nsimd::pack<int> iExponent;

    iExponent = nsimd::reinterpret<nsimd::pack<int> >(value.simdInternal_ & exponentMask);
    simdInternal_ = iExponent >> nsimd::cvt<nsimd::pack<int> >(23) - exponentBias;

    return {
               value.simdInternal_ & mantissaMask | half
    };
}

template <MathOptimization opt = MathOptimization::Safe>
static inline SimdFloat gmx_simdcall
ldexp(SimdFloat value, SimdFInt32 exponent)
{
    const nsimd::pack<int> exponentBias = nsimd::set1<nsimd::pack<int> >(127);
    nsimd::pack<int> iExponent = exponent.simdInternal_ + nsimd::cvt<nsimd::pack<int> >(exponentBias);

    if (opt == MathOptimization::Safe)
    {
        // Make sure biased argument is not negativ
        iExponent = nsimd::max(iExponent, nsimd::set1<nsimd::pack<int> >(0));
    }

    iExponent = iExponent << nsimd::cvt<nsimd::pack<int> >(23);
    return {
               value.simdInternal_ * nsimd::reinterpret<nsimd::pack<float> >(iExponent)
    };
}

static inline SimdFInt32 gmx_simdcall
operator&(SimdFInt32 a, SimdFInt32 b)
{
    return {
               a.simdInternal_ & b.simdInternal_
    };
}

static inline SimdFInt32 gmx_simdcall
andNot(SimdFInt32 a, SimdFInt32 b)
{
    return {
               nsimd::andnot(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdFInt32 gmx_simdcall
operator|(SimdFInt32 a, SimdFInt32 b)
{
    return {
               a.simdInternal_ | b.simdInternal_
    };
}

static inline SimdFInt32 gmx_simdcall
operator^(SimdFInt32 a, SimdFInt32 b)
{
    return {
               a.simdInternal_ ^ b.simdInternal_
    };
}

static inline SimdFInt32 gmx_simdcall
operator+(SimdFInt32 a, SimdFInt32 b)
{
    return {
               a.simdInternal_ + b.simdInternal_
    };
}

static inline SimdFInt32 gmx_simdcall
operator-(SimdFInt32 a, SimdFInt32 b)
{
    return {
               a.simdInternal_ - b.simdInternal_
    };
}

static inline SimdFInt32 gmx_simdcall
operator*(SimdFInt32 a, SimdFInt32 b)
{
    return {
               a.simdInternal_ * b.simdInternal_
    };
}

static inline SimdFIBool gmx_simdcall
operator==(SimdFInt32 a, SimdFInt32 b)
{
    return {
               nsimd::eq(a.simdInternal_, b.simdInternal_)
    };
}

static inline SimdFIBool gmx_simdcall
testBits(SimdFInt32 a)
{
    return {
               nsimd::andnot(nsimd::eq(a.simdInternal_, nsimd::set1<nsimd::pack<int> >(0)), nsimd::eq(a.simdInternal_, a.simdInternal_))
    };
}

static inline SimdFIBool gmx_simdcall
operator<(SimdFInt32 a, SimdFInt32 b)
{
    return {
               nsimd::gt(b.simdInternal_, a.simdInternal_)
    };
}

static inline SimdFIBool gmx_simdcall
operator&&(SimdFIBool a, SimdFIBool b)
{
    return {
               a.simdInternal_ & b.simdInternal_
    };
}

static inline SimdFIBool gmx_simdcall
operator||(SimdFIBool a, SimdFIBool b)
{
    return {
               a.simdInternal_ | b.simdInternal_
    };
}

static inline bool gmx_simdcall
anyTrue(SimdFIBool a) { return nsimd::any(a.simdInternal_); }

static inline SimdFInt32 gmx_simdcall
selectByMask(SimdFInt32 a, SimdFIBool mask)
{
    return {
               a.simdInternal_ & mask.simdInternal_
    };
}

static inline SimdFInt32 gmx_simdcall
selectByNotMask(SimdFInt32 a, SimdFIBool mask)
{
    return {
               nsimd::andnot(mask.simdInternal_, a.simdInternal_)
    };
}

static inline SimdFInt32 gmx_simdcall
blend(SimdFInt32 a, SimdFInt32 b, SimdFIBool sel)
{
    return {
               nsimd::blendv(a.simdInternal_, b.simdInternal_, sel.simdInternal_)
    };
}

static inline SimdFIBool gmx_simdcall
cvtB2IB(SimdFBool a)
{
    return {
               nsimd::reinterpret<nsimd::pack<int> >(a.simdInternal_)
    };
}

static inline SimdFBool gmx_simdcall
cvtIB2B(SimdFIBool a)
{
    return {
               nsimd::reinterpret<nsimd::pack<float> >(a.simdInternal_)
    };
}

}      // namespace gm

#endif // GMX_SIMD_IMPL_X86_AVX2_256_SIMD_FLOAT_
