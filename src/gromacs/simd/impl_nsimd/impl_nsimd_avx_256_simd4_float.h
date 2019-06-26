/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2014,2015, by the GROMACS development team, led by
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

#ifndef GMX_SIMD_IMPL_X86_AVX_256_SIMD4_FLOAT_H
#define GMX_SIMD_IMPL_X86_AVX_256_SIMD4_FLOAT_H

#include "config.h"

#include <cassert>
#include <cstddef>

#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/cxx_adv_api_functions.hpp>
#include <nsimd/nsimd.h>

namespace gmx
{

class Simd4Float
{
    public:
        Simd4Float() {}

        Simd4Float(float f) : simdInternal_(nsimd::set1<nsimd::pack<float> >(f)) {}

        // Internal utility constructor to simplify return statement
        Simd4Float(__m128 /*Invalid register*/ simd) : simdInternal_(simd) {}

        __m128 /*Invalid register*/ simdInternal_;
};

class Simd4FBool
{
    public:
        Simd4FBool() {}

        //! \brief Construct from scalar boo
        Simd4FBool(bool b) : simdInternal_(nsimd::reinterpret<nsimd::pack<float> >(nsimd::set1<nsimd::pack<int> >(b ? 4294967295U : 0))) {}

        // Internal utility constructor to simplify return statement
        Simd4FBool(__m128 /*Invalid register*/ simd) : simdInternal_(simd) {}

        __m128 /*Invalid register*/ simdInternal_;
};

static inline Simd4Float gmx_simdcall
load4(const float *m)
{
    assert(std::size_t(m) % 16 == 0);
    return {
               nsimd::loada<nsimd::pack<float> >(m)
    };
}

static inline void gmx_simdcall
store4(float *m, Simd4Float a)
{
    assert(std::size_t(m) % 16 == 0);
    nsimd::storea(m, a.simdInternal_);
}

static inline Simd4Float gmx_simdcall
load4U(const float *m)
{
    return {
               nsimd::loadu<nsimd::pack<float> >(m)
    };
}

static inline void gmx_simdcall
store4U(float *m, Simd4Float a)
{
    nsimd::storeu(m, a.simdInternal_);
}

static inline Simd4Float gmx_simdcall
simd4SetZeroF()
{
    return {
               nsimd::set1<nsimd::pack<float> >(0)
    };
}

static inline Simd4Float gmx_simdcall
operator&(Simd4Float a, Simd4Float b)
{
    return {
               a.simdInternal_ & b.simdInternal_
    };
}

static inline Simd4Float gmx_simdcall
andNot(Simd4Float a, Simd4Float b)
{
    return {
               nsimd::andnotb(a.simdInternal_, b.simdInternal_)
    };
}

static inline Simd4Float gmx_simdcall
operator|(Simd4Float a, Simd4Float b)
{
    return {
               a.simdInternal_ | b.simdInternal_
    };
}

static inline Simd4Float gmx_simdcall
operator^(Simd4Float a, Simd4Float b)
{
    return {
               a.simdInternal_ ^ b.simdInternal_
    };
}

static inline Simd4Float gmx_simdcall
operator+(Simd4Float a, Simd4Float b)
{
    return {
               a.simdInternal_ + b.simdInternal_
    };
}

static inline Simd4Float gmx_simdcall
operator-(Simd4Float a, Simd4Float b)
{
    return {
               a.simdInternal_ - b.simdInternal_
    };
}

static inline Simd4Float gmx_simdcall
operator-(Simd4Float x)
{
    return {
               x.simdInternal_ ^ _mm_set1_ps(GMX_FLOAT_NEGZERO)
    };
}

static inline Simd4Float gmx_simdcall
operator*(Simd4Float a, Simd4Float b)
{
    return {
               a.simdInternal_ * b.simdInternal_
    };
}

static inline Simd4Float gmx_simdcall
fma(Simd4Float a, Simd4Float b, Simd4Float c)
{
    return {
               a.simdInternal_ * b.simdInternal_ + c.simdInternal_
    };
}

static inline Simd4Float gmx_simdcall
fms(Simd4Float a, Simd4Float b, Simd4Float c)
{
    return {
               a.simdInternal_ * b.simdInternal_ - c.simdInternal_
    };
}

static inline Simd4Float gmx_simdcall
fnma(Simd4Float a, Simd4Float b, Simd4Float c)
{
    return {
               c.simdInternal_ - a.simdInternal_ * b.simdInternal_
    };
}

static inline Simd4Float gmx_simdcall
fnms(Simd4Float a, Simd4Float b, Simd4Float c)
{
    return {
               nsimd::set1<nsimd::pack<float> >(0) - a.simdInternal_ * b.simdInternal_ + c.simdInternal_
    };
}

static inline Simd4Float gmx_simdcall
rsqrt(Simd4Float x)
{
    return {
               nsimd::rsqrt11(x.simdInternal_)
    };
}

static inline Simd4Float gmx_simdcall
abs(Simd4Float x)
{
    return {
               nsimd::andnotb( nsimd::set1<nsimd::pack<float>>(GMX_FLOAT_NEGZERO), x.simdInternal_ )
    };
}

static inline Simd4Float gmx_simdcall
max(Simd4Float a, Simd4Float b)
{
    return {
               nsimd::max(a.simdInternal_, b.simdInternal_)
    };
}

static inline Simd4Float gmx_simdcall
min(Simd4Float a, Simd4Float b)
{
    return {
               nsimd::min(a.simdInternal_, b.simdInternal_)
    };
}

static inline Simd4Float gmx_simdcall
round(Simd4Float x)
{
    return {
               nsimd::round(x.simdInternal_, 0 | 0)
    };
}

static inline Simd4Float gmx_simdcall
trunc(Simd4Float x)
{
    return {
               nsimd::round(x.simdInternal_, 0 | 3)
    };
}

static inline float gmx_simdcall
dotProduct(Simd4Float a, Simd4Float b)
{
    __m128 /*Invalid register*/ c, d;
    c = a.simdInternal_ * b.simdInternal_;
    d = c + __builtin_ia32_vpermilps((__v4sf)(__m128)(c), (int)((((2) << 6) | ((1) << 4) | ((2) << 2) | (1))));
    d = d + __builtin_ia32_vpermilps((__v4sf)(__m128)(c), (int)((((3) << 6) | ((2) << 4) | ((3) << 2) | (2))));
    return *reinterpret_cast<float *>(&d);
}

static inline void gmx_simdcall
transpose(Simd4Float * v0, Simd4Float * v1,
          Simd4Float * v2, Simd4Float * v3)
{
    __m128 /*Invalid register*/ tmp3, tmp2, tmp1, tmp0;_MM_TRANSPOSE4_PS(v0->simdInternal_, v1->simdInternal_, v2->simdInternal_, v3->simdInternal_);
}

static inline Simd4FBool gmx_simdcall
operator==(Simd4Float a, Simd4Float b)
{
    return {
               nsimd::eq(a.simdInternal_, b.simdInternal_)
    };
}

static inline Simd4FBool gmx_simdcall
operator!=(Simd4Float a, Simd4Float b)
{
    return {
               nsimd::neq(a.simdInternal_, b.simdInternal_)
    };
}

static inline Simd4FBool gmx_simdcall
operator<(Simd4Float a, Simd4Float b)
{
    return {
               nsimd::lt(a.simdInternal_, b.simdInternal_)
    };
}

static inline Simd4FBool gmx_simdcall
operator<=(Simd4Float a, Simd4Float b)
{
    return {
               nsimd::leq(a.simdInternal_, b.simdInternal_)
    };
}

static inline Simd4FBool gmx_simdcall
operator&&(Simd4FBool a, Simd4FBool b)
{
    return {
               a.simdInternal_ & b.simdInternal_
    };
}

static inline Simd4FBool gmx_simdcall
operator||(Simd4FBool a, Simd4FBool b)
{
    return {
               a.simdInternal_ | b.simdInternal_
    };
}

static inline bool gmx_simdcall
anyTrue(Simd4FBool a) { return nsimd::any(a.simdInternal_); }

static inline Simd4Float gmx_simdcall
selectByMask(Simd4Float a, Simd4FBool mask)
{
    return {
               a.simdInternal_ & mask.simdInternal_
    };
}

static inline Simd4Float gmx_simdcall
selectByNotMask(Simd4Float a, Simd4FBool mask)
{
    return {
               nsimd::andnot(mask.simdInternal_, a.simdInternal_)
    };
}

static inline Simd4Float gmx_simdcall
blend(Simd4Float a, Simd4Float b, Simd4FBool sel)
{
    return {
               nsimd::if_else(a.simdInternal_, b.simdInternal_, sel.simdInternal_)
    };
}

static inline float gmx_simdcall
reduce(Simd4Float a)
{
    __m128 /*Invalid register*/ b;
    b = a.simdInternal_ + __builtin_ia32_vpermilps((__v4sf)(__m128)(a.simdInternal_), (int)((((1) << 6) | ((0) << 4) | ((3) << 2) | (2))));
    b = _mm_add_ss(b, _mm_permute_ps(b, _MM_SHUFFLE(0, 3, 2, 1)));
    return *reinterpret_cast<float *>(&b);
}

}      // namespace gm

#endif // GMX_SIMD_IMPL_X86_AVX_256_SIMD4_FLOAT_
