/
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

#ifndef GMX_SIMD_IMPL_X86_AVX2_256_SIMD_DOUBLE_H
#define GMX_SIMD_IMPL_X86_AVX2_256_SIMD_DOUBLE_H

#include "config.h"

#include <nsimd/nsimd.h>
#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/cxx_adv_api_functions.hpp>

#include "gromacs/math/utilities.h"
#include "gromacs/simd/impl_x86_avx_256/impl_x86_avx_256_simd_double.h"

namespace gmx
{

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
frexp(SimdDouble value, SimdDInt32 * exponent)
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
    exponent->simdInternal_  = _mm_unpacklo_epi64(_mm256_castsi256_si128(iExponent), iExponent128);

    return {
               value.simdInternal_ & mantissaMask | half
    };
}

template <MathOptimization opt = MathOptimization::Safe>
static inline SimdDouble
ldexp(SimdDouble value, SimdDInt32 exponent)
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

#endif // GMX_SIMD_IMPL_X86_AVX2_256_SIMD_DOUBLE_
