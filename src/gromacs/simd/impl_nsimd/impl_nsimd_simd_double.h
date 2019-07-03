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

#include <cassert>
#include <cstddef>
#include <cstdint>

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

class SimdDBool
{
    public:
        SimdDBool() {}

        SimdDBool(bool b) : simdInternal_(nsimd::reinterpret<nsimd::pack<double> >(nsimd::set1<nsimd::pack<long> >(b ? 0x1FFFFFFF : 0l))) {}

        // Internal utility constructor to simplify return statements
        SimdDBool(nsimd::pack<double> simd) : simdInternal_(simd) {}

        nsimd::pack<double> simdInternal_;
};

#include "impl_nsimd_simd_double_defined.h"


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
               nsimd::set1<nsimd::pack<double> >(0.0)
    };
}


static inline SimdDBool gmx_simdcall
operator&&(SimdDBool a, SimdDBool b)
{
    return {
               a.simdInternal_ & b.simdInternal_
    };
}

static inline SimdDBool gmx_simdcall
operator||(SimdDBool a, SimdDBool b)
{
    return {
               a.simdInternal_ | b.simdInternal_
    };
}


static inline SimdDBool gmx_simdcall
operator==(SimdDouble a, SimdDouble b)
{
    double x = 0xFFFFFFFF;
    return {
               nsimd::if_else1(a.simdInternal_== b.simdInternal_, nsimd::set1<nsimd::pack<double>>(x), nsimd::set1<nsimd::pack<double>>(0.0))
    };
}

static inline SimdDBool gmx_simdcall
operator!=(SimdDouble a, SimdDouble b)
{
    double x = 0xFFFFFFFF;
    return {
               nsimd::if_else1(a.simdInternal_!= b.simdInternal_, nsimd::set1<nsimd::pack<double>>(x), nsimd::set1<nsimd::pack<double>>(0.0))
    };
}

static inline SimdDBool gmx_simdcall
operator<(SimdDouble a, SimdDouble b)
{
    double x = 0xFFFFFFFF;
    return {
               nsimd::if_else1(a.simdInternal_< b.simdInternal_, nsimd::set1<nsimd::pack<double>>(x), nsimd::set1<nsimd::pack<double>>(0.0))
    };
}

static inline SimdDBool gmx_simdcall
operator<=(SimdDouble a, SimdDouble b)
{
    double x = 0xFFFFFFFF;
    return {
               nsimd::if_else1(a.simdInternal_<= b.simdInternal_, nsimd::set1<nsimd::pack<double>>(x), nsimd::set1<nsimd::pack<double>>(0.0))
    };
}


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
               a.simdInternal_ + (b.simdInternal_ & m.simdInternal_)
    };
}

static inline SimdDouble gmx_simdcall
maskzMul(SimdDouble a, SimdDouble b, SimdDBool m)
{
    return {
               (a.simdInternal_ * b.simdInternal_) & m.simdInternal_
    };
}

static inline SimdDouble
maskzFma(SimdDouble a, SimdDouble b, SimdDouble c, SimdDBool m)
{
    return {
               nsimd::fma(a.simdInternal_, b.simdInternal_, c.simdInternal_) & m.simdInternal_
    };
}

static inline SimdDouble
maskzRsqrt(SimdDouble x, SimdDBool m)
{
#ifndef NDEBUG
    x.simdInternal_ = nsimd::if_else1(nsimd::cvt<nsimd::packl<double>>(m.simdInternal_), x.simdInternal_, nsimd::set1<nsimd::pack<double> >(1.));
#endif
    return {
               nsimd::rsqrt11(x.simdInternal_) & m.simdInternal_
    };
}

static inline SimdDouble
maskzRcp(SimdDouble x, SimdDBool m)
{
#ifndef NDEBUG
    x.simdInternal_ = nsimd::if_else1(nsimd::cvt<nsimd::packl<double>>(m.simdInternal_), x.simdInternal_, nsimd::set1<nsimd::pack<double> >(1.));
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
            // nsimd::if_else1(nsimd::cvt<nsimd::packl<double> >(sel.simdInternal_), a.simdInternal_, b.simdInternal_)
            nsimd::if_else1(nsimd::cvt<nsimd::packl<double> >(sel.simdInternal_), b.simdInternal_, a.simdInternal_)
    };
}

// static inline bool gmx_simdcall
// anyTrue(SimdDIBool a) { return nsimd::any(nsimd::loadla<nsimd::packl<int> >(a.simdInternal_)); }


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


}      // namespace gm

#endif // GMX_SIMD_IMPL_NSIMD_SIMD_DOUBLE_H