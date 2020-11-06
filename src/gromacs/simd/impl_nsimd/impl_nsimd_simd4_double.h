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

#ifndef GMX_SIMD_IMPL_NSIMD_SIMD4_DOUBLE_H
#define GMX_SIMD_IMPL_NSIMD_SIMD4_DOUBLE_H

#include "config.h"
#include "float.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/cxx_adv_api_functions.hpp>
#include <nsimd/nsimd.h>

#include "impl_nsimd_general.h"

namespace gmx {

#if defined(NSIMD_SSE2) || defined(NSIMD_SSE42) || defined(NSIMD_NEON128) ||  \
    defined(NSIMD_AARCH64)
#error "No SIMD register that can store 4 doubles"
#elif defined(NSIMD_AVX)
typedef nsimd::pack<double, 1, nsimd::avx> pack4d_t;
typedef nsimd::packl<double, 1, nsimd::avx> packl4d_t;
#else
typedef nsimd::pack<double, 1, nsimd::avx2> pack4d_t;
typedef nsimd::packl<double, 1, nsimd::avx2> packl4d_t;
#endif

class Simd4Double {
public:
  Simd4Double() {}

  Simd4Double(double d)
      : simdInternal_(nsimd::set1<pack4d_t>(d)) {}

  // Internal utility constructor to simplify return statement
  Simd4Double(pack4d_t v) : simdInternal_(v) {}

  pack4d_t simdInternal_;
};

class Simd4DBool {
public:
  Simd4DBool() {}

  //! \brief Construct from scalar bool
  Simd4DBool(bool b)
      : simdInternal_(
            nsimd::to_logical(nsimd::set1<pack4d_t>(b ? 1.0 : 0.0))) {}

  Simd4DBool(packl4d_t v) : simdInternal_(v) {}

  packl4d_t simdInternal_;
};

static inline Simd4Double gmx_simdcall load4(const double *m) {
  assert(std::size_t(m) % 32 == 0);
  return {nsimd::loada<pack4d_t >(m)};
}

static inline void gmx_simdcall store4(double *m, Simd4Double a) {
  assert(std::size_t(m) % 32 == 0);
  nsimd::storea(m, a.simdInternal_);
}

static inline Simd4Double gmx_simdcall load4U(const double *m) {
  return {nsimd::loadu<pack4d_t >(m)};
}

static inline void gmx_simdcall store4U(double *m, Simd4Double a) {
  nsimd::storeu(m, a.simdInternal_);
}

static inline Simd4Double gmx_simdcall simd4SetZeroD() {
  return {nsimd::set1<pack4d_t >(0.0)};
}

static inline Simd4Double gmx_simdcall operator&(Simd4Double a,
                                                 Simd4Double b) {
  return {a.simdInternal_ & b.simdInternal_};
}

static inline Simd4Double gmx_simdcall andNot(Simd4Double a, Simd4Double b) {
  return {nsimd::andnotb(b.simdInternal_, a.simdInternal_)};
}

static inline Simd4Double gmx_simdcall operator|(Simd4Double a,
                                                 Simd4Double b) {
  return {a.simdInternal_ | b.simdInternal_};
}

static inline Simd4Double gmx_simdcall operator^(Simd4Double a,
                                                 Simd4Double b) {
  return {a.simdInternal_ ^ b.simdInternal_};
}

static inline Simd4Double gmx_simdcall operator+(Simd4Double a,
                                                 Simd4Double b) {
  return {a.simdInternal_ + b.simdInternal_};
}

static inline Simd4Double gmx_simdcall operator-(Simd4Double a,
                                                 Simd4Double b) {
  return {a.simdInternal_ - b.simdInternal_};
}

static inline Simd4Double gmx_simdcall operator-(Simd4Double x) {
  return {x.simdInternal_ ^ nsimd::set1<pack4d_t>(GMX_DOUBLE_NEGZERO)};
}

static inline Simd4Double gmx_simdcall operator*(Simd4Double a,
                                                 Simd4Double b) {
  return {a.simdInternal_ * b.simdInternal_};
}

static inline Simd4Double gmx_simdcall fma(Simd4Double a, Simd4Double b,
                                           Simd4Double c) {
  return {nsimd::fma(a.simdInternal_, b.simdInternal_, c.simdInternal_)};
}

static inline Simd4Double gmx_simdcall fms(Simd4Double a, Simd4Double b,
                                           Simd4Double c) {
  return {nsimd::fms(a.simdInternal_, b.simdInternal_, c.simdInternal_)};
}

static inline Simd4Double gmx_simdcall fnma(Simd4Double a, Simd4Double b,
                                            Simd4Double c) {
  return {nsimd::fnma(a.simdInternal_, b.simdInternal_, c.simdInternal_)};
}

static inline Simd4Double gmx_simdcall fnms(Simd4Double a, Simd4Double b,
                                            Simd4Double c) {
  return {nsimd::fnms(a.simdInternal_, b.simdInternal_, c.simdInternal_)};
}

static inline Simd4Double gmx_simdcall rsqrt(Simd4Double x) {
  return {nsimd::rsqrt11(x.simdInternal_)};
}

static inline Simd4Double gmx_simdcall abs(Simd4Double x) {
  return {nsimd::abs(x.simdInternal_)};
}

static inline Simd4Double gmx_simdcall max(Simd4Double a, Simd4Double b) {
  return {nsimd::max(a.simdInternal_, b.simdInternal_)};
}

static inline Simd4Double gmx_simdcall min(Simd4Double a, Simd4Double b) {
  return {nsimd::min(a.simdInternal_, b.simdInternal_)};
}

static inline Simd4Double gmx_simdcall round(Simd4Double x) {
  return {nsimd::round_to_even(x.simdInternal_)};
}

static inline Simd4Double gmx_simdcall trunc(Simd4Double x) {
  return {nsimd::trunc(x.simdInternal_)};
}

static inline Simd4DBool gmx_simdcall operator==(Simd4Double a,
                                                 Simd4Double b) {
  return Simd4DBool(a.simdInternal_ == b.simdInternal_);
}

static inline Simd4DBool gmx_simdcall operator!=(Simd4Double a,
                                                 Simd4Double b) {
  return Simd4DBool(a.simdInternal_ != b.simdInternal_);
}

static inline Simd4DBool gmx_simdcall operator<(Simd4Double a, Simd4Double b) {
  return Simd4DBool(a.simdInternal_ < b.simdInternal_);
}

static inline Simd4DBool gmx_simdcall operator<=(Simd4Double a,
                                                 Simd4Double b) {
  return Simd4DBool(a.simdInternal_ <= b.simdInternal_);
}

static inline Simd4DBool gmx_simdcall operator&&(Simd4DBool a, Simd4DBool b) {
  return {a.simdInternal_ && b.simdInternal_};
}

static inline Simd4DBool gmx_simdcall operator||(Simd4DBool a, Simd4DBool b) {
  return {a.simdInternal_ || b.simdInternal_};
}

// Include the rest of the functions
#include "impl_nsimd_simd4_double_defined.h"

} // namespace gmx

#endif // GMX_SIMD_IMPL_NSIMD_SIMD4_DOUBLE_H