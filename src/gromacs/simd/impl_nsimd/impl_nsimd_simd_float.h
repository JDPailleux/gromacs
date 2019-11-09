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

#ifndef GMX_SIMD_IMPL_NSIMD_SIMD_FLOAT_H
#define GMX_SIMD_IMPL_NSIMD_SIMD_FLOAT_H

#include "config.h"
#include "float.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/cxx_adv_api_functions.hpp>
#include <nsimd/nsimd.h>

#include "gromacs/math/utilities.h"
#include "impl_nsimd_general.h"

namespace gmx {

class SimdFloat {
public:
  SimdFloat() {}

  SimdFloat(float f) : simdInternal_(nsimd::set1<nsimd::pack<float> >(f)) {}

  // Internal utility constructor to simplify return statements
  SimdFloat(nsimd::pack<float> simd) : simdInternal_(simd) {}

  nsimd::pack<float> simdInternal_;
};

class SimdFInt32 {
public:
  SimdFInt32() {}

  SimdFInt32(std::int32_t i)
      : simdInternal_(nsimd::set1<nsimd::pack<int> >(i)) {}

  // Internal utility constructor to simplify return statements
  SimdFInt32(nsimd::pack<int> simd) : simdInternal_(simd) {}

  nsimd::pack<int> simdInternal_;
};

class SimdFBool {
public:
  SimdFBool() {}

  SimdFBool(bool b)
      : simdInternal_(nsimd::to_logical(
            nsimd::set1<nsimd::pack<float> >(b ? 1.0f : 0.0f))) {}

  // Internal utility constructor to simplify return statements
  SimdFBool(nsimd::packl<float> v) : simdInternal_(v) {}

  nsimd::packl<float> simdInternal_;
};

class SimdFIBool {
public:
  SimdFIBool() {}

  SimdFIBool(bool b)
      : simdInternal_(
            nsimd::to_logical(nsimd::set1<nsimd::pack<int> >(b ? 1 : 0))) {}

  // Internal utility constructor to simplify return statements
  SimdFIBool(nsimd::packl<int> v) : simdInternal_(v) {}

  nsimd::packl<int> simdInternal_;
};

#include "impl_nsimd_simd_float_defined.h"

static inline SimdFloat gmx_simdcall simdLoad(const float *m,
                                              SimdFloatTag /*unused*/ = {}) {
  assert(std::size_t(m) % 32 == 0);
  return {nsimd::loada<nsimd::pack<float> >(m)};
}

static inline void gmx_simdcall store(float *m, SimdFloat a) {
  assert(std::size_t(m) % 32 == 0);
  nsimd::storea(m, a.simdInternal_);
}

static inline SimdFloat gmx_simdcall simdLoadU(const float *m,
                                               SimdFloatTag /*unused*/ = {}) {
  return {nsimd::loadu<nsimd::pack<float> >(m)};
}

static inline void gmx_simdcall storeU(float *m, SimdFloat a) {
  nsimd::storeu(m, a.simdInternal_);
}

static inline SimdFloat gmx_simdcall setZeroF() {
  return {nsimd::set1<nsimd::pack<float> >(0.0f)};
}

static inline SimdFInt32 gmx_simdcall simdLoad(const std::int32_t *m,
                                               SimdFInt32Tag /*unused*/) {
  assert(std::size_t(m) % 32 == 0);
  return {nsimd::loada<nsimd::pack<int> >(m)};
}

static inline void gmx_simdcall store(std::int32_t *m, SimdFInt32 a) {
  assert(std::size_t(m) % 32 == 0);
  nsimd::storea(m, a.simdInternal_);
}

static inline SimdFInt32 gmx_simdcall simdLoadU(const std::int32_t *m,
                                                SimdFInt32Tag /*unused*/) {
  return {nsimd::loadu<nsimd::pack<int> >(m)};
}

static inline void gmx_simdcall storeU(std::int32_t *m, SimdFInt32 a) {
  nsimd::storeu(m, a.simdInternal_);
}

static inline SimdFInt32 gmx_simdcall setZeroFI() {
  return {nsimd::set1<nsimd::pack<int> >(0)};
}

static inline SimdFloat gmx_simdcall operator&(SimdFloat a, SimdFloat b) {
  return {a.simdInternal_ & b.simdInternal_};
}

static inline SimdFloat gmx_simdcall andNot(SimdFloat a, SimdFloat b) {
  return {nsimd::andnotb(b.simdInternal_, a.simdInternal_)};
}

static inline SimdFloat gmx_simdcall operator|(SimdFloat a, SimdFloat b) {
  return {a.simdInternal_ | b.simdInternal_};
}

static inline SimdFloat gmx_simdcall operator^(SimdFloat a, SimdFloat b) {
  return {a.simdInternal_ ^ b.simdInternal_};
}

static inline SimdFloat gmx_simdcall operator+(SimdFloat a, SimdFloat b) {
  return {a.simdInternal_ + b.simdInternal_};
}

static inline SimdFloat gmx_simdcall operator-(SimdFloat a, SimdFloat b) {
  return {a.simdInternal_ - b.simdInternal_};
}

static inline SimdFloat gmx_simdcall operator-(SimdFloat x) {
  return {x.simdInternal_ ^ nsimd::set1<nsimd::pack<float> >(-0.F)};
}

static inline SimdFloat gmx_simdcall operator*(SimdFloat a, SimdFloat b) {
  return {a.simdInternal_ * b.simdInternal_};
}

static inline SimdFloat gmx_simdcall rsqrt(SimdFloat x) {
  return {nsimd::rsqrt8(x.simdInternal_)};
}

static inline SimdFloat gmx_simdcall rcp(SimdFloat x) {
  return {nsimd::rec8(x.simdInternal_)};
}

static inline SimdFloat gmx_simdcall maskAdd(SimdFloat a, SimdFloat b,
                                             SimdFBool m) {
  return {a.simdInternal_ +
          (b.simdInternal_ & nsimd::to_mask(m.simdInternal_))};
}

static inline SimdFloat gmx_simdcall maskzMul(SimdFloat a, SimdFloat b,
                                              SimdFBool m) {
  return {(a.simdInternal_ * b.simdInternal_) &
          nsimd::to_mask(m.simdInternal_)};
}

static inline SimdFloat gmx_simdcall maskzFma(SimdFloat a, SimdFloat b,
                                              SimdFloat c, SimdFBool m) {
  return {nsimd::fma(a.simdInternal_, b.simdInternal_, c.simdInternal_) &
          nsimd::to_mask(m.simdInternal_)};
}

static inline SimdFloat maskzRsqrt(SimdFloat x, SimdFBool m) {
#ifndef NDEBUG
  x.simdInternal_ =
      nsimd::if_else(nsimd::cvt<nsimd::packl<float> >(m.simdInternal_),
                     x.simdInternal_, nsimd::set1<nsimd::pack<float> >(1.));
#endif
  return {nsimd::rsqrt8(x.simdInternal_) & nsimd::to_mask(m.simdInternal_)};
}

static inline SimdFloat maskzRcp(SimdFloat x, SimdFBool m) {
#ifndef NDEBUG
  x.simdInternal_ =
      nsimd::if_else(nsimd::cvt<nsimd::packl<float> >(m.simdInternal_),
                     x.simdInternal_, nsimd::set1<nsimd::pack<double> >(1.));
#endif
  return {nsimd::rec8(x.simdInternal_) & nsimd::to_mask(m.simdInternal_)};
}

static inline SimdFloat gmx_simdcall abs(SimdFloat x) {
  return {nsimd::abs(x.simdInternal_)};
}

static inline SimdFloat gmx_simdcall max(SimdFloat a, SimdFloat b) {
  return {nsimd::max(a.simdInternal_, b.simdInternal_)};
}

static inline SimdFloat gmx_simdcall min(SimdFloat a, SimdFloat b) {
  return {nsimd::min(a.simdInternal_, b.simdInternal_)};
}

static inline SimdFloat gmx_simdcall round(SimdFloat x) {
// GROMACS use a invalid trick of big values so we do the same
#ifdef NSIMD_SSE2
  return {nsimd::pack<float>(_mm_cvtepi32_ps(_mm_cvtps_epi32(
                                x.simdInternal_.native_register())))};
#else
  return {nsimd::round_to_even(x.simdInternal_)};
#endif
}

static inline SimdFloat gmx_simdcall trunc(SimdFloat x) {
#ifdef NSIMD_SSE2
  return {nsimd::pack<float>(_mm_cvtepi32_ps(_mm_cvttps_epi32(
                                x.simdInternal_.native_register())))};
#else
  return {nsimd::trunc(x.simdInternal_)};
#endif
}

//--------------------------------------------------------------------------------------------------
static inline SimdFloat gmx_simdcall fma(SimdFloat a, SimdFloat b,
                                         SimdFloat c) {
  return {nsimd::fma(a.simdInternal_, b.simdInternal_, c.simdInternal_)};
}

static inline SimdFloat gmx_simdcall fms(SimdFloat a, SimdFloat b,
                                         SimdFloat c) {
  return {nsimd::fms(a.simdInternal_, b.simdInternal_, c.simdInternal_)};
}

static inline SimdFloat gmx_simdcall fnma(SimdFloat a, SimdFloat b,
                                          SimdFloat c) {
  return {nsimd::fnma(a.simdInternal_, b.simdInternal_, c.simdInternal_)};
}

static inline SimdFloat gmx_simdcall fnms(SimdFloat a, SimdFloat b,
                                          SimdFloat c) {
  return {nsimd::fnms(a.simdInternal_, b.simdInternal_, c.simdInternal_)};
}

static inline SimdFBool gmx_simdcall testBits(SimdFloat a) {
  return {nsimd::to_logical(a.simdInternal_)};
}

static inline SimdFBool gmx_simdcall operator==(SimdFloat a, SimdFloat b) {
  return {a.simdInternal_ == b.simdInternal_};
}

static inline SimdFBool gmx_simdcall operator!=(SimdFloat a, SimdFloat b) {
  return {a.simdInternal_ != b.simdInternal_};
}

static inline SimdFBool gmx_simdcall operator<(SimdFloat a, SimdFloat b) {
  return {a.simdInternal_ < b.simdInternal_};
}

static inline SimdFBool gmx_simdcall operator<=(SimdFloat a, SimdFloat b) {
  return {a.simdInternal_ <= b.simdInternal_};
}

static inline SimdFInt32 gmx_simdcall operator&(SimdFInt32 a, SimdFInt32 b) {
  return {a.simdInternal_ & b.simdInternal_};
}

static inline SimdFInt32 gmx_simdcall andNot(SimdFInt32 a, SimdFInt32 b) {
  return {nsimd::andnotb(b.simdInternal_, a.simdInternal_)};
}

static inline SimdFInt32 gmx_simdcall operator|(SimdFInt32 a, SimdFInt32 b) {
  return {a.simdInternal_ | b.simdInternal_};
}

static inline SimdFInt32 gmx_simdcall operator^(SimdFInt32 a, SimdFInt32 b) {
  return {a.simdInternal_ ^ b.simdInternal_};
}

static inline SimdFInt32 gmx_simdcall operator+(SimdFInt32 a, SimdFInt32 b) {
  return {a.simdInternal_ + b.simdInternal_};
}

static inline SimdFInt32 gmx_simdcall operator-(SimdFInt32 a, SimdFInt32 b) {
  return {a.simdInternal_ - b.simdInternal_};
}

static inline SimdFInt32 gmx_simdcall operator*(SimdFInt32 a, SimdFInt32 b) {
  return {a.simdInternal_ * b.simdInternal_};
}

static inline SimdFIBool gmx_simdcall operator==(SimdFInt32 a, SimdFInt32 b) {
  return {a.simdInternal_ == b.simdInternal_};
}

static inline SimdFIBool gmx_simdcall testBits(SimdFInt32 a) {
  return {nsimd::to_logical(a.simdInternal_)};
}

static inline SimdFIBool gmx_simdcall operator<(SimdFInt32 a, SimdFInt32 b) {
  return {a.simdInternal_ < b.simdInternal_};
}

static inline SimdFBool gmx_simdcall operator&&(SimdFBool a, SimdFBool b) {
  return {a.simdInternal_ && b.simdInternal_};
}

static inline SimdFBool gmx_simdcall operator||(SimdFBool a, SimdFBool b) {
  return {a.simdInternal_ || b.simdInternal_};
}

static inline bool gmx_simdcall anyTrue(SimdFBool a) {
  return (bool)nsimd::any(a.simdInternal_);
}

static inline SimdFloat gmx_simdcall selectByMask(SimdFloat a,
                                                  SimdFBool mask) {
  return {a.simdInternal_ & nsimd::to_mask(mask.simdInternal_)};
}

static inline SimdFloat gmx_simdcall selectByNotMask(SimdFloat a,
                                                     SimdFBool mask) {
  return {a.simdInternal_ & nsimd::to_mask(!mask.simdInternal_)};
}

static inline SimdFloat gmx_simdcall blend(SimdFloat a, SimdFloat b,
                                           SimdFBool sel) {
  return SimdFloat{
      nsimd::if_else(sel.simdInternal_, b.simdInternal_, a.simdInternal_)};
}

static inline SimdFInt32 gmx_simdcall cvtR2I(SimdFloat a) {
  return {
      nsimd::cvt<nsimd::pack<int> >(nsimd::round_to_even(a.simdInternal_))};
}

static inline SimdFloat gmx_simdcall cvtI2R(SimdFInt32 a) {
  return {nsimd::cvt<nsimd::pack<float> >(a.simdInternal_)};
}

static inline SimdFIBool gmx_simdcall operator&&(SimdFIBool a, SimdFIBool b) {
  return {a.simdInternal_ && b.simdInternal_};
}

static inline SimdFIBool gmx_simdcall operator||(SimdFIBool a, SimdFIBool b) {
  return {a.simdInternal_ || b.simdInternal_};
}

static inline bool gmx_simdcall anyTrue(SimdFIBool a) {
  return nsimd::any(a.simdInternal_);
}

static inline SimdFInt32 gmx_simdcall selectByMask(SimdFInt32 a,
                                                   SimdFIBool mask) {
  return {a.simdInternal_ & nsimd::to_mask(mask.simdInternal_)};
}

static inline SimdFInt32 gmx_simdcall selectByNotMask(SimdFInt32 a,
                                                      SimdFIBool mask) {
  return {a.simdInternal_ & nsimd::to_mask(!mask.simdInternal_)};
}

static inline SimdFInt32 gmx_simdcall blend(SimdFInt32 a, SimdFInt32 b,
                                            SimdFIBool sel) {
  return {nsimd::if_else(sel.simdInternal_, b.simdInternal_, a.simdInternal_)};
}

static inline SimdFIBool gmx_simdcall cvtB2IB(SimdFBool a) {
  return {nsimd::reinterpretl<nsimd::packl<int> >(a.simdInternal_)};
}

static inline SimdFBool gmx_simdcall cvtIB2B(SimdFIBool a) {
  return {nsimd::reinterpretl<nsimd::packl<float> >(a.simdInternal_)};
}

} // namespace gmx

#endif // GMX_SIMD_IMPL_NSIMD_SIMD_FLOAT_
