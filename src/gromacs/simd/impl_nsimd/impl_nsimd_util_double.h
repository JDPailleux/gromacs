/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2014,2015,2018, by the GROMACS development team, led by
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

#ifndef GMX_SIMD_IMPL_NSIMD_UTIL_DOUBLE_H
#define GMX_SIMD_IMPL_NSIMD_UTIL_DOUBLE_H

#include "config.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/cxx_adv_api_functions.hpp>
#include <nsimd/nsimd.h>

#include "gromacs/utility/basedefinitions.h"

#include "impl_nsimd_simd_double.h"
#include "impl_nsimd_general.h"

namespace gmx
{

#if (defined(NSIMD_SSE2) || defined(NSIMD_SSE42))
template <int align>
static inline void gmx_simdcall
gatherLoadTranspose(const double *        base,
                    const std::int32_t    offset[],
                    SimdDouble *          v0,
                    SimdDouble *          v1,
                    SimdDouble *          v2,
                    SimdDouble *          v3)
{
    __m128d t1, t2, t3, t4;

    assert(std::size_t(base + align * offset[0]) % 16 == 0);
    assert(std::size_t(base + align * offset[1]) % 16 == 0);

    t1                = _mm_load_pd(base + align * offset[0]);
    t2                = _mm_load_pd(base + align * offset[1]);
    t3                = _mm_load_pd(base + align * offset[0] + 2);
    t4                = _mm_load_pd(base + align * offset[1] + 2);
    v0->simdInternal_ = _mm_unpacklo_pd(t1, t2);
    v1->simdInternal_ = _mm_unpackhi_pd(t1, t2);
    v2->simdInternal_ = _mm_unpacklo_pd(t3, t4);
    v3->simdInternal_ = _mm_unpackhi_pd(t3, t4);
}

template <int align>
static inline void gmx_simdcall
gatherLoadTranspose(const double *        base,
                    const std::int32_t    offset[],
                    SimdDouble *          v0,
                    SimdDouble *          v1)
{
    __m128d t1, t2;

    assert(std::size_t(base + align * offset[0]) % 16 == 0);
    assert(std::size_t(base + align * offset[1]) % 16 == 0);

    t1                = _mm_load_pd(base + align * offset[0]);
    t2                = _mm_load_pd(base + align * offset[1]);
    v0->simdInternal_ = _mm_unpacklo_pd(t1, t2);
    v1->simdInternal_ = _mm_unpackhi_pd(t1, t2);
}

static const int c_simdBestPairAlignmentDouble = 2;

template <int align>
static inline void gmx_simdcall
gatherLoadUTranspose(const double *        base,
                     const std::int32_t    offset[],
                     SimdDouble *          v0,
                     SimdDouble *          v1,
                     SimdDouble *          v2)
{
    __m128d t1, t2, t3, t4;
    t1                = _mm_loadu_pd(base + align * offset[0]);
    t2                = _mm_loadu_pd(base + align * offset[1]);
    t3                = _mm_load_sd(base + align * offset[0] + 2);
    t4                = _mm_load_sd(base + align * offset[1] + 2);
    v0->simdInternal_ = _mm_unpacklo_pd(t1, t2);
    v1->simdInternal_ = _mm_unpackhi_pd(t1, t2);
    v2->simdInternal_ = _mm_unpacklo_pd(t3, t4);
}

template <int align>
static inline void gmx_simdcall
transposeScatterStoreU(double *            base,
                       const std::int32_t  offset[],
                       SimdDouble          v0,
                       SimdDouble          v1,
                       SimdDouble          v2)
{
    __m128d t1, t2;
    t1  = _mm_unpacklo_pd(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
    t2  = _mm_unpackhi_pd(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
    _mm_storeu_pd(base + align * offset[0], t1);
    _mm_store_sd(base + align * offset[0] + 2, v2.simdInternal_.native_register());
    _mm_storeu_pd(base + align * offset[1], t2);
    _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[1] + 2), _mm_castpd_ps(v2.simdInternal_.native_register()));
}

template <int align>
static inline void gmx_simdcall
transposeScatterIncrU(double *            base,
                      const std::int32_t  offset[],
                      SimdDouble          v0,
                      SimdDouble          v1,
                      SimdDouble          v2)
{
    __m128d t1, t2, t3, t4, t5, t6, t7;

    t5          = _mm_unpacklo_pd(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
    t6          = _mm_unpackhi_pd(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
    t7          = _mm_unpackhi_pd(v2.simdInternal_.native_register(), v2.simdInternal_.native_register());

    t1          = _mm_loadu_pd(base + align * offset[0]);
    t2          = _mm_load_sd(base + align * offset[0] + 2);
    t1          = _mm_add_pd(t1, t5);
    t2          = _mm_add_sd(t2, v2.simdInternal_.native_register());
    _mm_storeu_pd(base + align * offset[0], t1);
    _mm_store_sd(base + align * offset[0] + 2, t2);

    t3          = _mm_loadu_pd(base + align * offset[1]);
    t4          = _mm_load_sd(base + align * offset[1] + 2);
    t3          = _mm_add_pd(t3, t6);
    t4          = _mm_add_sd(t4, t7);
    _mm_storeu_pd(base + align * offset[1], t3);
    _mm_store_sd(base + align * offset[1] + 2, t4);
}

template <int align>
static inline void gmx_simdcall
transposeScatterDecrU(double *            base,
                      const std::int32_t  offset[],
                      SimdDouble          v0,
                      SimdDouble          v1,
                      SimdDouble          v2)
{
    // This implementation is identical to the increment version, apart from using subtraction instead
    __m128d t1, t2, t3, t4, t5, t6, t7;

    t5          = _mm_unpacklo_pd(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
    t6          = _mm_unpackhi_pd(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
    t7          = _mm_unpackhi_pd(v2.simdInternal_.native_register(), v2.simdInternal_.native_register());

    t1          = _mm_loadu_pd(base + align * offset[0]);
    t2          = _mm_load_sd(base + align * offset[0] + 2);
    t1          = _mm_sub_pd(t1, t5);
    t2          = _mm_sub_sd(t2, v2.simdInternal_.native_register());
    _mm_storeu_pd(base + align * offset[0], t1);
    _mm_store_sd(base + align * offset[0] + 2, t2);

    t3          = _mm_loadu_pd(base + align * offset[1]);
    t4          = _mm_load_sd(base + align * offset[1] + 2);
    t3          = _mm_sub_pd(t3, t6);
    t4          = _mm_sub_sd(t4, t7);
    _mm_storeu_pd(base + align * offset[1], t3);
    _mm_store_sd(base + align * offset[1] + 2, t4);
}


static inline void gmx_simdcall
expandScalarsToTriplets(SimdDouble    scalar,
                        SimdDouble *  triplets0,
                        SimdDouble *  triplets1,
                        SimdDouble *  triplets2)
{
    triplets0->simdInternal_ = _mm_shuffle_pd(scalar.simdInternal_.native_register(), scalar.simdInternal_.native_register(), _MM_SHUFFLE2(0, 0));
    triplets1->simdInternal_ = _mm_shuffle_pd(scalar.simdInternal_.native_register(), scalar.simdInternal_.native_register(), _MM_SHUFFLE2(1, 0));
    triplets2->simdInternal_ = _mm_shuffle_pd(scalar.simdInternal_.native_register(), scalar.simdInternal_.native_register(), _MM_SHUFFLE2(1, 1));
}

template <int align>
static inline void gmx_simdcall
gatherLoadBySimdIntTranspose(const double *  base,
                             SimdDInt32      offset,
                             SimdDouble *    v0,
                             SimdDouble *    v1,
                             SimdDouble *    v2,
                             SimdDouble *    v3)
{
    __m128d t1, t2, t3, t4;
    // Use optimized bit-shift multiply for the most common alignments
    if (align == 4)
    {
        offset.simdInternal_ = _mm_slli_epi32(offset.simdInternal_.native_register(), 2);
    }
    else if (align == 8)
    {
        offset.simdInternal_ = _mm_slli_epi32(offset.simdInternal_.native_register(), 3);
    }
    else if (align == 12)
    {
        /* multiply by 3, then by 4 */
        offset.simdInternal_ = _mm_add_epi32(offset.simdInternal_.native_register(), _mm_slli_epi32(offset.simdInternal_.native_register(), 1));
        offset.simdInternal_ = _mm_slli_epi32(offset.simdInternal_.native_register(), 2);
    }
    else if (align == 16)
    {
        offset.simdInternal_ = _mm_slli_epi32(offset.simdInternal_.native_register(), 4);
    }

    if (align == 4 || align == 8 || align == 12 || align == 16)
    {
        assert(std::size_t(base + extract<0>(offset)) % 16 == 0);
        assert(std::size_t(base + extract<1>(offset)) % 16 == 0);

        t1  = _mm_load_pd(base + extract<0>(offset));
        t2  = _mm_load_pd(base + extract<1>(offset));
        t3  = _mm_load_pd(base + extract<0>(offset) + 2);
        t4  = _mm_load_pd(base + extract<1>(offset) + 2);
    }
    else
    {
        assert(std::size_t(base + align * extract<0>(offset)) % 16 == 0);
        assert(std::size_t(base + align * extract<1>(offset)) % 16 == 0);

        t1  = _mm_load_pd(base + align * extract<0>(offset));
        t2  = _mm_load_pd(base + align * extract<1>(offset));
        t3  = _mm_load_pd(base + align * extract<0>(offset) + 2);
        t4  = _mm_load_pd(base + align * extract<1>(offset) + 2);
    }
    v0->simdInternal_ = _mm_unpacklo_pd(t1, t2);
    v1->simdInternal_ = _mm_unpackhi_pd(t1, t2);
    v2->simdInternal_ = _mm_unpacklo_pd(t3, t4);
    v3->simdInternal_ = _mm_unpackhi_pd(t3, t4);
}

template <int align>
static inline void gmx_simdcall
gatherLoadBySimdIntTranspose(const double *    base,
                             SimdDInt32        offset,
                             SimdDouble *      v0,
                             SimdDouble *      v1)
{
    __m128d t1, t2;

    // Use optimized bit-shift multiply for the most common alignments
    if (align == 2)
    {
        offset.simdInternal_ = _mm_slli_epi32(offset.simdInternal_.native_register(), 1);
    }
    else if (align == 4)
    {
        offset.simdInternal_ = _mm_slli_epi32(offset.simdInternal_.native_register(), 2);
    }
    else if (align == 6)
    {
        // multiply by 3, then by 2
        offset.simdInternal_ = _mm_add_epi32(offset.simdInternal_.native_register(), _mm_slli_epi32(offset.simdInternal_.native_register(), 1));
        offset.simdInternal_ = _mm_slli_epi32(offset.simdInternal_.native_register(), 1);
    }
    else if (align == 8)
    {
        offset.simdInternal_ = _mm_slli_epi32(offset.simdInternal_.native_register(), 3);
    }
    else if (align == 12)
    {
        // multiply by 3, then by 4
        offset.simdInternal_ = _mm_add_epi32(offset.simdInternal_.native_register(), _mm_slli_epi32(offset.simdInternal_.native_register(), 1));
        offset.simdInternal_ = _mm_slli_epi32(offset.simdInternal_.native_register(), 2);
    }
    else if (align == 16)
    {
        offset.simdInternal_ = _mm_slli_epi32(offset.simdInternal_.native_register(), 4);
    }

    if (align == 2 || align == 4 || align == 6 ||
        align == 8 || align == 12 || align == 16)
    {
        assert(std::size_t(base + extract<0>(offset)) % 16 == 0);
        assert(std::size_t(base + extract<1>(offset)) % 16 == 0);

        t1  = _mm_load_pd(base + extract<0>(offset));
        t2  = _mm_load_pd(base + extract<1>(offset));
    }
    else
    {
        assert(std::size_t(base + align * extract<0>(offset)) % 16 == 0);
        assert(std::size_t(base + align * extract<1>(offset)) % 16 == 0);

        t1  = _mm_load_pd(base + align * extract<0>(offset));
        t2  = _mm_load_pd(base + align * extract<1>(offset));
    }
    v0->simdInternal_ = _mm_unpacklo_pd(t1, t2);
    v1->simdInternal_ = _mm_unpackhi_pd(t1, t2);
}


template <int align>
static inline void gmx_simdcall
gatherLoadUBySimdIntTranspose(const double *  base,
                              SimdDInt32      offset,
                              SimdDouble *    v0,
                              SimdDouble *    v1)
{
    __m128d t1, t2;
    // Use optimized bit-shift multiply for the most common alignments.

    // Do nothing for align == 1
    if (align == 2)
    {
        offset.simdInternal_ = _mm_slli_epi32(offset.simdInternal_.native_register(), 1);
    }
    else if (align == 4)
    {
        offset.simdInternal_ = _mm_slli_epi32(offset.simdInternal_.native_register(), 2);
    }

    if (align == 1 || align == 2 || align == 4)
    {
        t1  = _mm_loadu_pd(base + extract<0>(offset));
        t2  = _mm_loadu_pd(base + extract<1>(offset));
    }
    else
    {
        t1  = _mm_loadu_pd(base + align * extract<0>(offset));
        t2  = _mm_loadu_pd(base + align * extract<1>(offset));
    }
    v0->simdInternal_ = _mm_unpacklo_pd(t1, t2);
    v1->simdInternal_ = _mm_unpackhi_pd(t1, t2);
}

static inline double gmx_simdcall
reduceIncr4ReturnSum(double *    m,
                     SimdDouble  v0,
                     SimdDouble  v1,
                     SimdDouble  v2,
                     SimdDouble  v3)
{
    __m128d t1, t2, t3, t4;

    t1 = _mm_unpacklo_pd(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
    t2 = _mm_unpackhi_pd(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
    t3 = _mm_unpacklo_pd(v2.simdInternal_.native_register(), v3.simdInternal_.native_register());
    t4 = _mm_unpackhi_pd(v2.simdInternal_.native_register(), v3.simdInternal_.native_register());

    t1 = _mm_add_pd(t1, t2);
    t3 = _mm_add_pd(t3, t4);

    t2 = _mm_add_pd(t1, _mm_load_pd(m));
    t4 = _mm_add_pd(t3, _mm_load_pd(m + 2));

    assert(std::size_t(m) % 16 == 0);

    _mm_store_pd(m, t2);
    _mm_store_pd(m + 2, t4);

    t1 = _mm_add_pd(t1, t3);

    t2 = _mm_add_sd(t1, _mm_shuffle_pd(t1, t1, _MM_SHUFFLE2(1, 1)));
    return *reinterpret_cast<double *>(&t2);
}


#elif (defined(NSIMD_AVX) || defined(NSIMD_AVX2))

static inline void gmx_simdcall
avx256Transpose4By4(nsimd::pack<double> * v0,
                    nsimd::pack<double> * v1,
                    nsimd::pack<double> * v2,
                    nsimd::pack<double> * v3)
{
  __m256d tmp0 = v0->native_register(), tmp1 = v1->native_register(),
          tmp2 = v2->native_register(), tmp3 = v3->native_register();
  __m256d t1 = _mm256_unpacklo_pd(tmp0, tmp1);
  __m256d t2 = _mm256_unpackhi_pd(tmp0, tmp1);
  __m256d t3 = _mm256_unpacklo_pd(tmp2, tmp3);
  __m256d t4 = _mm256_unpackhi_pd(tmp2, tmp3);
  tmp0 = _mm256_permute2f128_pd(t1, t3, 0x20);
  tmp1 = _mm256_permute2f128_pd(t2, t4, 0x20);
  tmp2 = _mm256_permute2f128_pd(t1, t3, 0x31);
  tmp3 = _mm256_permute2f128_pd(t2, t4, 0x31);

  *v0 = tmp0;
  *v1 = tmp1;
  *v2 = tmp2;
  *v3 = tmp3;
}

static inline void gmx_simdcall
avx256Transpose4By4(nsimd::pack<double> * v0,
                    nsimd::pack<double> * v1,
                    nsimd::pack<double> * v2,
                    __m256d * v3)
{
    __m256d tmp0, tmp1, tmp2;
    __m256d t1 = _mm256_unpacklo_pd(tmp0, tmp1);
    __m256d t2 = _mm256_unpackhi_pd(tmp0, tmp1);
    __m256d t3 = _mm256_unpacklo_pd(tmp2, *v3);
    __m256d t4 = _mm256_unpackhi_pd(tmp2, *v3);
    tmp0        = _mm256_permute2f128_pd(t1, t3, 0x20);
    tmp1        = _mm256_permute2f128_pd(t2, t4, 0x20);
    tmp2        = _mm256_permute2f128_pd(t1, t3, 0x31);
    *v3        = _mm256_permute2f128_pd(t2, t4, 0x31);

    *v0 = tmp0;
    *v1 = tmp1;
    *v2 = tmp2;
}

template <int align>
static inline void gmx_simdcall
gatherLoadTranspose(const double *        base,
                    const std::int32_t    offset[],
                    SimdDouble *          v0,
                    SimdDouble *          v1,
                    SimdDouble *          v2,
                    SimdDouble *          v3)
{
    assert(std::size_t(offset) % 16 == 0);
    assert(std::size_t(base) % 32 == 0);
    assert(align % 4 == 0);

    v0->simdInternal_ = _mm256_load_pd( base + align * offset[0] );
    v1->simdInternal_ = _mm256_load_pd( base + align * offset[1] );
    v2->simdInternal_ = _mm256_load_pd( base + align * offset[2] );
    v3->simdInternal_ = _mm256_load_pd( base + align * offset[3] );
    avx256Transpose4By4(&(v0->simdInternal_), &(v1->simdInternal_), &(v2->simdInternal_), &(v3->simdInternal_));
}

template <int align>
static inline void gmx_simdcall
gatherLoadTranspose(const double *        base,
                    const std::int32_t    offset[],
                    SimdDouble *          v0,
                    SimdDouble *          v1)
{
    __m128d t1, t2, t3, t4;
    __m256d tA, tB;

    assert(std::size_t(offset) % 16 == 0);
    assert(std::size_t(base) % 16 == 0);
    assert(align % 2 == 0);

    t1   = _mm_load_pd( base + align * offset[0] );
    t2   = _mm_load_pd( base + align * offset[1] );
    t3   = _mm_load_pd( base + align * offset[2] );
    t4   = _mm_load_pd( base + align * offset[3] );
    tA   = _mm256_insertf128_pd(_mm256_castpd128_pd256(t1), t3, 0x1);
    tB   = _mm256_insertf128_pd(_mm256_castpd128_pd256(t2), t4, 0x1);

    v0->simdInternal_ = _mm256_unpacklo_pd(tA, tB);
    v1->simdInternal_ = _mm256_unpackhi_pd(tA, tB);
}

static const int c_simdBestPairAlignmentDouble = 2;

// With the implementation below, thread-sanitizer can detect false positives.
// For loading a triplet, we load 4 floats and ignore the last. Another thread
// might write to this element, but that will not affect the result.
// On AVX2 we can use a gather intrinsic instead.
template <int align>
static inline void gmx_simdcall
gatherLoadUTranspose(const double *        base,
                     const std::int32_t    offset[],
                     SimdDouble *          v0,
                     SimdDouble *          v1,
                     SimdDouble *          v2)
{
    assert(std::size_t(offset) % 16 == 0);

    __m256d t1, t2, t3, t4, t5, t6, t7, t8;
    if (align % 4 == 0)
    {
        t1                = _mm256_load_pd(base + align * offset[0]);
        t2                = _mm256_load_pd(base + align * offset[1]);
        t3                = _mm256_load_pd(base + align * offset[2]);
        t4                = _mm256_load_pd(base + align * offset[3]);
    }
    else
    {
        t1                = _mm256_loadu_pd(base + align * offset[0]);
        t2                = _mm256_loadu_pd(base + align * offset[1]);
        t3                = _mm256_loadu_pd(base + align * offset[2]);
        t4                = _mm256_loadu_pd(base + align * offset[3]);
    }
    t5                = _mm256_unpacklo_pd(t1, t2);
    t6                = _mm256_unpackhi_pd(t1, t2);
    t7                = _mm256_unpacklo_pd(t3, t4);
    t8                = _mm256_unpackhi_pd(t3, t4);
    v0->simdInternal_ = _mm256_permute2f128_pd(t5, t7, 0x20);
    v1->simdInternal_ = _mm256_permute2f128_pd(t6, t8, 0x20);
    v2->simdInternal_ = _mm256_permute2f128_pd(t5, t7, 0x31);
}

template <int align>
static inline void gmx_simdcall
transposeScatterStoreU(double *            base,
                       const std::int32_t  offset[],
                       SimdDouble          v0,
                       SimdDouble          v1,
                       SimdDouble          v2)
{
    __m256d t0, t1, t2;


    assert(std::size_t(offset) % 16 == 0);

    // v0: x0 x1 | x2 x3
    // v1: y0 y1 | y2 y3
    // v2: z0 z1 | z2 z3

    t0 = _mm256_unpacklo_pd(v0.simdInternal_.native_register(), v1.simdInternal_.native_register()); // x0 y0 | x2 y2
    t1 = _mm256_unpackhi_pd(v0.simdInternal_.native_register(), v1.simdInternal_.native_register()); // x1 y1 | x3 y3
    t2 = _mm256_unpackhi_pd(v2.simdInternal_.native_register(), v2.simdInternal_.native_register()); // z1 z1 | z3 z3

    _mm_storeu_pd(base + align * offset[0], _mm256_castpd256_pd128(t0));
    _mm_storeu_pd(base + align * offset[1], _mm256_castpd256_pd128(t1));
    _mm_storeu_pd(base + align * offset[2], _mm256_extractf128_pd(t0, 0x1));
    _mm_storeu_pd(base + align * offset[3], _mm256_extractf128_pd(t1, 0x1));
    _mm_store_sd(base + align * offset[0] + 2, _mm256_castpd256_pd128(v2.simdInternal_.native_register()));
    _mm_store_sd(base + align * offset[1] + 2, _mm256_castpd256_pd128(t2));
    _mm_store_sd(base + align * offset[2] + 2, _mm256_extractf128_pd(v2.simdInternal_.native_register(), 0x1));
    _mm_store_sd(base + align * offset[3] + 2, _mm256_extractf128_pd(t2, 0x1));
}

template <int align>
static inline void gmx_simdcall
transposeScatterIncrU(double *            base,
                      const std::int32_t  offset[],
                      SimdDouble          v0,
                      SimdDouble          v1,
                      SimdDouble          v2)
{
    __m256d t0, t1;
    __m128d t2, tA, tB;

    assert(std::size_t(offset) % 16 == 0);

    if (align % 4 == 0)
    {
        // we can use aligned load/store
        t0 = _mm256_setzero_pd();
        avx256Transpose4By4(&v0.simdInternal_, &v1.simdInternal_, &v2.simdInternal_, &t0);
        _mm256_store_pd(base + align * offset[0], _mm256_add_pd(_mm256_load_pd(base + align * offset[0]), v0.simdInternal_.native_register()));
        _mm256_store_pd(base + align * offset[1], _mm256_add_pd(_mm256_load_pd(base + align * offset[1]), v1.simdInternal_.native_register()));
        _mm256_store_pd(base + align * offset[2], _mm256_add_pd(_mm256_load_pd(base + align * offset[2]), v2.simdInternal_.native_register()));
        _mm256_store_pd(base + align * offset[3], _mm256_add_pd(_mm256_load_pd(base + align * offset[3]), t0));
    }
    else
    {
        // v0: x0 x1 | x2 x3
        // v1: y0 y1 | y2 y3
        // v2: z0 z1 | z2 z3

        t0 = _mm256_unpacklo_pd(v0.simdInternal_.native_register(), v1.simdInternal_.native_register()); // x0 y0 | x2 y2
        t1 = _mm256_unpackhi_pd(v0.simdInternal_.native_register(), v1.simdInternal_.native_register()); // x1 y1 | x3 y3
        t2 = _mm256_extractf128_pd(v2.simdInternal_.native_register(), 0x1);           // z2 z3

        tA = _mm_loadu_pd(base + align * offset[0]);
        tB = _mm_load_sd(base + align * offset[0] + 2);
        tA = _mm_add_pd(tA, _mm256_castpd256_pd128(t0));
        tB = _mm_add_pd(tB, _mm256_castpd256_pd128(v2.simdInternal_.native_register()));
        _mm_storeu_pd(base + align * offset[0], tA);
        _mm_store_sd(base + align * offset[0] + 2, tB);

        tA = _mm_loadu_pd(base + align * offset[1]);
        tB = _mm_loadh_pd(_mm_setzero_pd(), base + align * offset[1] + 2);
        tA = _mm_add_pd(tA, _mm256_castpd256_pd128(t1));
        tB = _mm_add_pd(tB, _mm256_castpd256_pd128(v2.simdInternal_.native_register()));
        _mm_storeu_pd(base + align * offset[1], tA);
        _mm_storeh_pd(base + align * offset[1] + 2, tB);

        tA = _mm_loadu_pd(base + align * offset[2]);
        tB = _mm_load_sd(base + align * offset[2] + 2);
        tA = _mm_add_pd(tA, _mm256_extractf128_pd(t0, 0x1));
        tB = _mm_add_pd(tB, t2);
        _mm_storeu_pd(base + align * offset[2], tA);
        _mm_store_sd(base + align * offset[2] + 2, tB);

        tA = _mm_loadu_pd(base + align * offset[3]);
        tB = _mm_loadh_pd(_mm_setzero_pd(), base + align * offset[3] + 2);
        tA = _mm_add_pd(tA, _mm256_extractf128_pd(t1, 0x1));
        tB = _mm_add_pd(tB, t2);
        _mm_storeu_pd(base + align * offset[3], tA);
        _mm_storeh_pd(base + align * offset[3] + 2, tB);
    }
}
template <int align>
static inline void gmx_simdcall
transposeScatterDecrU(double *            base,
                      const std::int32_t  offset[],
                      SimdDouble          v0,
                      SimdDouble          v1,
                      SimdDouble          v2)
{
    __m256d t0, t1;
    __m128d t2, tA, tB;

    assert(std::size_t(offset) % 16 == 0);

    if (align % 4 == 0)
    {
        // we can use aligned load/store
        t0 = _mm256_setzero_pd();
        avx256Transpose4By4(&v0.simdInternal_, &v1.simdInternal_, &v2.simdInternal_, &t0);
        _mm256_store_pd(base + align * offset[0], _mm256_sub_pd(_mm256_load_pd(base + align * offset[0]), v0.simdInternal_.native_register()));
        _mm256_store_pd(base + align * offset[1], _mm256_sub_pd(_mm256_load_pd(base + align * offset[1]), v1.simdInternal_.native_register()));
        _mm256_store_pd(base + align * offset[2], _mm256_sub_pd(_mm256_load_pd(base + align * offset[2]), v2.simdInternal_.native_register()));
        _mm256_store_pd(base + align * offset[3], _mm256_sub_pd(_mm256_load_pd(base + align * offset[3]), t0));
    }
    else
    {
        // v0: x0 x1 | x2 x3
        // v1: y0 y1 | y2 y3
        // v2: z0 z1 | z2 z3

        t0 = _mm256_unpacklo_pd(v0.simdInternal_.native_register(), v1.simdInternal_.native_register()); // x0 y0 | x2 y2
        t1 = _mm256_unpackhi_pd(v0.simdInternal_.native_register(), v1.simdInternal_.native_register()); // x1 y1 | x3 y3
        t2 = _mm256_extractf128_pd(v2.simdInternal_.native_register(), 0x1);           // z2 z3

        tA = _mm_loadu_pd(base + align * offset[0]);
        tB = _mm_load_sd(base + align * offset[0] + 2);
        tA = _mm_sub_pd(tA, _mm256_castpd256_pd128(t0));
        tB = _mm_sub_pd(tB, _mm256_castpd256_pd128(v2.simdInternal_.native_register()));
        _mm_storeu_pd(base + align * offset[0], tA);
        _mm_store_sd(base + align * offset[0] + 2, tB);

        tA = _mm_loadu_pd(base + align * offset[1]);
        tB = _mm_loadh_pd(_mm_setzero_pd(), base + align * offset[1] + 2);
        tA = _mm_sub_pd(tA, _mm256_castpd256_pd128(t1));
        tB = _mm_sub_pd(tB, _mm256_castpd256_pd128(v2.simdInternal_.native_register()));
        _mm_storeu_pd(base + align * offset[1], tA);
        _mm_storeh_pd(base + align * offset[1] + 2, tB);

        tA = _mm_loadu_pd(base + align * offset[2]);
        tB = _mm_load_sd(base + align * offset[2] + 2);
        tA = _mm_sub_pd(tA, _mm256_extractf128_pd(t0, 0x1));
        tB = _mm_sub_pd(tB, t2);
        _mm_storeu_pd(base + align * offset[2], tA);
        _mm_store_sd(base + align * offset[2] + 2, tB);

        tA = _mm_loadu_pd(base + align * offset[3]);
        tB = _mm_loadh_pd(_mm_setzero_pd(), base + align * offset[3] + 2);
        tA = _mm_sub_pd(tA, _mm256_extractf128_pd(t1, 0x1));
        tB = _mm_sub_pd(tB, t2);
        _mm_storeu_pd(base + align * offset[3], tA);
        _mm_storeh_pd(base + align * offset[3] + 2, tB);
    }
}

static inline void gmx_simdcall
expandScalarsToTriplets(SimdDouble    scalar,
                        SimdDouble *  triplets0,
                        SimdDouble *  triplets1,
                        SimdDouble *  triplets2)
{
    __m256d t0 = _mm256_permute2f128_pd(scalar.simdInternal_.native_register(), scalar.simdInternal_.native_register(), 0x21);
    __m256d t1 = _mm256_permute_pd(scalar.simdInternal_.native_register(), 0b0000);
    __m256d t2 = _mm256_permute_pd(scalar.simdInternal_.native_register(), 0b1111);
    triplets0->simdInternal_ = _mm256_blend_pd(t1, t0, 0b1100);
    triplets1->simdInternal_ = _mm256_blend_pd(t2, t1, 0b1100);
    triplets2->simdInternal_ = _mm256_blend_pd(t0, t2, 0b1100);
}

template <int align>
static inline void gmx_simdcall
gatherLoadBySimdIntTranspose(const double *  base,
                             SimdDInt32      offset,
                             SimdDouble *    v0,
                             SimdDouble *    v1,
                             SimdDouble *    v2,
                             SimdDouble *    v3)
{
    assert(std::size_t(base) % 32 == 0);
    assert(align % 4 == 0);

    alignas(GMX_SIMD_ALIGNMENT) std::int32_t ioffset[GMX_SIMD_DINT32_WIDTH];
    _mm_store_si128( reinterpret_cast<__m128i *>(ioffset), offset.simdInternal_.native_register());

    v0->simdInternal_ = _mm256_load_pd(base + align * ioffset[0]);
    v1->simdInternal_ = _mm256_load_pd(base + align * ioffset[1]);
    v2->simdInternal_ = _mm256_load_pd(base + align * ioffset[2]);
    v3->simdInternal_ = _mm256_load_pd(base + align * ioffset[3]);

    avx256Transpose4By4(&v0->simdInternal_, &v1->simdInternal_, &v2->simdInternal_, &v3->simdInternal_);
}

template <int align>
static inline void gmx_simdcall
gatherLoadBySimdIntTranspose(const double *    base,
                             SimdDInt32        offset,
                             SimdDouble *      v0,
                             SimdDouble *      v1)
{
    __m128d t1, t2, t3, t4;
    __m256d tA, tB;

    assert(std::size_t(base) % 16 == 0);
    assert(align % 2 == 0);

    alignas(GMX_SIMD_ALIGNMENT) std::int32_t  ioffset[GMX_SIMD_DINT32_WIDTH];
    _mm_store_si128( reinterpret_cast<__m128i *>(ioffset), offset.simdInternal_.native_register());

    t1  = _mm_load_pd(base + align * ioffset[0]);
    t2  = _mm_load_pd(base + align * ioffset[1]);
    t3  = _mm_load_pd(base + align * ioffset[2]);
    t4  = _mm_load_pd(base + align * ioffset[3]);

    tA                = _mm256_insertf128_pd(_mm256_castpd128_pd256(t1), t3, 0x1);
    tB                = _mm256_insertf128_pd(_mm256_castpd128_pd256(t2), t4, 0x1);
    v0->simdInternal_ = _mm256_unpacklo_pd(tA, tB);
    v1->simdInternal_ = _mm256_unpackhi_pd(tA, tB);
}

template <int align>
static inline void gmx_simdcall
gatherLoadUBySimdIntTranspose(const double *  base,
                              SimdDInt32      offset,
                              SimdDouble *    v0,
                              SimdDouble *    v1)
{
    __m128d t1, t2, t3, t4;
    __m256d tA, tB;

    alignas(GMX_SIMD_ALIGNMENT) std::int32_t ioffset[GMX_SIMD_DINT32_WIDTH];
    _mm_store_si128( reinterpret_cast<__m128i *>(ioffset), offset.simdInternal_.native_register());

    t1   = _mm_loadu_pd(base + align * ioffset[0]);
    t2   = _mm_loadu_pd(base + align * ioffset[1]);
    t3   = _mm_loadu_pd(base + align * ioffset[2]);
    t4   = _mm_loadu_pd(base + align * ioffset[3]);

    tA  = _mm256_insertf128_pd(_mm256_castpd128_pd256(t1), t3, 0x1);
    tB  = _mm256_insertf128_pd(_mm256_castpd128_pd256(t2), t4, 0x1);

    v0->simdInternal_ = _mm256_unpacklo_pd(tA, tB);
    v1->simdInternal_ = _mm256_unpackhi_pd(tA, tB);
}

static inline double gmx_simdcall
reduceIncr4ReturnSum(double *    m,
                     SimdDouble  v0,
                     SimdDouble  v1,
                     SimdDouble  v2,
                     SimdDouble  v3)
{
    __m256d t0, t1, t2;
    __m128d a0, a1;

    assert(std::size_t(m) % 32 == 0);

    t0 = _mm256_hadd_pd(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
    t1 = _mm256_hadd_pd(v2.simdInternal_.native_register(), v3.simdInternal_.native_register());
    t2 = _mm256_permute2f128_pd(t0, t1, 0x21);
    t0 = _mm256_add_pd(t0, t2);
    t1 = _mm256_add_pd(t1, t2);
    t0 = _mm256_blend_pd(t0, t1, 0b1100);

    t1 = _mm256_add_pd(t0, _mm256_load_pd(m));
    _mm256_store_pd(m, t1);

    t0  = _mm256_add_pd(t0, _mm256_permute_pd(t0, 0b0101 ));
    a0  = _mm256_castpd256_pd128(t0);
    a1  = _mm256_extractf128_pd(t0, 0x1);
    a0  = _mm_add_sd(a0, a1);

    return *reinterpret_cast<double *>(&a0);
}

// This version is marginally slower than the AVX 4-wide component load
// version on Intel Skylake. On older Intel architectures this version
// is significantly slower.
template <int align>
static inline void gmx_simdcall
gatherLoadUTransposeSafe(const double *        base,
                         const std::int32_t    offset[],
                         SimdDouble *          v0,
                         SimdDouble *          v1,
                         SimdDouble *          v2)
{
    assert(std::size_t(offset) % 16 == 0);

    const SimdDInt32 alignSimd = SimdDInt32(align);

    SimdDInt32       vindex = simdLoad(offset, SimdDInt32Tag());
    vindex = vindex*alignSimd;

    *v0 = _mm256_i32gather_pd(base + 0, vindex.simdInternal_.native_register(), sizeof(double));
    *v1 = _mm256_i32gather_pd(base + 1, vindex.simdInternal_.native_register(), sizeof(double));
    *v2 = _mm256_i32gather_pd(base + 2, vindex.simdInternal_.native_register(), sizeof(double));
}

#elif (defined(NSIMD_AVX512_KNL) || defined(NSIMD_AVX512_SKYLAKE))


static const int c_simdBestPairAlignmentDouble = 2;

namespace
{
// Multiply function optimized for powers of 2, for which it is done by
// shifting. Currently up to 8 is accelerated. Could be accelerated for any
// number with a constexpr log2 function.
template<int n>
SimdDInt32 fastMultiply(SimdDInt32 x)
{
    if (n == 2)
    {
        return {
          packd_t(_mm256_slli_epi32(x.simdInternal_.native_register(), 1))
        };
    }
    else if (n == 4)
    {
        return {
          packd_t(_mm256_slli_epi32(x.simdInternal_.native_register(), 2))
        };
    }
    else if (n == 8)
    {
        return {
          packd_t(_mm256_slli_epi32(x.simdInternal_.native_register(), 3))
        };
    }
    else
    {
        return x * n;
    }
}

template<int align>
static inline void gmx_simdcall
gatherLoadBySimdIntTranspose(const double *, SimdDInt32)
{
    //Nothing to do. Termination of recursion.
}
}


template <int align, typename ... Targs>
static inline void gmx_simdcall
gatherLoadBySimdIntTranspose(const double * base, SimdDInt32 offset, SimdDouble *v, Targs... Fargs)
{
    if (align > 1)
    {
        offset = fastMultiply<align>(offset);
    }
    constexpr size_t scale = sizeof(double);
    v->simdInternal_ = _mm512_i32gather_pd(offset.simdInternal_.native_register(), base, scale);
    gatherLoadBySimdIntTranspose<1>(base+1, offset, Fargs ...);
}

template <int align, typename ... Targs>
static inline void gmx_simdcall
gatherLoadUBySimdIntTranspose(const double *base, SimdDInt32 offset, SimdDouble *v, Targs... Fargs)
{
    gatherLoadBySimdIntTranspose<align>(base, offset, v, Fargs ...);
}

template <int align, typename ... Targs>
static inline void gmx_simdcall
gatherLoadTranspose(const double *base, const std::int32_t offset[], SimdDouble *v, Targs... Fargs)
{
    gatherLoadBySimdIntTranspose<align>(base, simdLoad(offset, SimdDInt32Tag()), v, Fargs ...);
}

template <int align, typename ... Targs>
static inline void gmx_simdcall
gatherLoadUTranspose(const double *base, const std::int32_t offset[], SimdDouble *v, Targs... Fargs)
{
    gatherLoadBySimdIntTranspose<align>(base, simdLoad(offset, SimdDInt32Tag()), v, Fargs ...);
}

template <int align>
static inline void gmx_simdcall
transposeScatterStoreU(double *             base,
                       const std::int32_t   offset[],
                       SimdDouble           v0,
                       SimdDouble           v1,
                       SimdDouble           v2)
{
    SimdDInt32 simdoffset = simdLoad(offset, SimdDInt32Tag());

    if (align > 1)
    {
        simdoffset = fastMultiply<align>(simdoffset);;
    }
    constexpr size_t scale = sizeof(double);
    _mm512_i32scatter_pd(base,   simdoffset.simdInternal_.native_register(), v0.simdInternal_.native_register(), scale);
    _mm512_i32scatter_pd(&(base[1]), simdoffset.simdInternal_.native_register(), v1.simdInternal_.native_register(), scale);
    _mm512_i32scatter_pd(&(base[2]), simdoffset.simdInternal_.native_register(), v2.simdInternal_.native_register(), scale);
}

template <int align>
static inline void gmx_simdcall
transposeScatterIncrU(double *            base,
                      const std::int32_t  offset[],
                      SimdDouble          v0,
                      SimdDouble          v1,
                      SimdDouble          v2)
{
    __m512d t[4], t5, t6, t7, t8;
    alignas(GMX_SIMD_ALIGNMENT) std::int64_t    o[8];
    //TODO: should use fastMultiply
    _mm512_store_epi64(o, _mm512_cvtepi32_epi64(_mm256_mullo_epi32(_mm256_load_si256((const __m256i*)(offset  )), _mm256_set1_epi32(align))));
    t5   = _mm512_unpacklo_pd(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
    t6   = _mm512_unpackhi_pd(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
    t7   = _mm512_unpacklo_pd(v2.simdInternal_.native_register(), _mm512_setzero_pd());
    t8   = _mm512_unpackhi_pd(v2.simdInternal_.native_register(), _mm512_setzero_pd());
    t[0] = _mm512_mask_permutex_pd(t5, avx512Int2Mask(0xCC), t7, 0x4E);
    t[1] = _mm512_mask_permutex_pd(t6, avx512Int2Mask(0xCC), t8, 0x4E);
    t[2] = _mm512_mask_permutex_pd(t7, avx512Int2Mask(0x33), t5, 0x4E);
    t[3] = _mm512_mask_permutex_pd(t8, avx512Int2Mask(0x33), t6, 0x4E);
    if (align < 4)
    {
        for (int i = 0; i < 4; i++)
        {
            _mm512_mask_storeu_pd(base + o[0 + i], avx512Int2Mask(7), _mm512_castpd256_pd512(
                                          _mm256_add_pd(_mm256_loadu_pd(base + o[0 + i]), _mm512_castpd512_pd256(t[i]))));
            _mm512_mask_storeu_pd(base + o[4 + i], avx512Int2Mask(7), _mm512_castpd256_pd512(
                                          _mm256_add_pd(_mm256_loadu_pd(base + o[4 + i]), _mm512_extractf64x4_pd(t[i], 1))));
        }
    }
    else
    {
        if (align % 4 == 0)
        {
            for (int i = 0; i < 4; i++)
            {
                _mm256_store_pd(base + o[0 + i],
                                _mm256_add_pd(_mm256_load_pd(base + o[0 + i]), _mm512_castpd512_pd256(t[i])));
                _mm256_store_pd(base + o[4 + i],
                                _mm256_add_pd(_mm256_load_pd(base + o[4 + i]), _mm512_extractf64x4_pd(t[i], 1)));
            }
        }
        else
        {
            for (int i = 0; i < 4; i++)
            {
                _mm256_storeu_pd(base + o[0 + i],
                                 _mm256_add_pd(_mm256_loadu_pd(base + o[0 + i]), _mm512_castpd512_pd256(t[i])));
                _mm256_storeu_pd(base + o[4 + i],
                                 _mm256_add_pd(_mm256_loadu_pd(base + o[4 + i]), _mm512_extractf64x4_pd(t[i], 1)));
            }
        }
    }
}

template <int align>
static inline void gmx_simdcall
transposeScatterDecrU(double *            base,
                      const std::int32_t  offset[],
                      SimdDouble          v0,
                      SimdDouble          v1,
                      SimdDouble          v2)
{
    __m512d t[4], t5, t6, t7, t8;
    alignas(GMX_SIMD_ALIGNMENT) std::int64_t    o[8];
    //TODO: should use fastMultiply
    _mm512_store_epi64(o, _mm512_cvtepi32_epi64(_mm256_mullo_epi32(_mm256_load_si256((const __m256i*)(offset  )), _mm256_set1_epi32(align))));
    t5   = _mm512_unpacklo_pd(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
    t6   = _mm512_unpackhi_pd(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
    t7   = _mm512_unpacklo_pd(v2.simdInternal_.native_register(), _mm512_setzero_pd());
    t8   = _mm512_unpackhi_pd(v2.simdInternal_.native_register(), _mm512_setzero_pd());
    t[0] = _mm512_mask_permutex_pd(t5, avx512Int2Mask(0xCC), t7, 0x4E);
    t[2] = _mm512_mask_permutex_pd(t7, avx512Int2Mask(0x33), t5, 0x4E);
    t[1] = _mm512_mask_permutex_pd(t6, avx512Int2Mask(0xCC), t8, 0x4E);
    t[3] = _mm512_mask_permutex_pd(t8, avx512Int2Mask(0x33), t6, 0x4E);
    if (align < 4)
    {
        for (int i = 0; i < 4; i++)
        {
            _mm512_mask_storeu_pd(base + o[0 + i], avx512Int2Mask(7), _mm512_castpd256_pd512(
                                          _mm256_sub_pd(_mm256_loadu_pd(base + o[0 + i]), _mm512_castpd512_pd256(t[i]))));
            _mm512_mask_storeu_pd(base + o[4 + i], avx512Int2Mask(7), _mm512_castpd256_pd512(
                                          _mm256_sub_pd(_mm256_loadu_pd(base + o[4 + i]), _mm512_extractf64x4_pd(t[i], 1))));
        }
    }
    else
    {
        if (align % 4 == 0)
        {
            for (int i = 0; i < 4; i++)
            {
                _mm256_store_pd(base + o[0 + i],
                                _mm256_sub_pd(_mm256_load_pd(base + o[0 + i]), _mm512_castpd512_pd256(t[i])));
                _mm256_store_pd(base + o[4 + i],
                                _mm256_sub_pd(_mm256_load_pd(base + o[4 + i]), _mm512_extractf64x4_pd(t[i], 1)));
            }
        }
        else
        {
            for (int i = 0; i < 4; i++)
            {
                _mm256_storeu_pd(base + o[0 + i],
                                 _mm256_sub_pd(_mm256_loadu_pd(base + o[0 + i]), _mm512_castpd512_pd256(t[i])));
                _mm256_storeu_pd(base + o[4 + i],
                                 _mm256_sub_pd(_mm256_loadu_pd(base + o[4 + i]), _mm512_extractf64x4_pd(t[i], 1)));
            }
        }
    }
}

static inline void gmx_simdcall
expandScalarsToTriplets(SimdDouble    scalar,
                        SimdDouble *  triplets0,
                        SimdDouble *  triplets1,
                        SimdDouble *  triplets2)
{
    triplets0->simdInternal_ = _mm512_castsi512_pd(_mm512_permutexvar_epi32(_mm512_set_epi32(5, 4, 5, 4, 3, 2, 3, 2, 3, 2, 1, 0, 1, 0, 1, 0),
                                                                            _mm512_castpd_si512(scalar.simdInternal_.native_register())));
    triplets1->simdInternal_ = _mm512_castsi512_pd(_mm512_permutexvar_epi32(_mm512_set_epi32(11, 10, 9, 8, 9, 8, 9, 8, 7, 6, 7, 6, 7, 6, 5, 4),
                                                                            _mm512_castpd_si512(scalar.simdInternal_.native_register())));
    triplets2->simdInternal_ = _mm512_castsi512_pd(_mm512_permutexvar_epi32(_mm512_set_epi32(15, 14, 15, 14, 15, 14, 13, 12, 13, 12, 13, 12, 11, 10, 11, 10),
                                                                            _mm512_castpd_si512(scalar.simdInternal_.native_register())));
}


static inline double gmx_simdcall
reduceIncr4ReturnSum(double *    m,
                     SimdDouble  v0,
                     SimdDouble  v1,
                     SimdDouble  v2,
                     SimdDouble  v3)
{
    __m512d t0, t2;
    __m256d t3, t4;

    assert(std::size_t(m) % 32 == 0);

    t0 = _mm512_add_pd(v0.simdInternal_.native_register(), _mm512_permute_pd(v0.simdInternal_.native_register(), 0x55));
    t2 = _mm512_add_pd(v2.simdInternal_.native_register(), _mm512_permute_pd(v2.simdInternal_.native_register(), 0x55));
    t0 = _mm512_mask_add_pd(t0, avx512Int2Mask(0xAA), v1.simdInternal_.native_register(), _mm512_permute_pd(v1.simdInternal_.native_register(), 0x55));
    t2 = _mm512_mask_add_pd(t2, avx512Int2Mask(0xAA), v3.simdInternal_.native_register(), _mm512_permute_pd(v3.simdInternal_.native_register(), 0x55));
    t0 = _mm512_add_pd(t0, _mm512_shuffle_f64x2(t0, t0, 0x4E));
    t0 = _mm512_mask_add_pd(t0, avx512Int2Mask(0xF0), t2, _mm512_shuffle_f64x2(t2, t2, 0x4E));
    t0 = _mm512_add_pd(t0, _mm512_shuffle_f64x2(t0, t0, 0xB1));
    t0 = _mm512_mask_shuffle_f64x2(t0, avx512Int2Mask(0x0C), t0, t0, 0xEE);

    t3 = _mm512_castpd512_pd256(t0);
    t4 = _mm256_load_pd(m);
    t4 = _mm256_add_pd(t4, t3);
    _mm256_store_pd(m, t4);

    t0 = _mm512_add_pd(t0, _mm512_permutex_pd(t0, 0x4E));
    t0 = _mm512_add_pd(t0, _mm512_permutex_pd(t0, 0xB1));

    return _mm_cvtsd_f64(_mm512_castpd512_pd128(t0));
}

static inline SimdDouble gmx_simdcall
loadDualHsimd(const double * m0,
              const double * m1)
{
    assert(std::size_t(m0) % 32 == 0);
    assert(std::size_t(m1) % 32 == 0);

    return {
               _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_load_pd(m0)),
                                  _mm256_load_pd(m1), 1)
    };
}

static inline SimdDouble gmx_simdcall
loadDuplicateHsimd(const double * m)
{
    assert(std::size_t(m) % 32 == 0);

    return {
               _mm512_broadcast_f64x4(_mm256_load_pd(m))
    };
}

static inline SimdDouble gmx_simdcall
loadU1DualHsimd(const double * m)
{
    return {
               _mm512_insertf64x4(_mm512_broadcastsd_pd(_mm_load_sd(m)),
                                  _mm256_broadcastsd_pd(_mm_load_sd(m+1)), 1)
    };
}


static inline void gmx_simdcall
storeDualHsimd(double *     m0,
               double *     m1,
               SimdDouble   a)
{
    assert(std::size_t(m0) % 32 == 0);
    assert(std::size_t(m1) % 32 == 0);

    _mm256_store_pd(m0, _mm512_castpd512_pd256(a.simdInternal_.native_register()));
    _mm256_store_pd(m1, _mm512_extractf64x4_pd(a.simdInternal_.native_register(), 1));
}

static inline void gmx_simdcall
incrDualHsimd(double *     m0,
              double *     m1,
              SimdDouble   a)
{
    assert(std::size_t(m0) % 32 == 0);
    assert(std::size_t(m1) % 32 == 0);

    __m256d x;

    // Lower half
    x = _mm256_load_pd(m0);
    x = _mm256_add_pd(x, _mm512_castpd512_pd256(a.simdInternal_.native_register()));
    _mm256_store_pd(m0, x);

    // Upper half
    x = _mm256_load_pd(m1);
    x = _mm256_add_pd(x, _mm512_extractf64x4_pd(a.simdInternal_.native_register(), 1));
    _mm256_store_pd(m1, x);
}

static inline void gmx_simdcall
decrHsimd(double *    m,
          SimdDouble  a)
{
    __m256d t;

    assert(std::size_t(m) % 32 == 0);

    a.simdInternal_ = _mm512_add_pd(a.simdInternal_.native_register(), _mm512_shuffle_f64x2(a.simdInternal_.native_register(), a.simdInternal_.native_register(), 0xEE));
    t               = _mm256_load_pd(m);
    t               = _mm256_sub_pd(t, _mm512_castpd512_pd256(a.simdInternal_.native_register()));
    _mm256_store_pd(m, t);
}


template <int align>
static inline void gmx_simdcall
gatherLoadTransposeHsimd(const double *       base0,
                         const double *       base1,
                         const std::int32_t   offset[],
                         SimdDouble *         v0,
                         SimdDouble *         v1)
{
    __m128i  idx0, idx1;
    __m256i  idx;
    __m512d  tmp1, tmp2;

    assert(std::size_t(offset) % 16 == 0);
    assert(std::size_t(base0) % 16 == 0);
    assert(std::size_t(base1) % 16 == 0);

    idx0 = _mm_load_si128(reinterpret_cast<const __m128i*>(offset));

    static_assert(align == 2 || align == 4, "If more are needed use fastMultiply");
    idx0 = _mm_slli_epi32(idx0, align == 2 ? 1 : 2);

    idx1 = _mm_add_epi32(idx0, _mm_set1_epi32(1));

    idx = _mm256_inserti128_si256(_mm256_castsi128_si256(idx0), idx1, 1);

    constexpr size_t scale = sizeof(double);
    tmp1 = _mm512_i32gather_pd(idx, base0, scale); //TODO: Might be faster to use invidual loads
    tmp2 = _mm512_i32gather_pd(idx, base1, scale);

    v0->simdInternal_ = _mm512_shuffle_f64x2(tmp1, tmp2, 0x44 );
    v1->simdInternal_ = _mm512_shuffle_f64x2(tmp1, tmp2, 0xEE );
}

static inline double gmx_simdcall
reduceIncr4ReturnSumHsimd(double *     m,
                          SimdDouble   v0,
                          SimdDouble   v1)
{
    __m512d  t0;
    __m256d  t2, t3;

    assert(std::size_t(m) % 32 == 0);

    t0 = _mm512_add_pd(v0.simdInternal_.native_register(), _mm512_permutex_pd(v0.simdInternal_.native_register(), 0x4E));
    t0 = _mm512_mask_add_pd(t0, avx512Int2Mask(0xCC), v1.simdInternal_.native_register(), _mm512_permutex_pd(v1.simdInternal_.native_register(), 0x4E));
    t0 = _mm512_add_pd(t0, _mm512_permutex_pd(t0, 0xB1));
    t0 = _mm512_mask_shuffle_f64x2(t0, avx512Int2Mask(0xAA), t0, t0, 0xEE);

    t2 = _mm512_castpd512_pd256(t0);
    t3 = _mm256_load_pd(m);
    t3 = _mm256_add_pd(t3, t2);
    _mm256_store_pd(m, t3);

    t0 = _mm512_add_pd(t0, _mm512_permutex_pd(t0, 0x4E));
    t0 = _mm512_add_pd(t0, _mm512_permutex_pd(t0, 0xB1));

    return _mm_cvtsd_f64(_mm512_castpd512_pd128(t0));
}

static inline SimdDouble gmx_simdcall
loadU4NOffset(const double *m, int offset)
{
    return {
               _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_loadu_pd(m)),
                                  _mm256_loadu_pd(m+offset), 1)
    };
}


#elif (defined(NSIMD_AARCH64) || defined(NSIMD_ARM_NEON))

#if defined(NSIMD_ARM_NEON)


template <int align>
static inline void gmx_simdcall
gatherLoadTranspose(const double *        base,
                    const std::int32_t    offset[],
                    SimdDouble *          v0,
                    SimdDouble *          v1,
                    SimdDouble *          v2,
                    SimdDouble *          v3)
{
    float64x2_t t1, t2, t3, t4;

    assert(std::size_t(offset) % 8 == 0);
    assert(std::size_t(base) % 16 == 0);
    assert(align % 2 == 0);

    t1                = vld1q_f64(base + align * offset[0]);
    t2                = vld1q_f64(base + align * offset[1]);
    t3                = vld1q_f64(base + align * offset[0] + 2);
    t4                = vld1q_f64(base + align * offset[1] + 2);
    v0->simdInternal_ = vuzp1q_f64(t1, t2);
    v1->simdInternal_ = vuzp2q_f64(t1, t2);
    v2->simdInternal_ = vuzp1q_f64(t3, t4);
    v3->simdInternal_ = vuzp2q_f64(t3, t4);
}

template <int align>
static inline void gmx_simdcall
gatherLoadTranspose(const double *        base,
                    const std::int32_t    offset[],
                    SimdDouble *          v0,
                    SimdDouble *          v1)
{
    float64x2_t t1, t2;

    assert(std::size_t(offset) % 8 == 0);
    assert(std::size_t(base) % 16 == 0);
    assert(align % 2 == 0);

    t1                = vld1q_f64(base + align * offset[0]);
    t2                = vld1q_f64(base + align * offset[1]);
    v0->simdInternal_ = vuzp1q_f64(t1, t2);
    v1->simdInternal_ = vuzp2q_f64(t1, t2);
}

static const int c_simdBestPairAlignmentDouble = 2;

template <int align>
static inline void gmx_simdcall
gatherLoadUTranspose(const double *        base,
                     const std::int32_t    offset[],
                     SimdDouble *          v0,
                     SimdDouble *          v1,
                     SimdDouble *          v2)
{
    float64x2_t t1, t2;
    float64x1_t t3, t4;

    assert(std::size_t(offset) % 8 == 0);

    t1                = vld1q_f64(base + align * offset[0]);
    t2                = vld1q_f64(base + align * offset[1]);
    t3                = vld1_f64(base + align * offset[0] + 2);
    t4                = vld1_f64(base + align * offset[1] + 2);
    v0->simdInternal_ = vuzp1q_f64(t1, t2);
    v1->simdInternal_ = vuzp2q_f64(t1, t2);
    v2->simdInternal_ = vcombine_f64(t3, t4);
}

template <int align>
static inline void gmx_simdcall
transposeScatterStoreU(double *             base,
                       const std::int32_t   offset[],
                       SimdDouble           v0,
                       SimdDouble           v1,
                       SimdDouble           v2)
{
    float64x2_t t0, t1;

    assert(std::size_t(offset) % 8 == 0);

    t0  = vuzp1q_f64(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
    t1  = vuzp2q_f64(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
    vst1q_f64(base + align * offset[0], t0);
    vst1q_f64(base + align * offset[1], t1);
    vst1_f64(base + align * offset[0] + 2, vget_low_f64(v2.simdInternal_.native_register()));
    vst1_f64(base + align * offset[1] + 2, vget_high_f64(v2.simdInternal_.native_register()));
}

template <int align>
static inline void gmx_simdcall
transposeScatterIncrU(double *             base,
                      const std::int32_t   offset[],
                      SimdDouble           v0,
                      SimdDouble           v1,
                      SimdDouble           v2)
{
    float64x2_t t0, t1, t2;
    float64x1_t t3;

    assert(std::size_t(offset) % 8 == 0);

    t0  = vuzp1q_f64(v0.simdInternal_.native_register(), v1.simdInternal_.native_register()); // x0 y0
    t1  = vuzp2q_f64(v0.simdInternal_.native_register(), v1.simdInternal_.native_register()); // x1 y1

    t2 = vld1q_f64(base + align * offset[0]);
    t2 = vaddq_f64(t2, t0);
    vst1q_f64(base + align * offset[0], t2);

    t3 = vld1_f64(base + align * offset[0] + 2);
    t3 = vadd_f64(t3, vget_low_f64(v2.simdInternal_.native_register()));
    vst1_f64(base + align * offset[0] + 2, t3);

    t2 = vld1q_f64(base + align * offset[1]);
    t2 = vaddq_f64(t2, t1);
    vst1q_f64(base + align * offset[1], t2);

    t3 = vld1_f64(base + align * offset[1] + 2);
    t3 = vadd_f64(t3, vget_high_f64(v2.simdInternal_.native_register()));
    vst1_f64(base + align * offset[1] + 2, t3);
}

template <int align>
static inline void gmx_simdcall
transposeScatterDecrU(double *             base,
                      const std::int32_t   offset[],
                      SimdDouble           v0,
                      SimdDouble           v1,
                      SimdDouble           v2)
{
    float64x2_t t0, t1, t2;
    float64x1_t t3;

    assert(std::size_t(offset) % 8 == 0);

    t0  = vuzp1q_f64(v0.simdInternal_.native_register(), v1.simdInternal_.native_register()); // x0 y0
    t1  = vuzp2q_f64(v0.simdInternal_.native_register(), v1.simdInternal_.native_register()); // x1 y1

    t2 = vld1q_f64(base + align * offset[0]);
    t2 = vsubq_f64(t2, t0);
    vst1q_f64(base + align * offset[0], t2);

    t3 = vld1_f64(base + align * offset[0] + 2);
    t3 = vsub_f64(t3, vget_low_f64(v2.simdInternal_.native_register()));
    vst1_f64(base + align * offset[0] + 2, t3);

    t2 = vld1q_f64(base + align * offset[1]);
    t2 = vsubq_f64(t2, t1);
    vst1q_f64(base + align * offset[1], t2);

    t3 = vld1_f64(base + align * offset[1] + 2);
    t3 = vsub_f64(t3, vget_high_f64(v2.simdInternal_.native_register()));
    vst1_f64(base + align * offset[1] + 2, t3);
}

static inline void gmx_simdcall
expandScalarsToTriplets(SimdDouble    scalar,
                        SimdDouble *  triplets0,
                        SimdDouble *  triplets1,
                        SimdDouble *  triplets2)
{
    triplets0->simdInternal_ = vuzp1q_f64(scalar.simdInternal_.native_register(), scalar.simdInternal_.native_register());
    triplets1->simdInternal_ = scalar.simdInternal_.native_register();
    triplets2->simdInternal_ = vuzp2q_f64(scalar.simdInternal_.native_register(), scalar.simdInternal_.native_register());
}

template <int align>
static inline void gmx_simdcall
gatherLoadBySimdIntTranspose(const double *  base,
                             SimdDInt32      offset,
                             SimdDouble *    v0,
                             SimdDouble *    v1,
                             SimdDouble *    v2,
                             SimdDouble *    v3)
{
    alignas(GMX_SIMD_ALIGNMENT) std::int32_t     ioffset[GMX_SIMD_DINT32_WIDTH];

    assert(std::size_t(base) % 16 == 0);
    assert(align % 2 == 0);

    vst1_s32(ioffset, offset.simdInternal_.native_register());
    gatherLoadTranspose<align>(base, ioffset, v0, v1, v2, v3);
}


template <int align>
static inline void gmx_simdcall
gatherLoadBySimdIntTranspose(const double *  base,
                             SimdDInt32      offset,
                             SimdDouble *    v0,
                             SimdDouble *    v1)
{
    alignas(GMX_SIMD_ALIGNMENT) std::int32_t     ioffset[GMX_SIMD_DINT32_WIDTH];

    assert(std::size_t(base) % 16 == 0);
    assert(align % 2 == 0);

    vst1_s32(ioffset, offset.simdInternal_.native_register());
    gatherLoadTranspose<align>(base, ioffset, v0, v1);
}

template <int align>
static inline void gmx_simdcall
gatherLoadUBySimdIntTranspose(const double *  base,
                              SimdDInt32      offset,
                              SimdDouble *    v0,
                              SimdDouble *    v1)
{
    alignas(GMX_SIMD_ALIGNMENT) std::int32_t     ioffset[GMX_SIMD_DINT32_WIDTH];

    vst1_s32(ioffset, offset.simdInternal_.native_register());

    float64x2_t t1, t2;

    t1                = vld1q_f64(base + align * ioffset[0]);
    t2                = vld1q_f64(base + align * ioffset[1]);
    v0->simdInternal_ = vuzp1q_f64(t1, t2);
    v1->simdInternal_ = vuzp2q_f64(t1, t2);
}


static inline double gmx_simdcall
reduceIncr4ReturnSum(double *    m,
                     SimdDouble  v0,
                     SimdDouble  v1,
                     SimdDouble  v2,
                     SimdDouble  v3)
{
    float64x2_t t1, t2, t3, t4;

    assert(std::size_t(m) % 8 == 0);

    t1 = vuzp1q_f64(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
    t2 = vuzp2q_f64(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
    t3 = vuzp1q_f64(v2.simdInternal_.native_register(), v3.simdInternal_.native_register());
    t4 = vuzp2q_f64(v2.simdInternal_.native_register(), v3.simdInternal_.native_register());

    t1 = vaddq_f64(t1, t2);
    t3 = vaddq_f64(t3, t4);

    t2 = vaddq_f64(t1, vld1q_f64(m));
    t4 = vaddq_f64(t3, vld1q_f64(m + 2));
    vst1q_f64(m, t2);
    vst1q_f64(m + 2, t4);

    t1 = vaddq_f64(t1, t3);
    t2 = vpaddq_f64(t1, t1);

    return vgetq_lane_f64(t2, 0);
}

#endif

#endif

}      //namespace gm

#endif // GMX_SIMD_IMPL_NSIMD_UTIL_DOUBLE_H
