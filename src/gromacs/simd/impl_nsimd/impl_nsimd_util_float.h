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

#ifndef GMX_SIMD_IMPL_NSIMD_UTIL_FLOAT_H
#define GMX_SIMD_IMPL_NSIMD_UTIL_FLOAT_H

#include "config.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/cxx_adv_api_functions.hpp>
#include <nsimd/nsimd.h>

#include "gromacs/utility/basedefinitions.h"

#include "impl_nsimd_simd_float.h"

namespace gmx
{

#if (defined(NSIMD_SSE2) || defined(NSIMD_SSE42))

template <int align>
static inline void gmx_simdcall
gatherLoadTranspose(const float *        base,
                    const std::int32_t   offset[],
                    SimdFloat *          v0,
                    SimdFloat *          v1,
                    SimdFloat *          v2,
                    SimdFloat *          v3)
{
    assert(std::size_t(base + align * offset[0]) % 16 == 0);
    assert(std::size_t(base + align * offset[1]) % 16 == 0);
    assert(std::size_t(base + align * offset[2]) % 16 == 0);
    assert(std::size_t(base + align * offset[3]) % 16 == 0);

    v0->simdInternal_ = _mm_load_ps( base + align * offset[0] );
    v1->simdInternal_ = _mm_load_ps( base + align * offset[1] );
    v2->simdInternal_ = _mm_load_ps( base + align * offset[2] );
    v3->simdInternal_ = _mm_load_ps( base + align * offset[3] );

    auto tmp_v0 = v0->simdInternal_.native_register();
    auto tmp_v1 =  v0->simdInternal_.native_register(); 
    auto tmp_v2 = v2->simdInternal_.native_register();
    auto tmp_v3 = v3->simdInternal_.native_register();
    _MM_TRANSPOSE4_PS(tmp_v0, tmp_v1, tmp_v2, tmp_v3);
    // _MM_TRANSPOSE4_PS(v0->simdInternal_.native_register(), v0->simdInternal_.native_register(), v2->simdInternal_.native_register(), v3->simdInternal_.native_register());
}

template <int align>
static inline void gmx_simdcall
gatherLoadTranspose(const float *        base,
                    const std::int32_t   offset[],
                    SimdFloat *          v0,
                    SimdFloat *          v1)
{
    __m128 t1, t2;

    v0->simdInternal_ = _mm_castpd_ps(_mm_load_sd( reinterpret_cast<const double *>( base + align * offset[0] ) ));
    v1->simdInternal_ = _mm_castpd_ps(_mm_load_sd( reinterpret_cast<const double *>( base + align * offset[1] ) ));
    t1                = _mm_castpd_ps(_mm_load_sd( reinterpret_cast<const double *>( base + align * offset[2] ) ));
    t2                = _mm_castpd_ps(_mm_load_sd( reinterpret_cast<const double *>( base + align * offset[3] ) ));
    t1                = _mm_unpacklo_ps(v0->simdInternal_.native_register(), t1);
    t2                = _mm_unpacklo_ps(v1->simdInternal_.native_register(), t2);
    v0->simdInternal_ = _mm_unpacklo_ps(t1, t2);
    v1->simdInternal_ = _mm_unpackhi_ps(t1, t2);
}

static const int c_simdBestPairAlignmentFloat = 2;

template <int align>
static inline void gmx_simdcall
gatherLoadUTranspose(const float *        base,
                     const std::int32_t   offset[],
                     SimdFloat *          v0,
                     SimdFloat *          v1,
                     SimdFloat *          v2)
{
    __m128 t1, t2, t3, t4, t5, t6, t7, t8;

    if (align % 4 != 0)
    {
        // general case, not aligned to 4-byte boundary
        t1                = _mm_loadu_ps( base + align * offset[0] );
        t2                = _mm_loadu_ps( base + align * offset[1] );
        t3                = _mm_loadu_ps( base + align * offset[2] );
        t4                = _mm_loadu_ps( base + align * offset[3] );
    }
    else
    {
        // aligned to 4-byte boundary or more
        t1                = _mm_load_ps( base + align * offset[0] );
        t2                = _mm_load_ps( base + align * offset[1] );
        t3                = _mm_load_ps( base + align * offset[2] );
        t4                = _mm_load_ps( base + align * offset[3] );
    }
    t5                = _mm_unpacklo_ps(t1, t2);
    t6                = _mm_unpacklo_ps(t3, t4);
    t7                = _mm_unpackhi_ps(t1, t2);
    t8                = _mm_unpackhi_ps(t3, t4);
    v0->simdInternal_               = _mm_movelh_ps(t5, t6);
    v1->simdInternal_               = _mm_movehl_ps(t6, t5);
    v2->simdInternal_              = _mm_movelh_ps(t7, t8);
}


template <int align>
static inline void gmx_simdcall
transposeScatterStoreU(float *              base,
                       const std::int32_t   offset[],
                       SimdFloat            v0,
                       SimdFloat            v1,
                       SimdFloat            v2)
{
    __m128 t1, t2;

    // general case, not aligned to 4-byte boundary
    t1   = _mm_unpacklo_ps(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
    t2   = _mm_unpackhi_ps(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
    _mm_storel_pi( reinterpret_cast< __m64 *>( base + align * offset[0] ), t1);
    _mm_store_ss(base + align * offset[0] + 2, v2.simdInternal_.native_register());
    _mm_storeh_pi( reinterpret_cast< __m64 *>( base + align * offset[1] ), t1);
    _mm_store_ss(base + align * offset[1] + 2, _mm_shuffle_ps(v2.simdInternal_.native_register(), v2.simdInternal_.native_register(), _MM_SHUFFLE(1, 1, 1, 1)));
    _mm_storel_pi( reinterpret_cast< __m64 *>( base + align * offset[2] ), t2);
    _mm_store_ss(base + align * offset[2] + 2, _mm_shuffle_ps(v2.simdInternal_.native_register(), v2.simdInternal_.native_register(), _MM_SHUFFLE(2, 2, 2, 2)));
    _mm_storeh_pi( reinterpret_cast< __m64 *>( base + align * offset[3] ), t2);
    _mm_store_ss(base + align * offset[3] + 2, _mm_shuffle_ps(v2.simdInternal_.native_register(), v2.simdInternal_.native_register(), _MM_SHUFFLE(3, 3, 3, 3)));
}


template <int align>
static inline void gmx_simdcall
transposeScatterIncrU(float *              base,
                      const std::int32_t   offset[],
                      SimdFloat            v0,
                      SimdFloat            v1,
                      SimdFloat            v2)
{
    __m128 t1, t2, t3, t4, t5, t6, t7, t8, t9, t10;

    if (align < 4)
    {
        t5          = _mm_unpacklo_ps(v1.simdInternal_.native_register(), v2.simdInternal_.native_register());
        t6          = _mm_unpackhi_ps(v1.simdInternal_.native_register(), v2.simdInternal_.native_register());
        t7          = _mm_shuffle_ps(v0.simdInternal_.native_register(), t5, _MM_SHUFFLE(1, 0, 0, 0));
        t8          = _mm_shuffle_ps(v0.simdInternal_.native_register(), t5, _MM_SHUFFLE(3, 2, 0, 1));
        t9          = _mm_shuffle_ps(v0.simdInternal_.native_register(), t6, _MM_SHUFFLE(1, 0, 0, 2));
        t10         = _mm_shuffle_ps(v0.simdInternal_.native_register(), t6, _MM_SHUFFLE(3, 2, 0, 3));

        t1          = _mm_load_ss(base + align * offset[0]);
        t1          = _mm_loadh_pi(t1, reinterpret_cast< __m64 *>(base + align * offset[0] + 1));
        t1          = _mm_add_ps(t1, t7);
        _mm_store_ss(base + align * offset[0], t1);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[0] + 1), t1);

        t2          = _mm_load_ss(base + align * offset[1]);
        t2          = _mm_loadh_pi(t2, reinterpret_cast< __m64 *>(base + align * offset[1] + 1));
        t2          = _mm_add_ps(t2, t8);
        _mm_store_ss(base + align * offset[1], t2);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[1] + 1), t2);

        t3          = _mm_load_ss(base + align * offset[2]);
        t3          = _mm_loadh_pi(t3, reinterpret_cast< __m64 *>(base + align * offset[2] + 1));
        t3          = _mm_add_ps(t3, t9);
        _mm_store_ss(base + align * offset[2], t3);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[2] + 1), t3);

        t4          = _mm_load_ss(base + align * offset[3]);
        t4          = _mm_loadh_pi(t4, reinterpret_cast< __m64 *>(base + align * offset[3] + 1));
        t4          = _mm_add_ps(t4, t10);
        _mm_store_ss(base + align * offset[3], t4);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[3] + 1), t4);
    }
    else
    {
        // Extra elements means we can use full width-4 load/store operations

        t1  = _mm_unpacklo_ps(v0.simdInternal_.native_register(), v2.simdInternal_.native_register()); // x0 z0 x1 z1
        t2  = _mm_unpackhi_ps(v0.simdInternal_.native_register(), v2.simdInternal_.native_register()); // x2 z2 x3 z3
        t3  = _mm_unpacklo_ps(v1.simdInternal_.native_register(), _mm_setzero_ps()); // y0  0 y1  0
        t4  = _mm_unpackhi_ps(v1.simdInternal_.native_register(), _mm_setzero_ps()); // y2  0 y3  0
        t5  = _mm_unpacklo_ps(t1, t3);                             // x0 y0 z0  0
        t6  = _mm_unpackhi_ps(t1, t3);                             // x1 y1 z1  0
        t7  = _mm_unpacklo_ps(t2, t4);                             // x2 y2 z2  0
        t8  = _mm_unpackhi_ps(t2, t4);                             // x3 y3 z3  0

        if (align % 4 == 0)
        {
            // alignment is a multiple of 4
            _mm_store_ps(base + align * offset[0], _mm_add_ps(_mm_load_ps(base + align * offset[0]), t5));
            _mm_store_ps(base + align * offset[1], _mm_add_ps(_mm_load_ps(base + align * offset[1]), t6));
            _mm_store_ps(base + align * offset[2], _mm_add_ps(_mm_load_ps(base + align * offset[2]), t7));
            _mm_store_ps(base + align * offset[3], _mm_add_ps(_mm_load_ps(base + align * offset[3]), t8));
        }
        else
        {
            // alignment >=5, but not a multiple of 4
            _mm_storeu_ps(base + align * offset[0], _mm_add_ps(_mm_loadu_ps(base + align * offset[0]), t5));
            _mm_storeu_ps(base + align * offset[1], _mm_add_ps(_mm_loadu_ps(base + align * offset[1]), t6));
            _mm_storeu_ps(base + align * offset[2], _mm_add_ps(_mm_loadu_ps(base + align * offset[2]), t7));
            _mm_storeu_ps(base + align * offset[3], _mm_add_ps(_mm_loadu_ps(base + align * offset[3]), t8));
        }
    }
}

template <int align>
static inline void gmx_simdcall
transposeScatterDecrU(float *              base,
                      const std::int32_t   offset[],
                      SimdFloat            v0,
                      SimdFloat            v1,
                      SimdFloat            v2)
{
    // This implementation is identical to the increment version, apart from using subtraction instead
    __m128 t1, t2, t3, t4, t5, t6, t7, t8, t9, t10;

    if (align < 4)
    {
        t5          = _mm_unpacklo_ps(v1.simdInternal_.native_register(), v2.simdInternal_.native_register());
        t6          = _mm_unpackhi_ps(v1.simdInternal_.native_register(), v2.simdInternal_.native_register());
        t7          = _mm_shuffle_ps(v0.simdInternal_.native_register(), t5, _MM_SHUFFLE(1, 0, 0, 0));
        t8          = _mm_shuffle_ps(v0.simdInternal_.native_register(), t5, _MM_SHUFFLE(3, 2, 0, 1));
        t9          = _mm_shuffle_ps(v0.simdInternal_.native_register(), t6, _MM_SHUFFLE(1, 0, 0, 2));
        t10         = _mm_shuffle_ps(v0.simdInternal_.native_register(), t6, _MM_SHUFFLE(3, 2, 0, 3));

        t1          = _mm_load_ss(base + align * offset[0]);
        t1          = _mm_loadh_pi(t1, reinterpret_cast< __m64 *>(base + align * offset[0] + 1));
        t1          = _mm_sub_ps(t1, t7);
        _mm_store_ss(base + align * offset[0], t1);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[0] + 1), t1);

        t2          = _mm_load_ss(base + align * offset[1]);
        t2          = _mm_loadh_pi(t2, reinterpret_cast< __m64 *>(base + align * offset[1] + 1));
        t2          = _mm_sub_ps(t2, t8);
        _mm_store_ss(base + align * offset[1], t2);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[1] + 1), t2);

        t3          = _mm_load_ss(base + align * offset[2]);
        t3          = _mm_loadh_pi(t3, reinterpret_cast< __m64 *>(base + align * offset[2] + 1));
        t3          = _mm_sub_ps(t3, t9);
        _mm_store_ss(base + align * offset[2], t3);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[2] + 1), t3);

        t4          = _mm_load_ss(base + align * offset[3]);
        t4          = _mm_loadh_pi(t4, reinterpret_cast< __m64 *>(base + align * offset[3] + 1));
        t4          = _mm_sub_ps(t4, t10);
        _mm_store_ss(base + align * offset[3], t4);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[3] + 1), t4);
    }
    else
    {
        // Extra elements means we can use full width-4 load/store operations

        t1  = _mm_unpacklo_ps(v0.simdInternal_.native_register(), v2.simdInternal_.native_register()); // x0 z0 x1 z1
        t2  = _mm_unpackhi_ps(v0.simdInternal_.native_register(), v2.simdInternal_.native_register()); // x2 z2 x3 z3
        t3  = _mm_unpacklo_ps(v1.simdInternal_.native_register(), _mm_setzero_ps()); // y0  0 y1  0
        t4  = _mm_unpackhi_ps(v1.simdInternal_.native_register(), _mm_setzero_ps()); // y2  0 y3  0
        t5  = _mm_unpacklo_ps(t1, t3);                             // x0 y0 z0  0
        t6  = _mm_unpackhi_ps(t1, t3);                             // x1 y1 z1  0
        t7  = _mm_unpacklo_ps(t2, t4);                             // x2 y2 z2  0
        t8  = _mm_unpackhi_ps(t2, t4);                             // x3 y3 z3  0

        if (align % 4 == 0)
        {
            // alignment is a multiple of 4
            _mm_store_ps(base + align * offset[0], _mm_sub_ps(_mm_load_ps(base + align * offset[0]), t5));
            _mm_store_ps(base + align * offset[1], _mm_sub_ps(_mm_load_ps(base + align * offset[1]), t6));
            _mm_store_ps(base + align * offset[2], _mm_sub_ps(_mm_load_ps(base + align * offset[2]), t7));
            _mm_store_ps(base + align * offset[3], _mm_sub_ps(_mm_load_ps(base + align * offset[3]), t8));
        }
        else
        {
            // alignment >=5, but not a multiple of 4
            _mm_storeu_ps(base + align * offset[0], _mm_sub_ps(_mm_loadu_ps(base + align * offset[0]), t5));
            _mm_storeu_ps(base + align * offset[1], _mm_sub_ps(_mm_loadu_ps(base + align * offset[1]), t6));
            _mm_storeu_ps(base + align * offset[2], _mm_sub_ps(_mm_loadu_ps(base + align * offset[2]), t7));
            _mm_storeu_ps(base + align * offset[3], _mm_sub_ps(_mm_loadu_ps(base + align * offset[3]), t8));
        }
    }
}

static inline void gmx_simdcall
expandScalarsToTriplets(SimdFloat    scalar,
                        SimdFloat *  triplets0,
                        SimdFloat *  triplets1,
                        SimdFloat *  triplets2)
{
    triplets0->simdInternal_ = _mm_shuffle_ps(scalar.simdInternal_.native_register(), scalar.simdInternal_.native_register(), _MM_SHUFFLE(1, 0, 0, 0));
    triplets1->simdInternal_ = _mm_shuffle_ps(scalar.simdInternal_.native_register(), scalar.simdInternal_.native_register(), _MM_SHUFFLE(2, 2, 1, 1));
    triplets2->simdInternal_ = _mm_shuffle_ps(scalar.simdInternal_.native_register(), scalar.simdInternal_.native_register(), _MM_SHUFFLE(3, 3, 3, 2));
}


template <int align>
static inline void gmx_simdcall
gatherLoadBySimdIntTranspose(const float *  base,
                             SimdFInt32     offset,
                             SimdFloat *    v0,
                             SimdFloat *    v1,
                             SimdFloat *    v2,
                             SimdFloat *    v3)
{
    // For present-generation x86 CPUs it appears to be faster to simply
    // store the SIMD integer to memory and then use the normal load operations.
    // This is likely because (a) the extract function is expensive, and (b)
    // the alignment scaling can often be done as part of the load instruction
    // (which is even cheaper than doing it in SIMD registers).
    alignas(GMX_SIMD_ALIGNMENT) std::int32_t ioffset[GMX_SIMD_FINT32_WIDTH];
    _mm_store_si128( (__m128i *)ioffset, offset.simdInternal_.native_register());
    gatherLoadTranspose<align>(base, ioffset, v0, v1, v2, v3);
}

template <int align>
static inline void gmx_simdcall
gatherLoadBySimdIntTranspose(const float *   base,
                             SimdFInt32      offset,
                             SimdFloat *     v0,
                             SimdFloat *     v1)
{
    // For present-generation x86 CPUs it appears to be faster to simply
    // store the SIMD integer to memory and then use the normal load operations.
    // This is likely because (a) the extract function is expensive, and (b)
    // the alignment scaling can often be done as part of the load instruction
    // (which is even cheaper than doing it in SIMD registers).
    alignas(GMX_SIMD_ALIGNMENT) std::int32_t ioffset[GMX_SIMD_FINT32_WIDTH];
    _mm_store_si128( (__m128i *)ioffset, offset.simdInternal_.native_register());
    gatherLoadTranspose<align>(base, ioffset, v0, v1);
}



template <int align>
static inline void gmx_simdcall
gatherLoadUBySimdIntTranspose(const float *  base,
                              SimdFInt32     offset,
                              SimdFloat *    v0,
                              SimdFloat *    v1)
{
    // For present-generation x86 CPUs it appears to be faster to simply
    // store the SIMD integer to memory and then use the normal load operations.
    // This is likely because (a) the extract function is expensive, and (b)
    // the alignment scaling can often be done as part of the load instruction
    // (which is even cheaper than doing it in SIMD registers).
    alignas(GMX_SIMD_ALIGNMENT) std::int32_t ioffset[GMX_SIMD_FINT32_WIDTH];
    _mm_store_si128( (__m128i *)ioffset, offset.simdInternal_.native_register());
    gatherLoadTranspose<align>(base, ioffset, v0, v1);
}

static inline float gmx_simdcall
reduceIncr4ReturnSum(float *    m,
                     SimdFloat  v0,
                     SimdFloat  v1,
                     SimdFloat  v2,
                     SimdFloat  v3)
{
    auto tmp_v0 = v0.simdInternal_.native_register(), tmp_v1 =  v1.simdInternal_.native_register(), tmp_v2 = v2.simdInternal_.native_register(), tmp_v3 = v3.simdInternal_.native_register();
    _MM_TRANSPOSE4_PS(tmp_v0, tmp_v1, tmp_v2, tmp_v3);
    v0.simdInternal_ = _mm_add_ps(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
    v2.simdInternal_ = _mm_add_ps(v2.simdInternal_.native_register(), v3.simdInternal_.native_register());
    v0.simdInternal_ = _mm_add_ps(v0.simdInternal_.native_register(), v2.simdInternal_.native_register());
    v2.simdInternal_ = _mm_add_ps(v0.simdInternal_.native_register(), _mm_load_ps(m));

    assert(std::size_t(m) % 16 == 0);
    _mm_store_ps(m, v2.simdInternal_.native_register());

    __m128 b = _mm_add_ps(v0.simdInternal_.native_register(), _mm_shuffle_ps(v0.simdInternal_.native_register(), v0.simdInternal_.native_register(), _MM_SHUFFLE(1, 0, 3, 2)));
    b = _mm_add_ss(b, _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 3, 2, 1)));
    return *reinterpret_cast<float *>(&b);
}

#elif (defined(NSIMD_AVX) || defined(NSIMD_AVX2))

/* This is an internal helper function used by the three functions storing,
 * incrementing, or decrementing data. Do NOT use it outside this file.
 *
 * Input v0: [x0 x1 x2 x3 x4 x5 x6 x7]
 * Input v1: [y0 y1 y2 y3 y4 y5 y6 y7]
 * Input v2: [z0 z1 z2 z3 z4 z5 z6 z7]
 * Input v3: Unused
 *
 * Output v0: [x0 y0 z0 -  x4 y4 z4 - ]
 * Output v1: [x1 y1 z1 -  x5 y5 z5 - ]
 * Output v2: [x2 y2 z2 -  x6 y6 z6 - ]
 * Output v3: [x3 y3 z3 -  x7 y7 z7 - ]
 *
 * Here, - means undefined. Note that such values will not be zero!
 */
static inline void gmx_simdcall
avx256Transpose3By4InLanes(__m256 * v0,
                           __m256 * v1,
                           __m256 * v2,
                           __m256 * v3)
{ 
    __m256 t1 = _mm256_unpacklo_ps(*v0, *v1);
    __m256 t2 = _mm256_unpackhi_ps(*v0, *v1);
    *v0       = _mm256_shuffle_ps(t1, *v2, _MM_SHUFFLE(0, 0, 1, 0));
    *v1       = _mm256_shuffle_ps(t1, *v2, _MM_SHUFFLE(0, 1, 3, 2));
    *v3       = _mm256_shuffle_ps(t2, *v2, _MM_SHUFFLE(0, 3, 3, 2));
    *v2       = _mm256_shuffle_ps(t2, *v2, _MM_SHUFFLE(0, 2, 1, 0));
}

static inline void gmx_simdcall
avx256Transpose3By4InLanes(nsimd::pack<float> * v0,
                           nsimd::pack<float> * v1,
                           nsimd::pack<float> * v2,
                           __m256 * v3)
{
  __m256 tmp0 = v0->native_register(), tmp1 = v1->native_register(),
         tmp2 = v2->native_register();
  __m256 t1 = _mm256_unpacklo_ps(tmp0, tmp1);
  __m256 t2 = _mm256_unpackhi_ps(tmp0, tmp1);
  tmp0 = _mm256_shuffle_ps(t1, tmp2, _MM_SHUFFLE(0, 0, 1, 0));
  tmp1 = _mm256_shuffle_ps(t1, tmp2, _MM_SHUFFLE(0, 1, 3, 2));
  *v3 = _mm256_shuffle_ps(t2, tmp2, _MM_SHUFFLE(0, 3, 3, 2));
  tmp2 = _mm256_shuffle_ps(t2, tmp2, _MM_SHUFFLE(0, 2, 1, 0));

  *v0 = tmp0;
  *v1 = tmp1;
  *v2 = tmp2;
}

template <int align>
static inline void gmx_simdcall
gatherLoadTranspose(const float *        base,
                    const std::int32_t   offset[],
                    SimdFloat *          v0,
                    SimdFloat *          v1,
                    SimdFloat *          v2,
                    SimdFloat *          v3)
{
    __m128 t1, t2, t3, t4, t5, t6, t7, t8;
    __m256 tA, tB, tC, tD;

    assert(std::size_t(offset) % 32 == 0);
    assert(std::size_t(base) % 16 == 0);
    assert(align % 4 == 0);

    t1  = _mm_load_ps( base + align * offset[0] );
    t2  = _mm_load_ps( base + align * offset[1] );
    t3  = _mm_load_ps( base + align * offset[2] );
    t4  = _mm_load_ps( base + align * offset[3] );
    t5  = _mm_load_ps( base + align * offset[4] );
    t6  = _mm_load_ps( base + align * offset[5] );
    t7  = _mm_load_ps( base + align * offset[6] );
    t8  = _mm_load_ps( base + align * offset[7] );

    v0->simdInternal_ = _mm256_insertf128_ps(_mm256_castps128_ps256(t1), t5, 0x1);
    v1->simdInternal_ = _mm256_insertf128_ps(_mm256_castps128_ps256(t2), t6, 0x1);
    v2->simdInternal_ = _mm256_insertf128_ps(_mm256_castps128_ps256(t3), t7, 0x1);
    v3->simdInternal_ = _mm256_insertf128_ps(_mm256_castps128_ps256(t4), t8, 0x1);

    tA  = _mm256_unpacklo_ps(v0->simdInternal_.native_register(), v1->simdInternal_.native_register());
    tB  = _mm256_unpacklo_ps(v2->simdInternal_.native_register(), v3->simdInternal_.native_register());
    tC  = _mm256_unpackhi_ps(v0->simdInternal_.native_register(), v1->simdInternal_.native_register());
    tD  = _mm256_unpackhi_ps(v2->simdInternal_.native_register(), v3->simdInternal_.native_register());

    v0->simdInternal_ = _mm256_shuffle_ps(tA, tB, _MM_SHUFFLE(1, 0, 1, 0));
    v1->simdInternal_ = _mm256_shuffle_ps(tA, tB, _MM_SHUFFLE(3, 2, 3, 2));
    v2->simdInternal_ = _mm256_shuffle_ps(tC, tD, _MM_SHUFFLE(1, 0, 1, 0));
    v3->simdInternal_ = _mm256_shuffle_ps(tC, tD, _MM_SHUFFLE(3, 2, 3, 2));
}

template <int align>
static inline void gmx_simdcall
gatherLoadTranspose(const float *        base,
                    const std::int32_t   offset[],
                    SimdFloat *          v0,
                    SimdFloat *          v1)
{
    __m128 t1, t2, t3, t4, t5, t6, t7, t8;
    __m256 tA, tB, tC, tD;

    assert(std::size_t(offset) % 32 == 0);
    assert(std::size_t(base) % 8 == 0);
    assert(align % 2 == 0);

    t1  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>( base + align * offset[0] ) );
    t2  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>( base + align * offset[1] ) );
    t3  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>( base + align * offset[2] ) );
    t4  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>( base + align * offset[3] ) );
    t5  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>( base + align * offset[4] ) );
    t6  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>( base + align * offset[5] ) );
    t7  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>( base + align * offset[6] ) );
    t8  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>( base + align * offset[7] ) );

    tA  = _mm256_insertf128_ps(_mm256_castps128_ps256(t1), t5, 0x1);
    tB  = _mm256_insertf128_ps(_mm256_castps128_ps256(t2), t6, 0x1);
    tC  = _mm256_insertf128_ps(_mm256_castps128_ps256(t3), t7, 0x1);
    tD  = _mm256_insertf128_ps(_mm256_castps128_ps256(t4), t8, 0x1);

    tA                = _mm256_unpacklo_ps(tA, tC);
    tB                = _mm256_unpacklo_ps(tB, tD);
    v0->simdInternal_ = _mm256_unpacklo_ps(tA, tB);
    v1->simdInternal_ = _mm256_unpackhi_ps(tA, tB);
}

static const int c_simdBestPairAlignmentFloat = 2;

// With the implementation below, thread-sanitizer can detect false positives.
// For loading a triplet, we load 4 floats and ignore the last. Another thread
// might write to this element, but that will not affect the result.
// On AVX2 we can use a gather intrinsic instead.
template <int align>
static inline void gmx_simdcall
gatherLoadUTranspose(const float *        base,
                     const std::int32_t   offset[],
                     SimdFloat *          v0,
                     SimdFloat *          v1,
                     SimdFloat *          v2)
{
    __m256  t1, t2, t3, t4, t5, t6, t7, t8;

    assert(std::size_t(offset) % 32 == 0);

    if (align % 4 == 0)
    {
        // we can use aligned loads since base should also be aligned in this case
        assert(std::size_t(base) % 16 == 0);
        t1  = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps( base + align * offset[0] )),
                                   _mm_load_ps( base + align * offset[4] ), 0x1);
        t2  = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(base + align * offset[1] )),
                                   _mm_load_ps( base + align * offset[5] ), 0x1);
        t3  = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(base + align * offset[2] )),
                                   _mm_load_ps( base + align * offset[6] ), 0x1);
        t4  = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(base + align * offset[3] )),
                                   _mm_load_ps( base + align * offset[7] ), 0x1);
    }
    else
    {
        // Use unaligned loads
        t1  = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps( base + align * offset[0] )),
                                   _mm_loadu_ps( base + align * offset[4] ), 0x1);
        t2  = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(base + align * offset[1] )),
                                   _mm_loadu_ps( base + align * offset[5] ), 0x1);
        t3  = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(base + align * offset[2] )),
                                   _mm_loadu_ps( base + align * offset[6] ), 0x1);
        t4  = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(base + align * offset[3] )),
                                   _mm_loadu_ps( base + align * offset[7] ), 0x1);
    }

    t5                = _mm256_unpacklo_ps(t1, t2);
    t6                = _mm256_unpacklo_ps(t3, t4);
    t7                = _mm256_unpackhi_ps(t1, t2);
    t8                = _mm256_unpackhi_ps(t3, t4);
    v0->simdInternal_ = _mm256_shuffle_ps(t5, t6, _MM_SHUFFLE(1, 0, 1, 0));
    v1->simdInternal_ = _mm256_shuffle_ps(t5, t6, _MM_SHUFFLE(3, 2, 3, 2));
    v2->simdInternal_ = _mm256_shuffle_ps(t7, t8, _MM_SHUFFLE(1, 0, 1, 0));
}

template <int align>
static inline void gmx_simdcall
transposeScatterStoreU(float *              base,
                       const std::int32_t   offset[],
                       SimdFloat            v0,
                       SimdFloat            v1,
                       SimdFloat            v2)
{
    __m256  tv3;
    __m128i mask = _mm_set_epi32(0, -1, -1, -1);

    assert(std::size_t(offset) % 32 == 0);

    avx256Transpose3By4InLanes(&v0.simdInternal_, &v1.simdInternal_, &v2.simdInternal_, &tv3);
    _mm_maskstore_ps( base + align * offset[0], mask, _mm256_castps256_ps128(v0.simdInternal_.native_register()));
    _mm_maskstore_ps( base + align * offset[1], mask, _mm256_castps256_ps128(v1.simdInternal_.native_register()));
    _mm_maskstore_ps( base + align * offset[2], mask, _mm256_castps256_ps128(v2.simdInternal_.native_register()));
    _mm_maskstore_ps( base + align * offset[3], mask, _mm256_castps256_ps128(tv3));
    _mm_maskstore_ps( base + align * offset[4], mask, _mm256_extractf128_ps(v0.simdInternal_.native_register(), 0x1));
    _mm_maskstore_ps( base + align * offset[5], mask, _mm256_extractf128_ps(v1.simdInternal_.native_register(), 0x1));
    _mm_maskstore_ps( base + align * offset[6], mask, _mm256_extractf128_ps(v2.simdInternal_.native_register(), 0x1));
    _mm_maskstore_ps( base + align * offset[7], mask, _mm256_extractf128_ps(tv3, 0x1));
}

template <int align>
static inline void gmx_simdcall
transposeScatterIncrU(float *              base,
                      const std::int32_t   offset[],
                      SimdFloat            v0,
                      SimdFloat            v1,
                      SimdFloat            v2)
{
    __m256 t1, t2, t3, t4, t5, t6, t7, t8, t9, t10;
    __m128 tA, tB, tC, tD, tE, tF, tG, tH, tX;

    if (align < 4)
    {
        t5          = _mm256_unpacklo_ps(v1.simdInternal_.native_register(), v2.simdInternal_.native_register());
        t6          = _mm256_unpackhi_ps(v1.simdInternal_.native_register(), v2.simdInternal_.native_register());
        t7          = _mm256_shuffle_ps(v0.simdInternal_.native_register(), t5, _MM_SHUFFLE(1, 0, 0, 0));
        t8          = _mm256_shuffle_ps(v0.simdInternal_.native_register(), t5, _MM_SHUFFLE(3, 2, 0, 1));
        t9          = _mm256_shuffle_ps(v0.simdInternal_.native_register(), t6, _MM_SHUFFLE(1, 0, 0, 2));
        t10         = _mm256_shuffle_ps(v0.simdInternal_.native_register(), t6, _MM_SHUFFLE(3, 2, 0, 3));

        tA          = _mm256_castps256_ps128(t7);
        tB          = _mm256_castps256_ps128(t8);
        tC          = _mm256_castps256_ps128(t9);
        tD          = _mm256_castps256_ps128(t10);
        tE          = _mm256_extractf128_ps(t7, 0x1);
        tF          = _mm256_extractf128_ps(t8, 0x1);
        tG          = _mm256_extractf128_ps(t9, 0x1);
        tH          = _mm256_extractf128_ps(t10, 0x1);

        tX          = _mm_load_ss(base + align * offset[0]);
        tX          = _mm_loadh_pi(tX, reinterpret_cast< __m64 *>(base + align * offset[0] + 1));
        tX          = _mm_add_ps(tX, tA);
        _mm_store_ss(base + align * offset[0], tX);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[0] + 1), tX);

        tX          = _mm_load_ss(base + align * offset[1]);
        tX          = _mm_loadh_pi(tX, reinterpret_cast< __m64 *>(base + align * offset[1] + 1));
        tX          = _mm_add_ps(tX, tB);
        _mm_store_ss(base + align * offset[1], tX);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[1] + 1), tX);

        tX          = _mm_load_ss(base + align * offset[2]);
        tX          = _mm_loadh_pi(tX, reinterpret_cast< __m64 *>(base + align * offset[2] + 1));
        tX          = _mm_add_ps(tX, tC);
        _mm_store_ss(base + align * offset[2], tX);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[2] + 1), tX);

        tX          = _mm_load_ss(base + align * offset[3]);
        tX          = _mm_loadh_pi(tX, reinterpret_cast< __m64 *>(base + align * offset[3] + 1));
        tX          = _mm_add_ps(tX, tD);
        _mm_store_ss(base + align * offset[3], tX);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[3] + 1), tX);

        tX          = _mm_load_ss(base + align * offset[4]);
        tX          = _mm_loadh_pi(tX, reinterpret_cast< __m64 *>(base + align * offset[4] + 1));
        tX          = _mm_add_ps(tX, tE);
        _mm_store_ss(base + align * offset[4], tX);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[4] + 1), tX);

        tX          = _mm_load_ss(base + align * offset[5]);
        tX          = _mm_loadh_pi(tX, reinterpret_cast< __m64 *>(base + align * offset[5] + 1));
        tX          = _mm_add_ps(tX, tF);
        _mm_store_ss(base + align * offset[5], tX);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[5] + 1), tX);

        tX          = _mm_load_ss(base + align * offset[6]);
        tX          = _mm_loadh_pi(tX, reinterpret_cast< __m64 *>(base + align * offset[6] + 1));
        tX          = _mm_add_ps(tX, tG);
        _mm_store_ss(base + align * offset[6], tX);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[6] + 1), tX);

        tX          = _mm_load_ss(base + align * offset[7]);
        tX          = _mm_loadh_pi(tX, reinterpret_cast< __m64 *>(base + align * offset[7] + 1));
        tX          = _mm_add_ps(tX, tH);
        _mm_store_ss(base + align * offset[7], tX);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[7] + 1), tX);
    }
    else
    {
        // Extra elements means we can use full width-4 load/store operations
        t1  = _mm256_unpacklo_ps(v0.simdInternal_.native_register(), v2.simdInternal_.native_register());
        t2  = _mm256_unpackhi_ps(v0.simdInternal_.native_register(), v2.simdInternal_.native_register());
        t3  = _mm256_unpacklo_ps(v1.simdInternal_.native_register(), _mm256_setzero_ps());
        t4  = _mm256_unpackhi_ps(v1.simdInternal_.native_register(), _mm256_setzero_ps());
        t5  = _mm256_unpacklo_ps(t1, t3);                             // x0 y0 z0  0 | x4 y4 z4 0
        t6  = _mm256_unpackhi_ps(t1, t3);                             // x1 y1 z1  0 | x5 y5 z5 0
        t7  = _mm256_unpacklo_ps(t2, t4);                             // x2 y2 z2  0 | x6 y6 z6 0
        t8  = _mm256_unpackhi_ps(t2, t4);                             // x3 y3 z3  0 | x7 y7 z7 0

        if (align % 4 == 0)
        {
            // We can use aligned load & store
            _mm_store_ps(base + align * offset[0], _mm_add_ps(_mm_load_ps(base + align * offset[0]), _mm256_castps256_ps128(t5)));
            _mm_store_ps(base + align * offset[1], _mm_add_ps(_mm_load_ps(base + align * offset[1]), _mm256_castps256_ps128(t6)));
            _mm_store_ps(base + align * offset[2], _mm_add_ps(_mm_load_ps(base + align * offset[2]), _mm256_castps256_ps128(t7)));
            _mm_store_ps(base + align * offset[3], _mm_add_ps(_mm_load_ps(base + align * offset[3]), _mm256_castps256_ps128(t8)));
            _mm_store_ps(base + align * offset[4], _mm_add_ps(_mm_load_ps(base + align * offset[4]), _mm256_extractf128_ps(t5, 0x1)));
            _mm_store_ps(base + align * offset[5], _mm_add_ps(_mm_load_ps(base + align * offset[5]), _mm256_extractf128_ps(t6, 0x1)));
            _mm_store_ps(base + align * offset[6], _mm_add_ps(_mm_load_ps(base + align * offset[6]), _mm256_extractf128_ps(t7, 0x1)));
            _mm_store_ps(base + align * offset[7], _mm_add_ps(_mm_load_ps(base + align * offset[7]), _mm256_extractf128_ps(t8, 0x1)));
        }
        else
        {
            // alignment >=5, but not a multiple of 4
            _mm_storeu_ps(base + align * offset[0], _mm_add_ps(_mm_loadu_ps(base + align * offset[0]), _mm256_castps256_ps128(t5)));
            _mm_storeu_ps(base + align * offset[1], _mm_add_ps(_mm_loadu_ps(base + align * offset[1]), _mm256_castps256_ps128(t6)));
            _mm_storeu_ps(base + align * offset[2], _mm_add_ps(_mm_loadu_ps(base + align * offset[2]), _mm256_castps256_ps128(t7)));
            _mm_storeu_ps(base + align * offset[3], _mm_add_ps(_mm_loadu_ps(base + align * offset[3]), _mm256_castps256_ps128(t8)));
            _mm_storeu_ps(base + align * offset[4], _mm_add_ps(_mm_loadu_ps(base + align * offset[4]), _mm256_extractf128_ps(t5, 0x1)));
            _mm_storeu_ps(base + align * offset[5], _mm_add_ps(_mm_loadu_ps(base + align * offset[5]), _mm256_extractf128_ps(t6, 0x1)));
            _mm_storeu_ps(base + align * offset[6], _mm_add_ps(_mm_loadu_ps(base + align * offset[6]), _mm256_extractf128_ps(t7, 0x1)));
            _mm_storeu_ps(base + align * offset[7], _mm_add_ps(_mm_loadu_ps(base + align * offset[7]), _mm256_extractf128_ps(t8, 0x1)));
        }
    }
}

template <int align>
static inline void gmx_simdcall
transposeScatterDecrU(float *              base,
                      const std::int32_t   offset[],
                      SimdFloat            v0,
                      SimdFloat            v1,
                      SimdFloat            v2)
{
    __m256 t1, t2, t3, t4, t5, t6, t7, t8, t9, t10;
    __m128 tA, tB, tC, tD, tE, tF, tG, tH, tX;

    if (align < 4)
    {
        t5          = _mm256_unpacklo_ps(v1.simdInternal_.native_register(), v2.simdInternal_.native_register());
        t6          = _mm256_unpackhi_ps(v1.simdInternal_.native_register(), v2.simdInternal_.native_register());
        t7          = _mm256_shuffle_ps(v0.simdInternal_.native_register(), t5, _MM_SHUFFLE(1, 0, 0, 0));
        t8          = _mm256_shuffle_ps(v0.simdInternal_.native_register(), t5, _MM_SHUFFLE(3, 2, 0, 1));
        t9          = _mm256_shuffle_ps(v0.simdInternal_.native_register(), t6, _MM_SHUFFLE(1, 0, 0, 2));
        t10         = _mm256_shuffle_ps(v0.simdInternal_.native_register(), t6, _MM_SHUFFLE(3, 2, 0, 3));

        tA          = _mm256_castps256_ps128(t7);
        tB          = _mm256_castps256_ps128(t8);
        tC          = _mm256_castps256_ps128(t9);
        tD          = _mm256_castps256_ps128(t10);
        tE          = _mm256_extractf128_ps(t7, 0x1);
        tF          = _mm256_extractf128_ps(t8, 0x1);
        tG          = _mm256_extractf128_ps(t9, 0x1);
        tH          = _mm256_extractf128_ps(t10, 0x1);

        tX          = _mm_load_ss(base + align * offset[0]);
        tX          = _mm_loadh_pi(tX, reinterpret_cast< __m64 *>(base + align * offset[0] + 1));
        tX          = _mm_sub_ps(tX, tA);
        _mm_store_ss(base + align * offset[0], tX);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[0] + 1), tX);

        tX          = _mm_load_ss(base + align * offset[1]);
        tX          = _mm_loadh_pi(tX, reinterpret_cast< __m64 *>(base + align * offset[1] + 1));
        tX          = _mm_sub_ps(tX, tB);
        _mm_store_ss(base + align * offset[1], tX);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[1] + 1), tX);

        tX          = _mm_load_ss(base + align * offset[2]);
        tX          = _mm_loadh_pi(tX, reinterpret_cast< __m64 *>(base + align * offset[2] + 1));
        tX          = _mm_sub_ps(tX, tC);
        _mm_store_ss(base + align * offset[2], tX);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[2] + 1), tX);

        tX          = _mm_load_ss(base + align * offset[3]);
        tX          = _mm_loadh_pi(tX, reinterpret_cast< __m64 *>(base + align * offset[3] + 1));
        tX          = _mm_sub_ps(tX, tD);
        _mm_store_ss(base + align * offset[3], tX);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[3] + 1), tX);

        tX          = _mm_load_ss(base + align * offset[4]);
        tX          = _mm_loadh_pi(tX, reinterpret_cast< __m64 *>(base + align * offset[4] + 1));
        tX          = _mm_sub_ps(tX, tE);
        _mm_store_ss(base + align * offset[4], tX);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[4] + 1), tX);

        tX          = _mm_load_ss(base + align * offset[5]);
        tX          = _mm_loadh_pi(tX, reinterpret_cast< __m64 *>(base + align * offset[5] + 1));
        tX          = _mm_sub_ps(tX, tF);
        _mm_store_ss(base + align * offset[5], tX);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[5] + 1), tX);

        tX          = _mm_load_ss(base + align * offset[6]);
        tX          = _mm_loadh_pi(tX, reinterpret_cast< __m64 *>(base + align * offset[6] + 1));
        tX          = _mm_sub_ps(tX, tG);
        _mm_store_ss(base + align * offset[6], tX);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[6] + 1), tX);

        tX          = _mm_load_ss(base + align * offset[7]);
        tX          = _mm_loadh_pi(tX, reinterpret_cast< __m64 *>(base + align * offset[7] + 1));
        tX          = _mm_sub_ps(tX, tH);
        _mm_store_ss(base + align * offset[7], tX);
        _mm_storeh_pi(reinterpret_cast< __m64 *>(base + align * offset[7] + 1), tX);
    }
    else
    {
        // Extra elements means we can use full width-4 load/store operations
        t1  = _mm256_unpacklo_ps(v0.simdInternal_.native_register(), v2.simdInternal_.native_register());
        t2  = _mm256_unpackhi_ps(v0.simdInternal_.native_register(), v2.simdInternal_.native_register());
        t3  = _mm256_unpacklo_ps(v1.simdInternal_.native_register(), _mm256_setzero_ps());
        t4  = _mm256_unpackhi_ps(v1.simdInternal_.native_register(), _mm256_setzero_ps());
        t5  = _mm256_unpacklo_ps(t1, t3);                             // x0 y0 z0  0 | x4 y4 z4 0
        t6  = _mm256_unpackhi_ps(t1, t3);                             // x1 y1 z1  0 | x5 y5 z5 0
        t7  = _mm256_unpacklo_ps(t2, t4);                             // x2 y2 z2  0 | x6 y6 z6 0
        t8  = _mm256_unpackhi_ps(t2, t4);                             // x3 y3 z3  0 | x7 y7 z7 0

        if (align % 4 == 0)
        {
            // We can use aligned load & store
            _mm_store_ps(base + align * offset[0], _mm_sub_ps(_mm_load_ps(base + align * offset[0]), _mm256_castps256_ps128(t5)));
            _mm_store_ps(base + align * offset[1], _mm_sub_ps(_mm_load_ps(base + align * offset[1]), _mm256_castps256_ps128(t6)));
            _mm_store_ps(base + align * offset[2], _mm_sub_ps(_mm_load_ps(base + align * offset[2]), _mm256_castps256_ps128(t7)));
            _mm_store_ps(base + align * offset[3], _mm_sub_ps(_mm_load_ps(base + align * offset[3]), _mm256_castps256_ps128(t8)));
            _mm_store_ps(base + align * offset[4], _mm_sub_ps(_mm_load_ps(base + align * offset[4]), _mm256_extractf128_ps(t5, 0x1)));
            _mm_store_ps(base + align * offset[5], _mm_sub_ps(_mm_load_ps(base + align * offset[5]), _mm256_extractf128_ps(t6, 0x1)));
            _mm_store_ps(base + align * offset[6], _mm_sub_ps(_mm_load_ps(base + align * offset[6]), _mm256_extractf128_ps(t7, 0x1)));
            _mm_store_ps(base + align * offset[7], _mm_sub_ps(_mm_load_ps(base + align * offset[7]), _mm256_extractf128_ps(t8, 0x1)));
        }
        else
        {
            // alignment >=5, but not a multiple of 4
            _mm_storeu_ps(base + align * offset[0], _mm_sub_ps(_mm_loadu_ps(base + align * offset[0]), _mm256_castps256_ps128(t5)));
            _mm_storeu_ps(base + align * offset[1], _mm_sub_ps(_mm_loadu_ps(base + align * offset[1]), _mm256_castps256_ps128(t6)));
            _mm_storeu_ps(base + align * offset[2], _mm_sub_ps(_mm_loadu_ps(base + align * offset[2]), _mm256_castps256_ps128(t7)));
            _mm_storeu_ps(base + align * offset[3], _mm_sub_ps(_mm_loadu_ps(base + align * offset[3]), _mm256_castps256_ps128(t8)));
            _mm_storeu_ps(base + align * offset[4], _mm_sub_ps(_mm_loadu_ps(base + align * offset[4]), _mm256_extractf128_ps(t5, 0x1)));
            _mm_storeu_ps(base + align * offset[5], _mm_sub_ps(_mm_loadu_ps(base + align * offset[5]), _mm256_extractf128_ps(t6, 0x1)));
            _mm_storeu_ps(base + align * offset[6], _mm_sub_ps(_mm_loadu_ps(base + align * offset[6]), _mm256_extractf128_ps(t7, 0x1)));
            _mm_storeu_ps(base + align * offset[7], _mm_sub_ps(_mm_loadu_ps(base + align * offset[7]), _mm256_extractf128_ps(t8, 0x1)));
        }
    }
}

// static inline void gmx_simdca

static inline void gmx_simdcall
expandScalarsToTriplets(SimdFloat    scalar,
                        SimdFloat *  triplets0,
                        SimdFloat *  triplets1,
                        SimdFloat *  triplets2)
{
    __m256 t0 = _mm256_permute2f128_ps(scalar.simdInternal_.native_register(), scalar.simdInternal_.native_register(), 0x21);
    __m256 t1 = _mm256_permute_ps(scalar.simdInternal_.native_register(), _MM_SHUFFLE(1, 0, 0, 0));
    __m256 t2 = _mm256_permute_ps(t0, _MM_SHUFFLE(2, 2, 1, 1));
    __m256 t3 = _mm256_permute_ps(scalar.simdInternal_.native_register(), _MM_SHUFFLE(3, 3, 3, 2));
    triplets0->simdInternal_ = _mm256_blend_ps(t1, t2, 0xF0);
    triplets1->simdInternal_ = _mm256_blend_ps(t3, t1, 0xF0);
    triplets2->simdInternal_ = _mm256_blend_ps(t2, t3, 0xF0);
}

template <int align>
static inline void gmx_simdcall
gatherLoadBySimdIntTranspose(const float *  base,
                             SimdFInt32     simdoffset,
                             SimdFloat *    v0,
                             SimdFloat *    v1,
                             SimdFloat *    v2,
                             SimdFloat *    v3)
{
    alignas(GMX_SIMD_ALIGNMENT) std::int32_t    offset[GMX_SIMD_FLOAT_WIDTH];
    _mm256_store_si256( reinterpret_cast<__m256i *>(offset), simdoffset.simdInternal_.native_register());
    gatherLoadTranspose<align>(base, offset, v0, v1, v2, v3);
}

template <int align>
static inline void gmx_simdcall
gatherLoadBySimdIntTranspose(const float *   base,
                             SimdFInt32      simdoffset,
                             SimdFloat *     v0,
                             SimdFloat *     v1)
{
    alignas(GMX_SIMD_ALIGNMENT) std::int32_t    offset[GMX_SIMD_FLOAT_WIDTH];
    _mm256_store_si256( reinterpret_cast<__m256i *>(offset), simdoffset.simdInternal_.native_register());
    gatherLoadTranspose<align>(base, offset, v0, v1);
}


template <int align>
static inline void gmx_simdcall
gatherLoadUBySimdIntTranspose(const float *  base,
                              SimdFInt32     simdoffset,
                              SimdFloat *    v0,
                              SimdFloat *    v1)
{
    __m128 t1, t2, t3, t4, t5, t6, t7, t8;
    __m256 tA, tB, tC, tD;

    alignas(GMX_SIMD_ALIGNMENT) std::int32_t     offset[GMX_SIMD_FLOAT_WIDTH];
    _mm256_store_si256( reinterpret_cast<__m256i *>(offset), simdoffset.simdInternal_.native_register());

    t1  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>( base + align * offset[0] ) );
    t2  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>( base + align * offset[1] ) );
    t3  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>( base + align * offset[2] ) );
    t4  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>( base + align * offset[3] ) );
    t5  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>( base + align * offset[4] ) );
    t6  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>( base + align * offset[5] ) );
    t7  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>( base + align * offset[6] ) );
    t8  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>( base + align * offset[7] ) );

    tA  = _mm256_insertf128_ps(_mm256_castps128_ps256(t1), t5, 0x1);
    tB  = _mm256_insertf128_ps(_mm256_castps128_ps256(t2), t6, 0x1);
    tC  = _mm256_insertf128_ps(_mm256_castps128_ps256(t3), t7, 0x1);
    tD  = _mm256_insertf128_ps(_mm256_castps128_ps256(t4), t8, 0x1);

    tA                = _mm256_unpacklo_ps(tA, tC);
    tB                = _mm256_unpacklo_ps(tB, tD);
    v0->simdInternal_ = _mm256_unpacklo_ps(tA, tB);
    v1->simdInternal_ = _mm256_unpackhi_ps(tA, tB);
}

static inline float gmx_simdcall
reduceIncr4ReturnSum(float *    m,
                     SimdFloat  v0,
                     SimdFloat  v1,
                     SimdFloat  v2,
                     SimdFloat  v3)
{
    __m128 t0, t2;

    assert(std::size_t(m) % 16 == 0);

    v0.simdInternal_ = _mm256_hadd_ps(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
    v2.simdInternal_ = _mm256_hadd_ps(v2.simdInternal_.native_register(), v3.simdInternal_.native_register());
    v0.simdInternal_ = _mm256_hadd_ps(v0.simdInternal_.native_register(), v2.simdInternal_.native_register());
    t0               = _mm_add_ps(_mm256_castps256_ps128(v0.simdInternal_.native_register()), _mm256_extractf128_ps(v0.simdInternal_.native_register(), 0x1));

    t2 = _mm_add_ps(t0, _mm_load_ps(m));
    _mm_store_ps(m, t2);

    t0 = _mm_add_ps(t0, _mm_permute_ps(t0, _MM_SHUFFLE(1, 0, 3, 2)));
    t0 = _mm_add_ss(t0, _mm_permute_ps(t0, _MM_SHUFFLE(0, 3, 2, 1)));
    return *reinterpret_cast<float *>(&t0);
}


/*************************************
 * Half-simd-width utility functions *
 *************************************/
static inline SimdFloat gmx_simdcall
loadDualHsimd(const float * m0,
              const float * m1)
{
    assert(std::size_t(m0) % 16 == 0);
    assert(std::size_t(m1) % 16 == 0);

    return {
               _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(m0)), _mm_load_ps(m1), 0x1)
    };
}

static inline SimdFloat gmx_simdcall
loadDuplicateHsimd(const float * m)
{
    assert(std::size_t(m) % 16 == 0);

    return {
               _mm256_broadcast_ps(reinterpret_cast<const __m128 *>(m))
    };
}

static inline SimdFloat gmx_simdcall
loadU1DualHsimd(const float * m)
{
    __m128 t0, t1;
    t0 = _mm_broadcast_ss(m);
    t1 = _mm_broadcast_ss(m+1);
    return {
               _mm256_insertf128_ps(_mm256_castps128_ps256(t0), t1, 0x1)
    };
}


static inline void gmx_simdcall
storeDualHsimd(float *     m0,
               float *     m1,
               SimdFloat   a)
{
    assert(std::size_t(m0) % 16 == 0);
    assert(std::size_t(m1) % 16 == 0);
    _mm_store_ps(m0, _mm256_castps256_ps128(a.simdInternal_.native_register()));
    _mm_store_ps(m1, _mm256_extractf128_ps(a.simdInternal_.native_register(), 0x1));
}

static inline void gmx_simdcall
incrDualHsimd(float *     m0,
              float *     m1,
              SimdFloat   a)
{
    assert(std::size_t(m0) % 16 == 0);
    assert(std::size_t(m1) % 16 == 0);
    _mm_store_ps(m0, _mm_add_ps(_mm256_castps256_ps128(a.simdInternal_.native_register()), _mm_load_ps(m0)));
    _mm_store_ps(m1, _mm_add_ps(_mm256_extractf128_ps(a.simdInternal_.native_register(), 0x1), _mm_load_ps(m1)));
}

static inline void gmx_simdcall
decrHsimd(float *    m,
          SimdFloat  a)
{
    assert(std::size_t(m) % 16 == 0);
    __m128 asum = _mm_add_ps(_mm256_castps256_ps128(a.simdInternal_.native_register()), _mm256_extractf128_ps(a.simdInternal_.native_register(), 0x1));
    _mm_store_ps(m, _mm_sub_ps(_mm_load_ps(m), asum));
}


template <int align>
static inline void gmx_simdcall
gatherLoadTransposeHsimd(const float *        base0,
                         const float *        base1,
                         const std::int32_t   offset[],
                         SimdFloat *          v0,
                         SimdFloat *          v1)
{
    __m128 t0, t1, t2, t3, t4, t5, t6, t7;
    __m256 tA, tB, tC, tD;

    assert(std::size_t(offset) % 16 == 0);
    assert(std::size_t(base0) % 8 == 0);
    assert(std::size_t(base1) % 8 == 0);
    assert(align % 2 == 0);

    t0  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>(base0 + align * offset[0]));
    t1  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>(base0 + align * offset[1]));
    t2  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>(base0 + align * offset[2]));
    t3  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>(base0 + align * offset[3]));
    t4  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>(base1 + align * offset[0]));
    t5  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>(base1 + align * offset[1]));
    t6  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>(base1 + align * offset[2]));
    t7  = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>(base1 + align * offset[3]));

    tA  = _mm256_insertf128_ps(_mm256_castps128_ps256(t0), t4, 0x1);
    tB  = _mm256_insertf128_ps(_mm256_castps128_ps256(t1), t5, 0x1);
    tC  = _mm256_insertf128_ps(_mm256_castps128_ps256(t2), t6, 0x1);
    tD  = _mm256_insertf128_ps(_mm256_castps128_ps256(t3), t7, 0x1);

    tA                = _mm256_unpacklo_ps(tA, tC);
    tB                = _mm256_unpacklo_ps(tB, tD);
    v0->simdInternal_ = _mm256_unpacklo_ps(tA, tB);
    v1->simdInternal_ = _mm256_unpackhi_ps(tA, tB);
}


static inline float gmx_simdcall
reduceIncr4ReturnSumHsimd(float *     m,
                          SimdFloat   v0,
                          SimdFloat   v1)
{
    __m128 t0, t1;

    v0.simdInternal_ = _mm256_hadd_ps(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
    t0               = _mm256_extractf128_ps(v0.simdInternal_.native_register(), 0x1);
    t0               = _mm_hadd_ps(_mm256_castps256_ps128(v0.simdInternal_.native_register()), t0);
    t0               = _mm_permute_ps(t0, _MM_SHUFFLE(3, 1, 2, 0));

    assert(std::size_t(m) % 16 == 0);

    t1   = _mm_add_ps(t0, _mm_load_ps(m));
    _mm_store_ps(m, t1);

    t0 = _mm_add_ps(t0, _mm_permute_ps(t0, _MM_SHUFFLE(1, 0, 3, 2)));
    t0 = _mm_add_ss(t0, _mm_permute_ps(t0, _MM_SHUFFLE(0, 3, 2, 1)));
    return *reinterpret_cast<float *>(&t0);
}

static inline SimdFloat gmx_simdcall
loadU4NOffset(const float *m, int offset)
{
    return {
               _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(m)), _mm_loadu_ps(m+offset), 0x1)
    };
}


// This version is marginally slower than the AVX 4-wide component load
// version on Intel Skylake. On older Intel architectures this version
// is significantly slower.
template <int align>
static inline void gmx_simdcall
gatherLoadUTransposeSafe(const float *        base,
                         const std::int32_t   offset[],
                         SimdFloat *          v0,
                         SimdFloat *          v1,
                         SimdFloat *          v2)
{
    assert(std::size_t(offset) % 32 == 0);

    const SimdFInt32 alignSimd = SimdFInt32(align);

    SimdFInt32       vindex = simdLoad(offset, SimdFInt32Tag());
    vindex = vindex*alignSimd;

    *v0 = _mm256_i32gather_ps(base + 0, vindex.simdInternal_.native_register(), sizeof(float));
    *v1 = _mm256_i32gather_ps(base + 1, vindex.simdInternal_.native_register(), sizeof(float));
    *v2 = _mm256_i32gather_ps(base + 2, vindex.simdInternal_.native_register(), sizeof(float));
}

#elif (defined(NSIMD_AVX512_KNL) || defined(NSIMD_AVX512_SKYLAKE))

static const int c_simdBestPairAlignmentFloat = 2;

namespace
{
// Multiply function optimized for powers of 2, for which it is done by
// shifting. Currently up to 8 is accelerated. Could be accelerated for any
// number with a constexpr log2 function.
template<int n>
SimdFInt32 fastMultiply(SimdFInt32 x)
{
    if (n == 2)
    {
        nsimd::shl(x.simdInternal_, 1);
	    // return _mm512_slli_epi32(x.simdInternal_.native_register(), 1);
    }
    else if (n == 4)
    {
	    nsimd::shl(x.simdInternal_, 2);
	    // return _mm512_slli_epi32(x.simdInternal_.native_register(), 2);
    }
    else if (n == 8)
    {
        nsimd::shl(x.simdInternal_, 3);
	    // return _mm512_slli_epi32(x.simdInternal_.native_register(), 3);
    }
    return x * n;
}

template<int align>
static inline void gmx_simdcall
gatherLoadBySimdIntTranspose(const float *, SimdFInt32)
{
    //Nothing to do. Termination of recursion.
}
}

template <int align, typename ... Targs>
static inline void gmx_simdcall
gatherLoadBySimdIntTranspose(const float *base, SimdFInt32 offset, SimdFloat *v, Targs... Fargs)
{
    // For align 1 or 2: No multiplication of offset is needed
    if (align > 2)
    {
        offset = fastMultiply<align>(offset);
    }
    // For align 2: Scale of 2*sizeof(float) is used (maximum supported scale)
    constexpr int align_ = (align > 2) ? 1 : align;
    v->simdInternal_ = _mm512_i32gather_ps(offset.simdInternal_.native_register(), base, sizeof(float)*align_);
    // Gather remaining elements. Avoid extra multiplication (new align is 1 or 2).
    gatherLoadBySimdIntTranspose<align_>(base+1, offset, Fargs ...);
}

template <int align, typename ... Targs>
static inline void gmx_simdcall
gatherLoadUBySimdIntTranspose(const float *base, SimdFInt32 offset, SimdFloat *v, Targs... Fargs)
{
    gatherLoadBySimdIntTranspose<align>(base, offset, v, Fargs ...);
}

template <int align, typename ... Targs>
static inline void gmx_simdcall
gatherLoadTranspose(const float *base, const std::int32_t offset[], SimdFloat *v, Targs... Fargs)
{
    gatherLoadBySimdIntTranspose<align>(base, simdLoad(offset, SimdFInt32Tag()), v, Fargs ...);
}

template <int align, typename ... Targs>
static inline void gmx_simdcall
gatherLoadUTranspose(const float *base, const std::int32_t offset[], SimdFloat *v, Targs... Fargs)
{
    gatherLoadBySimdIntTranspose<align>(base, simdLoad(offset, SimdFInt32Tag()), v, Fargs ...);
}

template <int align>
static inline void gmx_simdcall
transposeScatterStoreU(float *              base,
                       const std::int32_t   offset[],
                       SimdFloat            v0,
                       SimdFloat            v1,
                       SimdFloat            v2)
{
    SimdFInt32 simdoffset = simdLoad(offset, SimdFInt32Tag());
    if (align > 2)
    {
        simdoffset = fastMultiply<align>(simdoffset);
    }
    constexpr size_t scale = (align > 2) ? sizeof(float) : sizeof(float) * align;

    _mm512_i32scatter_ps(base,       simdoffset.simdInternal_.native_register(), v0.simdInternal_.native_register(), scale);
    _mm512_i32scatter_ps(&(base[1]), simdoffset.simdInternal_.native_register(), v1.simdInternal_.native_register(), scale);
    _mm512_i32scatter_ps(&(base[2]), simdoffset.simdInternal_.native_register(), v2.simdInternal_.native_register(), scale);
}

template <int align>
static inline void gmx_simdcall
transposeScatterIncrU(float *              base,
                      const std::int32_t   offset[],
                      SimdFloat            v0,
                      SimdFloat            v1,
                      SimdFloat            v2)
{
    __m512 t[4], t5, t6, t7, t8;
    int    i;
    alignas(GMX_SIMD_ALIGNMENT) std::int32_t    o[16];
    store(o, fastMultiply<align>(simdLoad(offset, SimdFInt32Tag())));
    if (align < 4)
    {
        t5   = _mm512_unpacklo_ps(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
        t6   = _mm512_unpackhi_ps(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
        t[0] = _mm512_shuffle_ps(t5, v2.simdInternal_.native_register(), _MM_SHUFFLE(0, 0, 1, 0));
        t[1] = _mm512_shuffle_ps(t5, v2.simdInternal_.native_register(), _MM_SHUFFLE(1, 1, 3, 2));
        t[2] = _mm512_shuffle_ps(t6, v2.simdInternal_.native_register(), _MM_SHUFFLE(2, 2, 1, 0));
        t[3] = _mm512_shuffle_ps(t6, v2.simdInternal_.native_register(), _MM_SHUFFLE(3, 3, 3, 2));
        for (i = 0; i < 4; i++)
        {
            _mm512_mask_storeu_ps(base + o[i], avx512Int2Mask(7), _mm512_castps128_ps512(
                                          _mm_add_ps(_mm_loadu_ps(base + o[i]), _mm512_castps512_ps128(t[i]))));
            _mm512_mask_storeu_ps(base + o[ 4 + i], avx512Int2Mask(7), _mm512_castps128_ps512(
                                          _mm_add_ps(_mm_loadu_ps(base + o[ 4 + i]), _mm512_extractf32x4_ps(t[i], 1))));
            _mm512_mask_storeu_ps(base + o[ 8 + i], avx512Int2Mask(7), _mm512_castps128_ps512(
                                          _mm_add_ps(_mm_loadu_ps(base + o[ 8 + i]), _mm512_extractf32x4_ps(t[i], 2))));
            _mm512_mask_storeu_ps(base + o[12 + i], avx512Int2Mask(7), _mm512_castps128_ps512(
                                          _mm_add_ps(_mm_loadu_ps(base + o[12 + i]), _mm512_extractf32x4_ps(t[i], 3))));
        }
    }
    else
    {
        //One could use shuffle here too if it is OK to overwrite the padded elements for alignment
        t5    = _mm512_unpacklo_ps(v0.simdInternal_.native_register(), v2.simdInternal_.native_register());
        t6    = _mm512_unpackhi_ps(v0.simdInternal_.native_register(), v2.simdInternal_.native_register());
        t7    = _mm512_unpacklo_ps(v1.simdInternal_.native_register(), _mm512_setzero_ps());
        t8    = _mm512_unpackhi_ps(v1.simdInternal_.native_register(), _mm512_setzero_ps());
        t[0]  = _mm512_unpacklo_ps(t5, t7);                             // x0 y0 z0  0 | x4 y4 z4 0
        t[1]  = _mm512_unpackhi_ps(t5, t7);                             // x1 y1 z1  0 | x5 y5 z5 0
        t[2]  = _mm512_unpacklo_ps(t6, t8);                             // x2 y2 z2  0 | x6 y6 z6 0
        t[3]  = _mm512_unpackhi_ps(t6, t8);                             // x3 y3 z3  0 | x7 y7 z7 0
        if (align % 4 == 0)
        {
            for (i = 0; i < 4; i++)
            {
                _mm_store_ps(base + o[i], _mm_add_ps(_mm_load_ps(base + o[i]), _mm512_castps512_ps128(t[i])));
                _mm_store_ps(base + o[ 4 + i],
                             _mm_add_ps(_mm_load_ps(base + o[ 4 + i]), _mm512_extractf32x4_ps(t[i], 1)));
                _mm_store_ps(base + o[ 8 + i],
                             _mm_add_ps(_mm_load_ps(base + o[ 8 + i]), _mm512_extractf32x4_ps(t[i], 2)));
                _mm_store_ps(base + o[12 + i],
                             _mm_add_ps(_mm_load_ps(base + o[12 + i]), _mm512_extractf32x4_ps(t[i], 3)));
            }
        }
        else
        {
            for (i = 0; i < 4; i++)
            {
                _mm_storeu_ps(base + o[i], _mm_add_ps(_mm_loadu_ps(base + o[i]), _mm512_castps512_ps128(t[i])));
                _mm_storeu_ps(base + o[ 4 + i],
                              _mm_add_ps(_mm_loadu_ps(base + o[ 4 + i]), _mm512_extractf32x4_ps(t[i], 1)));
                _mm_storeu_ps(base + o[ 8 + i],
                              _mm_add_ps(_mm_loadu_ps(base + o[ 8 + i]), _mm512_extractf32x4_ps(t[i], 2)));
                _mm_storeu_ps(base + o[12 + i],
                              _mm_add_ps(_mm_loadu_ps(base + o[12 + i]), _mm512_extractf32x4_ps(t[i], 3)));
            }
        }
    }
}

template <int align>
static inline void gmx_simdcall
transposeScatterDecrU(float *              base,
                      const std::int32_t   offset[],
                      SimdFloat            v0,
                      SimdFloat            v1,
                      SimdFloat            v2)
{
    __m512 t[4], t5, t6, t7, t8;
    int    i;
    alignas(GMX_SIMD_ALIGNMENT) std::int32_t    o[16];
    store(o, fastMultiply<align>(simdLoad(offset, SimdFInt32Tag())));
    if (align < 4)
    {
        t5   = _mm512_unpacklo_ps(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
        t6   = _mm512_unpackhi_ps(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());
        t[0] = _mm512_shuffle_ps(t5, v2.simdInternal_.native_register(), _MM_SHUFFLE(0, 0, 1, 0));
        t[1] = _mm512_shuffle_ps(t5, v2.simdInternal_.native_register(), _MM_SHUFFLE(1, 1, 3, 2));
        t[2] = _mm512_shuffle_ps(t6, v2.simdInternal_.native_register(), _MM_SHUFFLE(2, 2, 1, 0));
        t[3] = _mm512_shuffle_ps(t6, v2.simdInternal_.native_register(), _MM_SHUFFLE(3, 3, 3, 2));
        for (i = 0; i < 4; i++)
        {
            _mm512_mask_storeu_ps(base + o[i], avx512Int2Mask(7), _mm512_castps128_ps512(
                                          _mm_sub_ps(_mm_loadu_ps(base + o[i]), _mm512_castps512_ps128(t[i]))));
            _mm512_mask_storeu_ps(base + o[ 4 + i], avx512Int2Mask(7), _mm512_castps128_ps512(
                                          _mm_sub_ps(_mm_loadu_ps(base + o[ 4 + i]), _mm512_extractf32x4_ps(t[i], 1))));
            _mm512_mask_storeu_ps(base + o[ 8 + i], avx512Int2Mask(7), _mm512_castps128_ps512(
                                          _mm_sub_ps(_mm_loadu_ps(base + o[ 8 + i]), _mm512_extractf32x4_ps(t[i], 2))));
            _mm512_mask_storeu_ps(base + o[12 + i], avx512Int2Mask(7), _mm512_castps128_ps512(
                                          _mm_sub_ps(_mm_loadu_ps(base + o[12 + i]), _mm512_extractf32x4_ps(t[i], 3))));
        }
    }
    else
    {
        //One could use shuffle here too if it is OK to overwrite the padded elements for alignment
        t5    = _mm512_unpacklo_ps(v0.simdInternal_.native_register(), v2.simdInternal_.native_register());
        t6    = _mm512_unpackhi_ps(v0.simdInternal_.native_register(), v2.simdInternal_.native_register());
        t7    = _mm512_unpacklo_ps(v1.simdInternal_.native_register(), _mm512_setzero_ps());
        t8    = _mm512_unpackhi_ps(v1.simdInternal_.native_register(), _mm512_setzero_ps());
        t[0]  = _mm512_unpacklo_ps(t5, t7);                             // x0 y0 z0  0 | x4 y4 z4 0
        t[1]  = _mm512_unpackhi_ps(t5, t7);                             // x1 y1 z1  0 | x5 y5 z5 0
        t[2]  = _mm512_unpacklo_ps(t6, t8);                             // x2 y2 z2  0 | x6 y6 z6 0
        t[3]  = _mm512_unpackhi_ps(t6, t8);                             // x3 y3 z3  0 | x7 y7 z7 0
        if (align % 4 == 0)
        {
            for (i = 0; i < 4; i++)
            {
                _mm_store_ps(base + o[i], _mm_sub_ps(_mm_load_ps(base + o[i]), _mm512_castps512_ps128(t[i])));
                _mm_store_ps(base + o[ 4 + i],
                             _mm_sub_ps(_mm_load_ps(base + o[ 4 + i]), _mm512_extractf32x4_ps(t[i], 1)));
                _mm_store_ps(base + o[ 8 + i],
                             _mm_sub_ps(_mm_load_ps(base + o[ 8 + i]), _mm512_extractf32x4_ps(t[i], 2)));
                _mm_store_ps(base + o[12 + i],
                             _mm_sub_ps(_mm_load_ps(base + o[12 + i]), _mm512_extractf32x4_ps(t[i], 3)));
            }
        }
        else
        {
            for (i = 0; i < 4; i++)
            {
                _mm_storeu_ps(base + o[i], _mm_sub_ps(_mm_loadu_ps(base + o[i]), _mm512_castps512_ps128(t[i])));
                _mm_storeu_ps(base + o[ 4 + i],
                              _mm_sub_ps(_mm_loadu_ps(base + o[ 4 + i]), _mm512_extractf32x4_ps(t[i], 1)));
                _mm_storeu_ps(base + o[ 8 + i],
                              _mm_sub_ps(_mm_loadu_ps(base + o[ 8 + i]), _mm512_extractf32x4_ps(t[i], 2)));
                _mm_storeu_ps(base + o[12 + i],
                              _mm_sub_ps(_mm_loadu_ps(base + o[12 + i]), _mm512_extractf32x4_ps(t[i], 3)));
            }
        }
    }
}

static inline void gmx_simdcall
expandScalarsToTriplets(SimdFloat    scalar,
                        SimdFloat *  triplets0,
                        SimdFloat *  triplets1,
                        SimdFloat *  triplets2)
{
    triplets0->simdInternal_ = _mm512_permutexvar_ps(_mm512_set_epi32(5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0),
                                                     scalar.simdInternal_.native_register());
    triplets1->simdInternal_ = _mm512_permutexvar_ps(_mm512_set_epi32(10, 10, 9, 9, 9, 8, 8, 8, 7, 7, 7, 6, 6, 6, 5, 5),
                                                     scalar.simdInternal_.native_register());
    triplets2->simdInternal_ = _mm512_permutexvar_ps(_mm512_set_epi32(15, 15, 15, 14, 14, 14, 13, 13, 13, 12, 12, 12, 11, 11, 11, 10),
                                                     scalar.simdInternal_.native_register());
}


static inline float gmx_simdcall
reduceIncr4ReturnSum(float *    m,
                     SimdFloat  v0,
                     SimdFloat  v1,
                     SimdFloat  v2,
                     SimdFloat  v3)
{
    __m512 t0, t1, t2;
    __m128 t3, t4;

    assert(std::size_t(m) % 16 == 0);

    t0 = _mm512_add_ps(v0.simdInternal_.native_register(), _mm512_permute_ps(v0.simdInternal_.native_register(), 0x4E));
    t0 = _mm512_mask_add_ps(t0, avx512Int2Mask(0xCCCC), v2.simdInternal_.native_register(), _mm512_permute_ps(v2.simdInternal_.native_register(), 0x4E));
    t1 = _mm512_add_ps(v1.simdInternal_.native_register(), _mm512_permute_ps(v1.simdInternal_.native_register(), 0x4E));
    t1 = _mm512_mask_add_ps(t1, avx512Int2Mask(0xCCCC), v3.simdInternal_.native_register(), _mm512_permute_ps(v3.simdInternal_.native_register(), 0x4E));
    t2 = _mm512_add_ps(t0, _mm512_permute_ps(t0, 0xB1));
    t2 = _mm512_mask_add_ps(t2, avx512Int2Mask(0xAAAA), t1, _mm512_permute_ps(t1, 0xB1));

    t2 = _mm512_add_ps(t2, _mm512_shuffle_f32x4(t2, t2, 0x4E));
    t2 = _mm512_add_ps(t2, _mm512_shuffle_f32x4(t2, t2, 0xB1));

    t3 = _mm512_castps512_ps128(t2);
    t4 = _mm_load_ps(m);
    t4 = _mm_add_ps(t4, t3);
    _mm_store_ps(m, t4);

    t3 = _mm_add_ps(t3, _mm_permute_ps(t3, 0x4E));
    t3 = _mm_add_ps(t3, _mm_permute_ps(t3, 0xB1));

    return _mm_cvtss_f32(t3);

}

static inline SimdFloat gmx_simdcall
loadDualHsimd(const float * m0,
              const float * m1)
{
    assert(std::size_t(m0) % 32 == 0);
    assert(std::size_t(m1) % 32 == 0);

    return {
               _mm512_castpd_ps(_mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_load_pd(reinterpret_cast<const double*>(m0))),
                                                   _mm256_load_pd(reinterpret_cast<const double*>(m1)), 1))
    };
}

static inline SimdFloat gmx_simdcall
loadDuplicateHsimd(const float * m)
{
    assert(std::size_t(m) % 32 == 0);
    return {
               _mm512_castpd_ps(_mm512_broadcast_f64x4(_mm256_load_pd(reinterpret_cast<const double*>(m))))
    };
}

static inline SimdFloat gmx_simdcall
loadU1DualHsimd(const float * m)
{
    return {
               _mm512_shuffle_f32x4(_mm512_broadcastss_ps(_mm_load_ss(m)),
                                    _mm512_broadcastss_ps(_mm_load_ss(m+1)), 0x44)
    };
}


static inline void gmx_simdcall
storeDualHsimd(float *     m0,
               float *     m1,
               SimdFloat   a)
{
    assert(std::size_t(m0) % 32 == 0);
    assert(std::size_t(m1) % 32 == 0);

    _mm256_store_ps(m0, _mm512_castps512_ps256(a.simdInternal_.native_register()));
    _mm256_store_pd(reinterpret_cast<double*>(m1), _mm512_extractf64x4_pd(_mm512_castps_pd(a.simdInternal_.native_register()), 1));
}

static inline void gmx_simdcall
incrDualHsimd(float *     m0,
              float *     m1,
              SimdFloat   a)
{
    assert(std::size_t(m0) % 32 == 0);
    assert(std::size_t(m1) % 32 == 0);

    __m256 x;

    // Lower half
    x = _mm256_load_ps(m0);
    x = _mm256_add_ps(x, _mm512_castps512_ps256(a.simdInternal_.native_register()));
    _mm256_store_ps(m0, x);

    // Upper half
    x = _mm256_load_ps(m1);
    x = _mm256_add_ps(x, _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(a.simdInternal_.native_register()), 1)));
    _mm256_store_ps(m1, x);
}

static inline void gmx_simdcall
decrHsimd(float *    m,
          SimdFloat  a)
{
    __m256 t;

    assert(std::size_t(m) % 32 == 0);

    a.simdInternal_ = _mm512_add_ps(a.simdInternal_.native_register(), _mm512_shuffle_f32x4(a.simdInternal_.native_register(), a.simdInternal_.native_register(), 0xEE));
    t               = _mm256_load_ps(m);
    t               = _mm256_sub_ps(t, _mm512_castps512_ps256(a.simdInternal_.native_register()));
    _mm256_store_ps(m, t);
}


template <int align>
static inline void gmx_simdcall
gatherLoadTransposeHsimd(const float *        base0,
                         const float *        base1,
                         const std::int32_t   offset[],
                         SimdFloat *          v0,
                         SimdFloat *          v1)
{
    __m256i idx;
    __m512  tmp1, tmp2;

    assert(std::size_t(offset) % 32 == 0);
    assert(std::size_t(base0) % 8 == 0);
    assert(std::size_t(base1) % 8 == 0);

    idx = _mm256_load_si256(reinterpret_cast<const __m256i*>(offset));

    static_assert(align == 2 || align == 4, "If more are needed use fastMultiply");
    if (align == 4)
    {
        idx = _mm256_slli_epi32(idx, 1);
    }

    tmp1 = _mm512_castpd_ps(_mm512_i32gather_pd(idx, reinterpret_cast<const double *>(base0), sizeof(double)));
    tmp2 = _mm512_castpd_ps(_mm512_i32gather_pd(idx, reinterpret_cast<const double *>(base1), sizeof(double)));

    v0->simdInternal_ = _mm512_mask_moveldup_ps(tmp1, 0xAAAA, tmp2);
    v1->simdInternal_ = _mm512_mask_movehdup_ps(tmp2, 0x5555, tmp1);

    v0->simdInternal_ = _mm512_permutexvar_ps(_mm512_set_epi32(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0), v0->simdInternal_.native_register());
    v1->simdInternal_ = _mm512_permutexvar_ps(_mm512_set_epi32(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0), v1->simdInternal_.native_register());
}

static inline float gmx_simdcall
reduceIncr4ReturnSumHsimd(float *     m,
                          SimdFloat   v0,
                          SimdFloat   v1)
{
    __m512 t0, t1;
    __m128 t2, t3;

    assert(std::size_t(m) % 16 == 0);

    t0 = _mm512_shuffle_f32x4(v0.simdInternal_.native_register(), v1.simdInternal_.native_register(), 0x88);
    t1 = _mm512_shuffle_f32x4(v0.simdInternal_.native_register(), v1.simdInternal_.native_register(), 0xDD);
    t0 = _mm512_add_ps(t0, t1);
    t0 = _mm512_add_ps(t0, _mm512_permute_ps(t0, 0x4E));
    t0 = _mm512_add_ps(t0, _mm512_permute_ps(t0, 0xB1));
    t0 = _mm512_maskz_compress_ps(avx512Int2Mask(0x1111), t0);

    t3 = _mm512_castps512_ps128(t0);
    t2 = _mm_load_ps(m);
    t2 = _mm_add_ps(t2, t3);
    _mm_store_ps(m, t2);

    t3 = _mm_add_ps(t3, _mm_permute_ps(t3, 0x4E));
    t3 = _mm_add_ps(t3, _mm_permute_ps(t3, 0xB1));

    return _mm_cvtss_f32(t3);
}

static inline SimdFloat gmx_simdcall
loadUNDuplicate4(const float* f)
{
    return {
               _mm512_permute_ps(_mm512_maskz_expandloadu_ps(0x1111, f), 0)
    };
}

static inline SimdFloat gmx_simdcall
load4DuplicateN(const float* f)
{
    return {
               _mm512_broadcast_f32x4(_mm_load_ps(f))
    };
}

static inline SimdFloat gmx_simdcall
loadU4NOffset(const float* f, int offset)
{
    const __m256i idx = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
    const __m256i gdx = _mm256_add_epi32(_mm256_setr_epi32(0, 2, 0, 2, 0, 2, 0, 2),
                                         _mm256_mullo_epi32(idx, _mm256_set1_epi32(offset)));
    return {
               _mm512_castpd_ps(_mm512_i32gather_pd(gdx, reinterpret_cast<const double*>(f), sizeof(float)))
    };
}

#elif (defined(NSIMD_AARCH64) || defined(NSIMD_ARM_NEON))


template <int align>
static inline void gmx_simdcall
gatherLoadTranspose(const float *        base,
                    const std::int32_t   offset[],
                    SimdFloat *          v0,
                    SimdFloat *          v1,
                    SimdFloat *          v2,
                    SimdFloat *          v3)
{
    assert(std::size_t(offset) % 16 == 0);
    assert(std::size_t(base) % 16 == 0);
    assert(align % 4 == 0);

    // Unfortunately we cannot use the beautiful Neon structured load
    // instructions since the data comes from four different memory locations.
    float32x4x2_t  t0 = vuzpq_f32(vld1q_f32( base + align * offset[0] ), vld1q_f32( base + align * offset[2] ));
    float32x4x2_t  t1 = vuzpq_f32(vld1q_f32( base + align * offset[1] ), vld1q_f32( base + align * offset[3] ));
    float32x4x2_t  t2 = vtrnq_f32(t0.val[0], t1.val[0]);
    float32x4x2_t  t3 = vtrnq_f32(t0.val[1], t1.val[1]);
    v0->simdInternal_ = t2.val[0];
    v1->simdInternal_ = t3.val[0];
    v2->simdInternal_ = t2.val[1];
    v3->simdInternal_ = t3.val[1];
}

template <int align>
static inline void gmx_simdcall
gatherLoadTranspose(const float *        base,
                    const std::int32_t   offset[],
                    SimdFloat *          v0,
                    SimdFloat *          v1)
{
    assert(std::size_t(offset) % 16 == 0);
    assert(std::size_t(base) % 8 == 0);
    assert(align % 2 == 0);

    v0->simdInternal_  = vcombine_f32(vld1_f32( base + align * offset[0] ),
                                      vld1_f32( base + align * offset[2] ));
    v1->simdInternal_  = vcombine_f32(vld1_f32( base + align * offset[1] ),
                                      vld1_f32( base + align * offset[3] ));

    float32x4x2_t tmp  = vtrnq_f32(v0->simdInternal_.native_register(), v1->simdInternal_.native_register());

    v0->simdInternal_  = tmp.val[0];
    v1->simdInternal_  = tmp.val[1];
}

static const int c_simdBestPairAlignmentFloat = 2;

template <int align>
static inline void gmx_simdcall
gatherLoadUTranspose(const float *        base,
                     const std::int32_t   offset[],
                     SimdFloat *          v0,
                     SimdFloat *          v1,
                     SimdFloat *          v2)
{
    assert(std::size_t(offset) % 16 == 0);

    float32x4x2_t  t0 = vuzpq_f32(vld1q_f32( base + align * offset[0] ), vld1q_f32( base + align * offset[2] ));
    float32x4x2_t  t1 = vuzpq_f32(vld1q_f32( base + align * offset[1] ), vld1q_f32( base + align * offset[3] ));
    float32x4x2_t  t2 = vtrnq_f32(t0.val[0], t1.val[0]);
    float32x4x2_t  t3 = vtrnq_f32(t0.val[1], t1.val[1]);
    v0->simdInternal_ = t2.val[0];
    v1->simdInternal_ = t3.val[0];
    v2->simdInternal_ = t2.val[1];
}


template <int align>
static inline void gmx_simdcall
transposeScatterStoreU(float *              base,
                       const std::int32_t   offset[],
                       SimdFloat            v0,
                       SimdFloat            v1,
                       SimdFloat            v2)
{
    assert(std::size_t(offset) % 16 == 0);

    float32x4x2_t tmp = vtrnq_f32(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());

    vst1_f32( base + align * offset[0], vget_low_f32(tmp.val[0]) );
    vst1_f32( base + align * offset[1], vget_low_f32(tmp.val[1]) );
    vst1_f32( base + align * offset[2], vget_high_f32(tmp.val[0]) );
    vst1_f32( base + align * offset[3], vget_high_f32(tmp.val[1]) );

    vst1q_lane_f32( base + align * offset[0] + 2, v2.simdInternal_.native_register(), 0);
    vst1q_lane_f32( base + align * offset[1] + 2, v2.simdInternal_.native_register(), 1);
    vst1q_lane_f32( base + align * offset[2] + 2, v2.simdInternal_.native_register(), 2);
    vst1q_lane_f32( base + align * offset[3] + 2, v2.simdInternal_.native_register(), 3);
}


template <int align>
static inline void gmx_simdcall
transposeScatterIncrU(float *              base,
                      const std::int32_t   offset[],
                      SimdFloat            v0,
                      SimdFloat            v1,
                      SimdFloat            v2)
{
    assert(std::size_t(offset) % 16 == 0);

    if (align < 4)
    {
        float32x2_t   t0, t1, t2, t3;
        float32x4x2_t tmp = vtrnq_f32(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());

        t0 = vget_low_f32(tmp.val[0]);
        t1 = vget_low_f32(tmp.val[1]);
        t2 = vget_high_f32(tmp.val[0]);
        t3 = vget_high_f32(tmp.val[1]);

        t0 = vadd_f32(t0, vld1_f32(base + align * offset[0]));
        vst1_f32(base + align * offset[0], t0);
        base[ align * offset[0] + 2] += vgetq_lane_f32(v2.simdInternal_.native_register(), 0);

        t1 = vadd_f32(t1, vld1_f32(base + align * offset[1]));
        vst1_f32(base + align * offset[1], t1);
        base[ align * offset[1] + 2] += vgetq_lane_f32(v2.simdInternal_.native_register(), 1);

        t2 = vadd_f32(t2, vld1_f32(base + align * offset[2]));
        vst1_f32(base + align * offset[2], t2);
        base[ align * offset[2] + 2] += vgetq_lane_f32(v2.simdInternal_.native_register(), 2);

        t3 = vadd_f32(t3, vld1_f32(base + align * offset[3]));
        vst1_f32(base + align * offset[3], t3);
        base[ align * offset[3] + 2] += vgetq_lane_f32(v2.simdInternal_.native_register(), 3);
    }
    else
    {
        // Extra elements means we can use full width-4 load/store operations
        float32x4x2_t  t0 = vuzpq_f32(v0.simdInternal_.native_register(), v2.simdInternal_.native_register());
        float32x4x2_t  t1 = vuzpq_f32(v1.simdInternal_.native_register(), vdupq_n_f32(0.0f));
        float32x4x2_t  t2 = vtrnq_f32(t0.val[0], t1.val[0]);
        float32x4x2_t  t3 = vtrnq_f32(t0.val[1], t1.val[1]);
        float32x4_t    t4 = t2.val[0];
        float32x4_t    t5 = t3.val[0];
        float32x4_t    t6 = t2.val[1];
        float32x4_t    t7 = t3.val[1];

        vst1q_f32(base + align * offset[0], vaddq_f32(t4, vld1q_f32(base + align * offset[0])));
        vst1q_f32(base + align * offset[1], vaddq_f32(t5, vld1q_f32(base + align * offset[1])));
        vst1q_f32(base + align * offset[2], vaddq_f32(t6, vld1q_f32(base + align * offset[2])));
        vst1q_f32(base + align * offset[3], vaddq_f32(t7, vld1q_f32(base + align * offset[3])));
    }
}

template <int align>
static inline void gmx_simdcall
transposeScatterDecrU(float *              base,
                      const std::int32_t   offset[],
                      SimdFloat            v0,
                      SimdFloat            v1,
                      SimdFloat            v2)
{
    assert(std::size_t(offset) % 16 == 0);

    if (align < 4)
    {
        float32x2_t   t0, t1, t2, t3;
        float32x4x2_t tmp = vtrnq_f32(v0.simdInternal_.native_register(), v1.simdInternal_.native_register());

        t0 = vget_low_f32(tmp.val[0]);
        t1 = vget_low_f32(tmp.val[1]);
        t2 = vget_high_f32(tmp.val[0]);
        t3 = vget_high_f32(tmp.val[1]);

        t0 = vsub_f32(vld1_f32(base + align * offset[0]), t0);
        vst1_f32(base + align * offset[0], t0);
        base[ align * offset[0] + 2] -= vgetq_lane_f32(v2.simdInternal_.native_register(), 0);

        t1 = vsub_f32(vld1_f32(base + align * offset[1]), t1);
        vst1_f32(base + align * offset[1], t1);
        base[ align * offset[1] + 2] -= vgetq_lane_f32(v2.simdInternal_.native_register(), 1);

        t2 = vsub_f32(vld1_f32(base + align * offset[2]), t2);
        vst1_f32(base + align * offset[2], t2);
        base[ align * offset[2] + 2] -= vgetq_lane_f32(v2.simdInternal_.native_register(), 2);

        t3 = vsub_f32(vld1_f32(base + align * offset[3]), t3);
        vst1_f32(base + align * offset[3], t3);
        base[ align * offset[3] + 2] -= vgetq_lane_f32(v2.simdInternal_.native_register(), 3);
    }
    else
    {
        // Extra elements means we can use full width-4 load/store operations
        float32x4x2_t  t0 = vuzpq_f32(v0.simdInternal_.native_register(), v2.simdInternal_.native_register());
        float32x4x2_t  t1 = vuzpq_f32(v1.simdInternal_.native_register(), vdupq_n_f32(0.0f));
        float32x4x2_t  t2 = vtrnq_f32(t0.val[0], t1.val[0]);
        float32x4x2_t  t3 = vtrnq_f32(t0.val[1], t1.val[1]);
        float32x4_t    t4 = t2.val[0];
        float32x4_t    t5 = t3.val[0];
        float32x4_t    t6 = t2.val[1];
        float32x4_t    t7 = t3.val[1];

        vst1q_f32(base + align * offset[0], vsubq_f32(vld1q_f32(base + align * offset[0]), t4));
        vst1q_f32(base + align * offset[1], vsubq_f32(vld1q_f32(base + align * offset[1]), t5));
        vst1q_f32(base + align * offset[2], vsubq_f32(vld1q_f32(base + align * offset[2]), t6));
        vst1q_f32(base + align * offset[3], vsubq_f32(vld1q_f32(base + align * offset[3]), t7));
    }
}

static inline void gmx_simdcall
expandScalarsToTriplets(SimdFloat    scalar,
                        SimdFloat *  triplets0,
                        SimdFloat *  triplets1,
                        SimdFloat *  triplets2)
{
    float32x2_t lo, hi;
    float32x4_t t0, t1, t2, t3;

    lo = vget_low_f32(scalar.simdInternal_.native_register());
    hi = vget_high_f32(scalar.simdInternal_.native_register());

    t0 = vdupq_lane_f32(lo, 0);
    t1 = vdupq_lane_f32(lo, 1);
    t2 = vdupq_lane_f32(hi, 0);
    t3 = vdupq_lane_f32(hi, 1);

    triplets0->simdInternal_ = vextq_f32(t0, t1, 1);
    triplets1->simdInternal_ = vextq_f32(t1, t2, 2);
    triplets2->simdInternal_ = vextq_f32(t2, t3, 3);
}


template <int align>
static inline void gmx_simdcall
gatherLoadBySimdIntTranspose(const float *  base,
                             SimdFInt32     offset,
                             SimdFloat *    v0,
                             SimdFloat *    v1,
                             SimdFloat *    v2,
                             SimdFloat *    v3)
{
    alignas(GMX_SIMD_ALIGNMENT) std::int32_t  ioffset[GMX_SIMD_FINT32_WIDTH];

    assert(std::size_t(base) % 16 == 0);
    assert(align % 4 == 0);

    store(ioffset, offset);
    gatherLoadTranspose<align>(base, ioffset, v0, v1, v2, v3);
}

template <int align>
static inline void gmx_simdcall
gatherLoadBySimdIntTranspose(const float *   base,
                             SimdFInt32      offset,
                             SimdFloat *     v0,
                             SimdFloat *     v1)
{
    alignas(GMX_SIMD_ALIGNMENT) std::int32_t  ioffset[GMX_SIMD_FINT32_WIDTH];

    store(ioffset, offset);
    gatherLoadTranspose<align>(base, ioffset, v0, v1);
}



template <int align>
static inline void gmx_simdcall
gatherLoadUBySimdIntTranspose(const float *  base,
                              SimdFInt32     offset,
                              SimdFloat *    v0,
                              SimdFloat *    v1)
{
    alignas(GMX_SIMD_ALIGNMENT) std::int32_t  ioffset[GMX_SIMD_FINT32_WIDTH];

    store(ioffset, offset);
    v0->simdInternal_ = vcombine_f32(vld1_f32( base + align * ioffset[0] ),
                                     vld1_f32( base + align * ioffset[2] ));
    v1->simdInternal_ = vcombine_f32(vld1_f32( base + align * ioffset[1] ),
                                     vld1_f32( base + align * ioffset[3] ));
    float32x4x2_t tmp = vtrnq_f32(v0->simdInternal_.native_register(), v1->simdInternal_.native_register());
    v0->simdInternal_ = tmp.val[0];
    v1->simdInternal_ = tmp.val[1];
}

static inline float gmx_simdcall
reduceIncr4ReturnSum(float *    m,
                     SimdFloat  v0,
                     SimdFloat  v1,
                     SimdFloat  v2,
                     SimdFloat  v3)
{
    assert(std::size_t(m) % 16 == 0);

    float32x4x2_t  t0 = vuzpq_f32(v0.simdInternal_.native_register(), v2.simdInternal_.native_register());
    float32x4x2_t  t1 = vuzpq_f32(v1.simdInternal_.native_register(), v3.simdInternal_.native_register());
    float32x4x2_t  t2 = vtrnq_f32(t0.val[0], t1.val[0]);
    float32x4x2_t  t3 = vtrnq_f32(t0.val[1], t1.val[1]);
    v0.simdInternal_ = t2.val[0];
    v1.simdInternal_ = t3.val[0];
    v2.simdInternal_ = t2.val[1];
    v3.simdInternal_ = t3.val[1];

    v0 = v0 + v1;
    v2 = v2 + v3;
    v0 = v0 + v2;
    v2 = v0 + simdLoad(m);
    store(m, v2);

    return reduce(v0);
}

#endif

}      // namespace gm

#endif // GMX_SIMD_IMPL_NSIMD_UTIL_FLOAT_H
