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

#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/cxx_adv_api_functions.hpp>
#include <nsimd/nsimd.h>

#include "impl_nsimd_simd_double.h"

namespace gmx
{

// Internal utility function: Full 4x4 transpose of __m256
static inline void gmx_simdcall
avx256Transpose4By4(nsimd::pack<double> * v0,
                    nsimd::pack<double> * v1,
                    nsimd::pack<double> * v2,
                    nsimd::pack<double> * v3)
{
    nsimd::pack<double> t1 = _mm256_unpacko_pd(*v0, *v1);
    nsimd::pack<double> t2 = _mm256_unpackhi_pd(*v0, *v1);
    nsimd::pack<double> t3 = _mm256_unpacko_pd(*v2, *v3);
    nsimd::pack<double> t4 = _mm256_unpackhi_pd(*v2, *v3);
    *v0        = _mm256_permute2f128_pd(t1, t3, 0x20);
    *v1        = _mm256_permute2f128_pd(t2, t4, 0x20);
    *v2        = _mm256_permute2f128_pd(t1, t3, 0x31);
    *v3        = _mm256_permute2f128_pd(t2, t4, 0x31);
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

    simdInternal_ = nsimd::loada<nsimd::pack<double> >(base + align * offset[0]);
    simdInternal_ = nsimd::loada<nsimd::pack<double> >(base + align * offset[1]);
    simdInternal_ = nsimd::loada<nsimd::pack<double> >(base + align * offset[2]);
    simdInternal_ = nsimd::loada<nsimd::pack<double> >(base + align * offset[3]);
    avx256Transpose4By4(&v0->simdInternal_, &v1->simdInternal_, &v2->simdInternal_, &v3->simdInternal_);
}

template <int align>
static inline void gmx_simdcall
gatherLoadTranspose(const double *        base,
                    const std::int32_t    offset[],
                    SimdDouble *          v0,
                    SimdDouble *          v1)
{
    __m128d /*Invalid register*/ t1, t2, t3, t4;
    nsimd::pack<double> tA, tB;

    assert(std::size_t(offset) % 16 == 0);
    assert(std::size_t(base) % 16 == 0);
    assert(align % 2 == 0);

    t1 = nsimd::loada<nsimd::pack<double> >(base + align * offset[0]);
    t2 = nsimd::loada<nsimd::pack<double> >(base + align * offset[1]);
    t3 = nsimd::loada<nsimd::pack<double> >(base + align * offset[2]);
    t4 = nsimd::loada<nsimd::pack<double> >(base + align * offset[3]);
    tA   = _mm256_insertf128_pd(_mm256_castpd128_pd256(t1), t3, 0x1);
    tB   = _mm256_insertf128_pd(_mm256_castpd128_pd256(t2), t4, 0x1);

    v0->simdInternal_ = _mm256_unpacko_pd(tA, tB);
    v1->simdInternal_ = _mm256_unpackhi_pd(tA, tB);
}

static const int c_simdBestPairAlignmentDouble = 2;

// With the implementation below, thread-sanitizer can detect false positives
// For loading a triplet, we load 4 floats and ignore the last. Another threa
// might write to this element, but that will not affect the result
// On AVX2 we can use a gather intrinsic instead
template <int align>
static inline void gmx_simdcall
gatherLoadUTranspose(const double *        base,
                     const std::int32_t    offset[],
                     SimdDouble *          v0,
                     SimdDouble *          v1,
                     SimdDouble *          v2)
{
    assert(std::size_t(offset) % 16 == 0);

    nsimd::pack<double> t1, t2, t3, t4, t5, t6, t7, t8;
    if (align % 4 == 0)
    {
        t1 = nsimd::loada<nsimd::pack<double> >(base + align * offset[0]);
        t2 = nsimd::loada<nsimd::pack<double> >(base + align * offset[1]);
        t3 = nsimd::loada<nsimd::pack<double> >(base + align * offset[2]);
        t4 = nsimd::loada<nsimd::pack<double> >(base + align * offset[3]);
    }
    else
    {
        t1 = nsimd::loadu<nsimd::pack<double> >(base + align * offset[0]);
        t2 = nsimd::loadu<nsimd::pack<double> >(base + align * offset[1]);
        t3 = nsimd::loadu<nsimd::pack<double> >(base + align * offset[2]);
        t4 = nsimd::loadu<nsimd::pack<double> >(base + align * offset[3]);
    }
    t5                = _mm256_unpacko_pd(t1, t2);
    t6                = _mm256_unpackhi_pd(t1, t2);
    t7                = _mm256_unpacko_pd(t3, t4);
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
    nsimd::pack<double> t0, t1, t2;


    assert(std::size_t(offset) % 16 == 0);

    // v0: x0 x1 | x2 x
    // v1: y0 y1 | y2 y
    // v2: z0 z1 | z2 z

    t0 = _mm256_unpacko_pd(v0.simdInternal_, v1.simdInternal_); // x0 y0 | x2 y
    t1 = _mm256_unpackhi_pd(v0.simdInternal_, v1.simdInternal_); // x1 y1 | x3 y
    t2 = _mm256_unpackhi_pd(v2.simdInternal_, v2.simdInternal_); // z1 z1 | z3 z

    nsimd::storeu(base + align * offset[0], _mm256_castpd256_pd128(t0));
    nsimd::storeu(base + align * offset[1], _mm256_castpd256_pd128(t1));
    nsimd::storeu(base + align * offset[2], __builtin_ia32_vextractf128_pd256((__v4df)(__m256d)(t0), (int)(1)));
    nsimd::storeu(base + align * offset[3], __builtin_ia32_vextractf128_pd256((__v4df)(__m256d)(t1), (int)(1)));
    _mm_store_sd(base + align * offset[0] + 2, _mm256_castpd256_pd128(v2.simdInternal_));
    _mm_store_sd(base + align * offset[1] + 2, _mm256_castpd256_pd128(t2));
    _mm_store_sd(base + align * offset[2] + 2, _mm256_extractf128_pd(v2.simdInternal_, 0x1));
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
    nsimd::pack<double> t0, t1;
    __m128d /*Invalid register*/ t2, tA, tB;

    assert(std::size_t(offset) % 16 == 0);

    if (align % 4 == 0)
    {
        // we can use aligned load/stor
        t0 = nsimd::set1<nsimd::pack<double> >(0);
        avx256Transpose4By4(&v0.simdInternal_, &v1.simdInternal_, &v2.simdInternal_, &t0);
        nsimd::storea(base + align * offset[0], nsimd::loada<nsimd::pack<double> >(base + align * offset[0]) + v0.simdInternal_);
        nsimd::storea(base + align * offset[1], nsimd::loada<nsimd::pack<double> >(base + align * offset[1]) + v1.simdInternal_);
        nsimd::storea(base + align * offset[2], nsimd::loada<nsimd::pack<double> >(base + align * offset[2]) + v2.simdInternal_);
        nsimd::storea(base + align * offset[3], nsimd::loada<nsimd::pack<double> >(base + align * offset[3]) + t0);
    }
    else
    {
        // v0: x0 x1 | x2 x
        // v1: y0 y1 | y2 y
        // v2: z0 z1 | z2 z

        t0 = _mm256_unpacko_pd(v0.simdInternal_, v1.simdInternal_); // x0 y0 | x2 y
        t1 = _mm256_unpackhi_pd(v0.simdInternal_, v1.simdInternal_); // x1 y1 | x3 y
        t2 = _mm256_extractf128_pd(v2.simdInternal_, 0x1);           // z2 z

        tA = nsimd::loadu<nsimd::pack<double> >(base + align * offset[0]);
        tB = _mm_load_sd(base + align * offset[0] + 2);
        tA = tA + _mm256_castpd256_pd128(t0);
        tB = tB + _mm256_castpd256_pd128(v2.simdInternal_);
        nsimd::storeu(base + align * offset[0], tA);
        _mm_store_sd(base + align * offset[0] + 2, tB);

        tA = nsimd::loadu<nsimd::pack<double> >(base + align * offset[1]);
        tB = _mm_loadh_pd(nsimd::set1<nsimd::pack<double> >(0), base + align * offset[1] + 2);
        tA = tA + _mm256_castpd256_pd128(t1);
        tB = tB + _mm256_castpd256_pd128(v2.simdInternal_);
        nsimd::storeu(base + align * offset[1], tA);
        _mm_storeh_pd(base + align * offset[1] + 2, tB);

        tA = nsimd::loadu<nsimd::pack<double> >(base + align * offset[2]);
        tB = _mm_load_sd(base + align * offset[2] + 2);
        tA = tA + __builtin_ia32_vextractf128_pd256((__v4df)(__m256d)(t0), (int)(1));
        tB = tB + t2;
        nsimd::storeu(base + align * offset[2], tA);
        _mm_store_sd(base + align * offset[2] + 2, tB);

        tA = nsimd::loadu<nsimd::pack<double> >(base + align * offset[3]);
        tB = _mm_loadh_pd(nsimd::set1<nsimd::pack<double> >(0), base + align * offset[3] + 2);
        tA = tA + __builtin_ia32_vextractf128_pd256((__v4df)(__m256d)(t1), (int)(1));
        tB = tB + t2;
        nsimd::storeu(base + align * offset[3], tA);
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
    nsimd::pack<double> t0, t1;
    __m128d /*Invalid register*/ t2, tA, tB;

    assert(std::size_t(offset) % 16 == 0);

    if (align % 4 == 0)
    {
        // we can use aligned load/stor
        t0 = nsimd::set1<nsimd::pack<double> >(0);
        avx256Transpose4By4(&v0.simdInternal_, &v1.simdInternal_, &v2.simdInternal_, &t0);
        nsimd::storea(base + align * offset[0], nsimd::loada<nsimd::pack<double> >(base + align * offset[0]) - v0.simdInternal_);
        nsimd::storea(base + align * offset[1], nsimd::loada<nsimd::pack<double> >(base + align * offset[1]) - v1.simdInternal_);
        nsimd::storea(base + align * offset[2], nsimd::loada<nsimd::pack<double> >(base + align * offset[2]) - v2.simdInternal_);
        nsimd::storea(base + align * offset[3], nsimd::loada<nsimd::pack<double> >(base + align * offset[3]) - t0);
    }
    else
    {
        // v0: x0 x1 | x2 x
        // v1: y0 y1 | y2 y
        // v2: z0 z1 | z2 z

        t0 = _mm256_unpacko_pd(v0.simdInternal_, v1.simdInternal_); // x0 y0 | x2 y
        t1 = _mm256_unpackhi_pd(v0.simdInternal_, v1.simdInternal_); // x1 y1 | x3 y
        t2 = _mm256_extractf128_pd(v2.simdInternal_, 0x1);           // z2 z

        tA = nsimd::loadu<nsimd::pack<double> >(base + align * offset[0]);
        tB = _mm_load_sd(base + align * offset[0] + 2);
        tA = tA - _mm256_castpd256_pd128(t0);
        tB = tB - _mm256_castpd256_pd128(v2.simdInternal_);
        nsimd::storeu(base + align * offset[0], tA);
        _mm_store_sd(base + align * offset[0] + 2, tB);

        tA = nsimd::loadu<nsimd::pack<double> >(base + align * offset[1]);
        tB = _mm_loadh_pd(nsimd::set1<nsimd::pack<double> >(0), base + align * offset[1] + 2);
        tA = tA - _mm256_castpd256_pd128(t1);
        tB = tB - _mm256_castpd256_pd128(v2.simdInternal_);
        nsimd::storeu(base + align * offset[1], tA);
        _mm_storeh_pd(base + align * offset[1] + 2, tB);

        tA = nsimd::loadu<nsimd::pack<double> >(base + align * offset[2]);
        tB = _mm_load_sd(base + align * offset[2] + 2);
        tA = tA - __builtin_ia32_vextractf128_pd256((__v4df)(__m256d)(t0), (int)(1));
        tB = tB - t2;
        nsimd::storeu(base + align * offset[2], tA);
        _mm_store_sd(base + align * offset[2] + 2, tB);

        tA = nsimd::loadu<nsimd::pack<double> >(base + align * offset[3]);
        tB = _mm_loadh_pd(nsimd::set1<nsimd::pack<double> >(0), base + align * offset[3] + 2);
        tA = tA - __builtin_ia32_vextractf128_pd256((__v4df)(__m256d)(t1), (int)(1));
        tB = tB - t2;
        nsimd::storeu(base + align * offset[3], tA);
        _mm_storeh_pd(base + align * offset[3] + 2, tB);
    }
}

static inline void gmx_simdcall
expandScalarsToTriplets(SimdDouble    scalar,
                        SimdDouble *  triplets0,
                        SimdDouble *  triplets1,
                        SimdDouble *  triplets2)
{
    nsimd::pack<double> t0 = __builtin_ia32_vperm2f128_pd256((__v4df)(__m256d)(scalar.simdInternal_), (__v4df)(__m256d)(scalar.simdInternal_), (int)(33));
    nsimd::pack<double> t1 = __builtin_ia32_vpermilpd256((__v4df)(__m256d)(scalar.simdInternal_), (int)(0));
    nsimd::pack<double> t2 = __builtin_ia32_vpermilpd256((__v4df)(__m256d)(scalar.simdInternal_), (int)(15));
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
    _mm_store_si128( reinterpret_cast<__m128i *>(ioffset), offset.simdInternal_);

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
    __m128d /*Invalid register*/ t1, t2, t3, t4;
    nsimd::pack<double> tA, tB;

    assert(std::size_t(base) % 16 == 0);
    assert(align % 2 == 0);

    alignas(GMX_SIMD_ALIGNMENT) std::int32_t  ioffset[GMX_SIMD_DINT32_WIDTH];
    _mm_store_si128( reinterpret_cast<__m128i *>(ioffset), offset.simdInternal_);

    t1  = _mm_load_pd(base + align * ioffset[0]);
    t2  = _mm_load_pd(base + align * ioffset[1]);
    t3  = _mm_load_pd(base + align * ioffset[2]);
    t4  = _mm_load_pd(base + align * ioffset[3]);

    tA                = _mm256_insertf128_pd(_mm256_castpd128_pd256(t1), t3, 0x1);
    tB                = _mm256_insertf128_pd(_mm256_castpd128_pd256(t2), t4, 0x1);
    v0->simdInternal_ = _mm256_unpacko_pd(tA, tB);
    v1->simdInternal_ = _mm256_unpackhi_pd(tA, tB);
}

template <int align>
static inline void gmx_simdcall
gatherLoadUBySimdIntTranspose(const double *  base,
                              SimdDInt32      offset,
                              SimdDouble *    v0,
                              SimdDouble *    v1)
{
    __m128d /*Invalid register*/ t1, t2, t3, t4;
    nsimd::pack<double> tA, tB;

    alignas(GMX_SIMD_ALIGNMENT) std::int32_t ioffset[GMX_SIMD_DINT32_WIDTH];
    _mm_store_si128( reinterpret_cast<__m128i *>(ioffset), offset.simdInternal_);

    t1   = _mm_loadu_pd(base + align * ioffset[0]);
    t2   = _mm_loadu_pd(base + align * ioffset[1]);
    t3   = _mm_loadu_pd(base + align * ioffset[2]);
    t4   = _mm_loadu_pd(base + align * ioffset[3]);

    tA  = _mm256_insertf128_pd(_mm256_castpd128_pd256(t1), t3, 0x1);
    tB  = _mm256_insertf128_pd(_mm256_castpd128_pd256(t2), t4, 0x1);

    v0->simdInternal_ = _mm256_unpacko_pd(tA, tB);
    v1->simdInternal_ = _mm256_unpackhi_pd(tA, tB);
}

static inline double gmx_simdcall
reduceIncr4ReturnSum(double *    m,
                     SimdDouble  v0,
                     SimdDouble  v1,
                     SimdDouble  v2,
                     SimdDouble  v3)
{
    nsimd::pack<double> t0, t1, t2;
    __m128d /*Invalid register*/ a0, a1;

    assert(std::size_t(m) % 32 == 0);

    t0 = _mm256_hadd_pd(v0.simdInternal_, v1.simdInternal_);
    t1 = _mm256_hadd_pd(v2.simdInternal_, v3.simdInternal_);
    t2 = _mm256_permute2f128_pd(t0, t1, 0x21);
    t0 = t0 + t2;
    t1 = t1 + t2;
    t0 = _mm256_blend_pd(t0, t1, 0b1100);

    t1 = t0 + nsimd::loada<nsimd::pack<double> >(m);
    nsimd::storea(m, t1);

    t0 = t0 + __builtin_ia32_vpermilpd256((__v4df)(__m256d)(t0), (int)(5));
    a0  = _mm256_castpd256_pd128(t0);
    a1  = _mm256_extractf128_pd(t0, 0x1);
    a0  = _mm_add_sd(a0, a1);

    return *reinterpret_cast<double *>(&a0);
}


// This version is marginally slower than the AVX 4-wide component loa
// version on Intel Skylake. On older Intel architectures this versio
// is significantly slower
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

    *v0 = _mm256_i32gather_pd(base + 0, vindex.simdInternal_, sizeof(double));
    *v1 = _mm256_i32gather_pd(base + 1, vindex.simdInternal_, sizeof(double));
    *v2 = _mm256_i32gather_pd(base + 2, vindex.simdInternal_, sizeof(double));
}

}      //namespace gm

#endif // GMX_SIMD_IMPL_NSIMD_UTIL_DOUBLE_H
