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

#ifndef GMX_SIMD_IMPL_NSIMD_SIMD4_FLOAT_H
#define GMX_SIMD_IMPL_NSIMD_SIMD4_FLOAT_H

#include "config.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/cxx_adv_api_functions.hpp>
#include <nsimd/nsimd.h>

#include "impl_nsimd_simd4_float_defined.h"
#include "impl_nsimd_general.h"

/*
#if defined(NSIMD_SSE2)
#undef GMX_SIMD_X86_SSE2
#define GMX_SIMD_X86_SSE2 1
#include "gromacs/simd/impl_x86_sse2/impl_x86_sse2_simd4_float.h"
#undef GMX_SIMD_X86_SSE2
#define GMX_SIMD_X86_SSE2 0

#elif defined(NSIMD_SSE42)
#undef GMX_SIMD_X86_SSE4_1
#define GMX_SIMD_X86_SSE4_1 1
#include "gromacs/simd/impl_x86_sse4_1/impl_x86_sse4_1_simd4_float.h"
#undef GMX_SIMD_X86_SSE4_1
#define GMX_SIMD_X86_SSE4_1 0

#elif defined(NSIMD_AVX)
#undef GMX_SIMD_X86_AVX_256
#define GMX_SIMD_X86_AVX_256 1
#include "gromacs/simd/impl_x86_avx_256/impl_x86_avx_256_simd4_float.h"
#undef GMX_SIMD_X86_AVX_256
#define GMX_SIMD_X86_AVX_256 0

#elif defined(NSIMD_AVX2)
#undef GMX_SIMD_X86_AVX2_256
#define GMX_SIMD_X86_AVX2_256 1
#include "gromacs/simd/impl_x86_avx2_256/impl_x86_avx2_256_simd4_float.h"
#undef GMX_SIMD_X86_AVX2_256
#define GMX_SIMD_X86_AVX2_256 0

#elif defined(NSIMD_AVX512_KNL)
#undef GMX_SIMD_X86_AVX_512_KNL
#define GMX_SIMD_X86_AVX_512_KNL 1
#include "gromacs/simd/impl_x86_avx_512_knl/impl_x86_avx_512_knl_simd4_float.h"
#undef GMX_SIMD_X86_AVX_512_KNL
#define GMX_SIMD_X86_AVX_512_KNL 0

#elif defined(NSIMD_AVX512_SKYLAKE)
#undef GMX_SIMD_X86_AVX_512
#define GMX_SIMD_X86_AVX_512 1
#include "gromacs/simd/impl_x86_avx_512/impl_x86_avx_512_simd4_float.h"
#undef GMX_SIMD_X86_AVX_512
#define GMX_SIMD_X86_AVX_512 0

#elif defined(NSIMD_AARCH64)
#undef GMX_SIMD_ARM_NEON_ASIMD
#define GMX_SIMD_ARM_NEON_ASIMD 1
#include "gromacs/simd/impl_arm_neon_asimd/impl_arm_neon_asimd_simd4_float.h"
#undef GMX_SIMD_ARM_NEON_ASIMD
#define GMX_SIMD_ARM_NEON_ASIMD 0

#elif defined(NSIMD_ARM_NEON)
#undef GMX_SIMD_ARM_NEON
#define GMX_SIMD_ARM_NEON 1
#include "gromacs/simd/impl_arm_neon/impl_arm_neon_simd4_float.h"
#undef GMX_SIMD_ARM_NEON
#define GMX_SIMD_ARM_NEON 0

#endif
*/

#endif // GMX_SIMD_IMPL_NSIMD_SIMD4_FLOAT_
