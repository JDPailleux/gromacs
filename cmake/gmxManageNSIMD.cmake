
# Manage the NSIMD setup. 

include(gmxDetectCpu)
include(gmxDetectSimd)

# Set variables path 
set(GMX_NSIMD_ROOT "/home/jpailleux/Bureau/nsdev/nsimd")
set(GMX_NSIMD_INCLUDE_PATH1 "${GMX_NSIMD_ROOT}/include")
set(GMX_NSIMD_INCLUDE_PATH2 "${GMX_NSIMD_ROOT}/nsimd/include")
set(GMX_NSIMD_LIBRARY_PATH "${GMX_NSIMD_ROOT}/build")

# Detection of architecture
# Section to set compiler flags for NSIMD.


# FIND LIBRARY AND SET COMPILE FLAGS/ LINKER FLAGS
gmx_suggest_simd(GMX_SUGGESTED_SIMD)
if ((${GMX_SUGGESTED_SIMD} STREQUAL "AVX2_256") OR (${GMX_SUGGESTED_SIMD} STREQUAL "AVX2_128"))
  set(NSIMD_COMPILE_FLAGS "-DAVX2 -mavx -mavx2")
  find_library(NSIMD_LIBRARY "libnsimd_x86_64")
  set(NSIMD_LINKER_FLAGS "-L${GMX_NSIMD_ROOT}/build -lnsimd_x86_64")

elseif(${GMX_SUGGESTED_SIMD} STREQUAL "SSE2")
  set(NSIMD_COMPILE_FLAGS "-DSSE2 -msse2")
  find_library(NSIMD_LIBRARY "libnsimd_x86_64")
  set(NSIMD_LINKER_FLAGS "-L${GMX_NSIMD_ROOT}/build -lnsimd_x86_64")

elseif(${GMX_SUGGESTED_SIMD} STREQUAL "SSE4.1")
  set(NSIMD_COMPILE_FLAGS "-DSSE42 -msse42")
  find_library(NSIMD_LIBRARY "libnsimd_x86_64")
  set(NSIMD_LINKER_FLAGS "-L${GMX_NSIMD_ROOT}/build -lnsimd_x86_64")

elseif ((${GMX_SUGGESTED_SIMD} STREQUAL "AVX_128_FMA") OR (${GMX_SUGGESTED_SIMD} STREQUAL "AVX_256"))
  set(NSIMD_COMPILE_FLAGS "-DAVX -mavx")
  find_library(NSIMD_LIBRARY "libnsimd_x86_64")
  set(NSIMD_LINKER_FLAGS "-L${GMX_NSIMD_ROOT}/build -lnsimd_x86_64")

elseif(${GMX_SUGGESTED_SIMD} STREQUAL "AVX_512_KNL")
  set(NSIMD_COMPILE_FLAGS "-DAVX512_KNL -mavx512_knl")
  find_library(NSIMD_LIBRARY "libnsimd_x86_64")
  set(NSIMD_LINKER_FLAGS "-L${GMX_NSIMD_ROOT}/build -lnsimd_x86_64")

elseif(${GMX_SUGGESTED_SIMD} STREQUAL "AVX_512")
  set(NSIMD_COMPILE_FLAGS "-DAVX512_SKYLAKE -mavx512_skylake")
  find_library(NSIMD_LIBRARY "libnsimd_x86_64")
  set(NSIMD_LINKER_FLAGS "-L${GMX_NSIMD_ROOT}/build -lnsimd_x86_64")

elseif(${GMX_SUGGESTED_SIMD} STREQUAL "ARM_NEON_ASIMD")
  set(NSIMD_COMPILE_FLAGS "-DAARCH64 -maarch64")
  find_library(NSIMD_LIBRARY "libnsimd_aarch64")
  set(NSIMD_LINKER_FLAGS "-L${GMX_NSIMD_ROOT}/build -lnsimd_aarch64")

elseif(${GMX_SUGGESTED_SIMD} STREQUAL "ARM_NEON")
  set(NSIMD_COMPILE_FLAGS "-DNEON128 -mneon128")
  find_library(NSIMD_LIBRARY "libnsimd_armv7")
  set(NSIMD_LINKER_FLAGS "-L${GMX_NSIMD_ROOT}/build -lnsimd_armv7")

else()
  message(FATAL_ERROR "Unsupported SIMD instruction sets by NSIMD")
endif()

# SET LINLER FLAGS
set(CMAKE_EXE_LINKER_FLAGS "${NSIMD_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${NSIMD_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")
set(GMX_PUBLIC_LIBRARIES "${NSIMD_COMPILE_FLAGS} ${NSIMD_LINKER_FLAGS}${GMX_PUBLIC_LIBRARIES}")

include_directories(SYSTEM ${GMX_NSIMD_INCLUDE_PATH1} ${GMX_NSIMD_INCLUDE_PATH2})
if (NSIMD_LIBRARY)
  list(APPEND GMX_EXTRA_LIBRARIES ${NSIMD_LIBRARY})
  list(APPEND GMX_COMMON_LIBRARIES ${NSIMD_LIBRARY})
  set(GMX_SIMD_NSIMD 1)
endif()

