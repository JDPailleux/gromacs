
# Manage the NSIMD setup. 

# Set variables path 
set(GMX_NSIMD_ROOT "/home/jpailleux/Bureau/nsdev/nsimd")
set(GMX_NSIMD_INCLUDE_PATH "${GMX_NSIMD_ROOT}/include")
set(GMX_NSIMD_LIBRARY_PATH "${GMX_NSIMD_ROOT}/build")

#set(GMX_NSIMD_FLAGS "-I$(GMX_NSIMD_INC) -L$(GMX_NSIMD_LIB) -lnsimd")

find_library(NSIMD_LIBRARY "libnsimd_x86_64")


set(NSIMD_COMPILE_FLAGS "-DAVX -mavx2")
set(NSIMD_LINKER_FLAGS "-L${GMX_NSIMD_ROOT}/build -lnsimd_x86_64")
include_directories(SYSTEM ${GMX_NSIMD_INCLUDE_PATH})
if (NSIMD_LIBRARY)
  list(APPEND GMX_EXTRA_LIBRARIES ${NSIMD_LIBRARY})
endif()

