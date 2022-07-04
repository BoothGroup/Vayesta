cmake_minimum_required (VERSION 3.5)
project ("Vayesta")

# OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
  set(HAVE_OPENMP 1)
  if [[ $CMAKE_C_COMPILER_ID == *"Clang"* ]]; then
    set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -lomp")
  endif()
else ()
  set(OpenMP_C_FLAGS " ")
endif()

# BLAS
if (NOT BLAS_LIBRARIES)
find_package(BLAS)
endif()
if (NOT BLAS_LIBRARIES)
    message(FATAL_ERROR "A required library with BLAS API not found.")
else()
    message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
endif()

# HDF5
#find_package(HDF5 REQUIRED COMPONENTS C HL)
#message(STATUS "HDF5 libraries: ${HDF5_LIBRARIES}")

# C FLAGS
set(CMAKE_C_FLAGS "-Wall ${CMAKE_C_FLAGS}")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
include(CheckCCompilerFlag)
CHECK_C_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if (COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
endif()
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=apple-m1")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -funroll-loops -ftree-vectorize")

add_subdirectory(core)
#add_subdirectory(df)
