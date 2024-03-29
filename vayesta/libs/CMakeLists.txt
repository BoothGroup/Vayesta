cmake_minimum_required (VERSION 3.5)
project ("Vayesta")

# OpenMP
set(OPENMP_C_PROPERTIES "")
find_package(OpenMP)
if(OPENMP_FOUND)
    set(HAVE_OPENMP 1)
    set(OPENMP_C_PROPERTIES OpenMP::OpenMP_C)
else()
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

if (APPLE)
    include_directories("${BLAS_LIBRARIES}/Frameworks/vecLib.framework/Headers")
endif()

if (APPLE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/deps/lib;@loader_path/deps/lib64")
    set(CMAKE_BUILD_RPATH "@loader_path;@loader_path/deps/lib;@loader_path/deps/lib64")
else()
    set(CMAKE_SKIP_BUILD_RPATH  True)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH True)
    set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/deps/lib:\$ORIGIN/deps/lib64")
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
