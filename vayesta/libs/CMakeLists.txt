cmake_minimum_required (VERSION 3.5)
project ("Vayesta")

# OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
  set(HAVE_OPENMP 1)
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

# C FLAGS
set(CMAKE_C_FLAGS "-Wall ${CMAKE_C_FLAGS}")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -funroll-loops -ftree-vectorize")

add_subdirectory(core)
