cmake_minimum_required (VERSION 3.5)
project ("Vayesta")

if (NOT BLAS_LIBRARIES)
find_package(BLAS)
endif()

if (NOT BLAS_LIBRARIES)
  message(FATAL_ERROR "A required library with BLAS API not found.")
else()
  message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
endif()

add_subdirectory(core)
