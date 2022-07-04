add_library(core SHARED)

add_subdirectory(src)

target_link_libraries(core ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES})
