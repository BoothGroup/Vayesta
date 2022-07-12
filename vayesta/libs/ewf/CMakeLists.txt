add_library(embwf SHARED)

add_subdirectory(src)

target_link_libraries(embwf ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES})
