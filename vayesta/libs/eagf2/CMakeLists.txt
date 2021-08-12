add_library(eagf2 SHARED)

add_subdirectory(src)

target_link_libraries(eagf2 ${BLAS_LIBRARIES})
