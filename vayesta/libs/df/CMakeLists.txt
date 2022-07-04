add_library(df SHARED)

add_subdirectory(src)

target_link_libraries(df ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES})
target_include_directories(df PRIVATE ${HDF5_C_INCLUDE_DIRS})
target_link_libraries(df ${HDF5_HL_LIBRARIES} ${OPENMP_C_PROPERTIES})
