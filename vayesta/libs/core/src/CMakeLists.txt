target_sources(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/j3c_kao2gmo.c)
target_sources(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/j3c_jk.c)

set_target_properties(core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})
