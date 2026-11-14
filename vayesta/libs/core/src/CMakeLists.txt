target_sources(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/j3c_kao2gmo.c)
target_sources(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/j3c_jk.c)

set_target_properties(core PROPERTIES
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

if(SKBUILD)
  # Built via pip/scikit-build-core: install into the wheel
  install(TARGETS core LIBRARY DESTINATION vayesta/libs)
else()
  # Standalone CMake build: place the library next to vayesta/libs/__init__.py
  set_target_properties(core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()
