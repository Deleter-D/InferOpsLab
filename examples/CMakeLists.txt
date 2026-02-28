# Examples CMakeLists.txt Automatically discovers and builds all examples in
# subdirectories

# Get all subdirectories (example categories: gemm, conv, etc.)
file(
  GLOB EXAMPLE_CATEGORIES
  LIST_DIRECTORIES true
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "*")

foreach(CATEGORY ${EXAMPLE_CATEGORIES})
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${CATEGORY})
    # Check if the category has a CMakeLists.txt
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${CATEGORY}/CMakeLists.txt)
      add_subdirectory(${CATEGORY})
    else()
      # Auto-discover .cu files in the category
      file(GLOB EXAMPLE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${CATEGORY}/*.cu")
      foreach(SOURCE ${EXAMPLE_SOURCES})
        get_filename_component(EXAMPLE_NAME ${SOURCE} NAME_WE)
        add_executable(${EXAMPLE_NAME} ${SOURCE})
        target_link_libraries(${EXAMPLE_NAME} PRIVATE inferopslab_core)
      endforeach()
    endif()
  endif()
endforeach()
