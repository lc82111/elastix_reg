# Minimum CMake version. Intended to correspond with the `cmake_minimum_required` version of elastix.
cmake_minimum_required(VERSION 3.16.3)
project(ElastixRigidRegistration)

find_package(Elastix REQUIRED)

# Elastix might use OpenMP.
message(STATUS "Checking for OpenMP")
find_package(OpenMP QUIET)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Use the version of ITK from Elastix.
message(STATUS "Using ITK from Elastix: ${ELASTIX_ITK_DIR}")
if (DEFINED ELASTIX_ITK_DIR)
  set(ITK_DIR "${ELASTIX_ITK_DIR}" CACHE PATH "ITK_DIR from Elastix" FORCE)
endif()
find_package(ITK REQUIRED)

include(${ELASTIX_CONFIG_TARGETS_FILE})

message(STATUS "Elastix libraries: ${ELASTIX_LIBRARIES}")
message(STATUS "elastix_lib: ${elastix_lib}")
message(STATUS "transformix_lib: ${transformix_lib}")
message(STATUS "Elastix include directories: ${ELASTIX_INCLUDE_DIRS}")

# Enable debug flags
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

add_executable(rigid_registration main.cpp)

set_property(TARGET rigid_registration PROPERTY CXX_STANDARD 14)

target_include_directories(rigid_registration
  PRIVATE ${ELASTIX_INCLUDE_DIRS} ${ITK_INCLUDE_DIRS})

target_link_libraries(rigid_registration
  PRIVATE ${ITK_LIBRARIES} elastix_lib)