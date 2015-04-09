cmake_minimum_required(VERSION 2.8.3)
project(humanoid_localization)
set(Boost_DEBUG ON)

find_package(catkin REQUIRED COMPONENTS cmake_modules roscpp pcl_conversions pcl_ros message_filters sensor_msgs)
find_package(OpenMP)
find_package(Boost REQUIRED COMPONENTS random)
find_package(Eigen REQUIRED)
find_package(octomap REQUIRED)
include_directories(${OCTOMAP_INCLUDE_DIRS})
link_libraries(${OCTOMAP_LIBRARIES})

# required for OpenMP
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

find_package(dynamicEDT3D)
if (${dynamicEDT3D_FOUND})                                                                       
  include_directories(${DYNAMICEDT3D_INCLUDE_DIRS})                                                          
  link_directories(${DYNAMICEDT3D_LIBRARY_DIRS})
  LINK_LIBRARIES(${OCTOMAP_LIBRARIES} ${DYNAMICEDT3D_LIBRARIES})
  list(APPEND LIB_FILES src/EndpointModel.cpp)                                      
else (${dynamicEDT3D_FOUND})
  MESSAGE(WARNING "dynamicEDT3D library (part of OctoMap >1.5) not found, skipping endpoint model")
  add_definitions(-DSKIP_ENDPOINT_MODEL)
endif (${dynamicEDT3D_FOUND})

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES mapmodel motionmodel observationmodel raycastingmodel observationmodel humanoidlocalization
  CATKIN_DEPENDS tf
  DEPENDS octomap OpenMP
)

find_package(PCL 1.7.0 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(include
  ${catkin_INCLUDE_DIRS}
)

set(EXECUTABLE_OUTPUT_PATH /home/sysadm/catkin_workspace/devel/lib/humanoid_localization)

if (${dynamicEDT3D_FOUND})
  add_library(endpointmodel src/EndpointModel.cpp)
endif (${dynamicEDT3D_FOUND})

add_library(mapmodel src/MapModel.cpp)
add_library(motionmodel src/MotionModel.cpp)
add_library(observationmodel src/ObservationModel.cpp)

add_library(raycastingmodel src/RaycastingModel.cpp)
target_link_libraries(raycastingmodel observationmodel)

add_library(humanoidlocalization src/HumanoidLocalization.cpp)
if (${dynamicEDT3D_FOUND})
target_link_libraries(humanoidlocalization mapmodel motionmodel observationmodel raycastingmodel endpointmodel ${catkin_LIBRARIES})                                     
else (${dynamicEDT3D_FOUND})
target_link_libraries(humanoidlocalization mapmodel motionmodel observationmodel raycastingmodel ${catkin_LIBRARIES})
endif (${dynamicEDT3D_FOUND})

add_executable(localization_node src/localization_node.cpp)
target_link_libraries(localization_node humanoidlocalization ${catkin_LIBRARIES} ${PCL_LIBRARIES})

# install
install(TARGETS mapmodel motionmodel observationmodel raycastingmodel humanoidlocalization
        DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)
set(CATKIN_PACKAGE_BIN_DESTINATION /home/sysadm/catkin_workspace/devel/lib/humanoid_localization)
install(TARGETS localization_node
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
