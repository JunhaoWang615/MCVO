cmake_minimum_required(VERSION 2.8.3)
project(mcvo)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++1z")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(Torch_DIR ~/libtorch/share/cmake/Torch)  #你解压的libtorch的绝对路径



find_package(Torch REQUIRED)
find_package(OpenCV 4.5 REQUIRED)

add_definitions(-DENABLE_GPU=1)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  nav_msgs
  tf
  cv_bridge
  camera_model
  )
find_package(roslib REQUIRED)
find_package(Gflags)
# set(GLOG_LIBRARY "/usr/local/lib/libglog.so")
# set(GLOG_INCLUDE_DIR "/usr/local/include")
find_package(Glog  REQUIRED) 
find_library(GLOG_LIBRARY NAMES glog PATHS /usr/local/lib)
find_path(GLOG_INCLUDE_DIR NAMES logging.h PATHS /usr/local/include/glog)
# set(GLOG_INCLUDE_DIRS /usr/local/include)
# set(GLOG_LIBRARIES /usr/local/lib/libglog.so)

message(STATUS "Glog Libraries: ${GLOG_LIBRARIES}")
find_package(Eigen3 REQUIRED)

find_package(PCL REQUIRED)
message(STATUS "Found PCL version: ${PCL_VERSION}")
find_package(Ceres REQUIRED)
message(STATUS "Ceres Solver version: ${CERES_VERSION}")
message(STATUS "Ceres Solver include dirs: ${CERES_INCLUDE_DIRS}")
message(STATUS "Ceres Solver libraries: ${CERES_LIBRARIES}")


find_package(TBB REQUIRED)   
message(STATUS "Found TBB version: ${TBB_VERSION}")
find_package(vpi 2.0 REQUIRED)
  
#find_package( Sophus REQUIRED )
#include_directories( ${Sophus_INCLUDE_DIRS} )
include_directories(thirdparty/Sophus)
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${TBB_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS})

  link_directories(${OpenCV_LIBRARY_DIRS})
catkin_package(CATKIN_DEPENDS camera_model)


# add_executable(mcvofrontend_test
#   src/Frontend/feature_tracker/trackerbase.cpp
#   src/Frontend/feature_tracker/feature_tracker.cpp
#   src/Frontend/feature_tracker/vpi_feature_tracker.cpp
#   src/Frontend/feature_tracker/ORBfeature_tracker.cpp
#   src/Frontend/feature_tracker/ORBextractor/ORBextractor.cc

#   src/Frontend/sensors.cpp
#   src/Frontend/lidar.cpp
#   src/Frontend/MCVOfrontend.cpp

#   src/utility/utility.cpp
#   src/utility/visualization.cpp
#   src/utility/CameraPoseVisualization.cpp

#   src/test_main.cpp
#   )

add_executable(mcvo_estimator
  
  src/Frontend/feature_tracker/trackerbase.cpp
  src/Frontend/feature_tracker/feature_tracker.cpp
  src/Frontend/feature_tracker/vpi_feature_tracker.cpp
  src/Frontend/feature_tracker/ORBfeature_tracker.cpp
  src/Frontend/feature_tracker/ORBextractor/ORBextractor.cc
  src/Frontend/feature_tracker/Superpoint_tracker.cpp
  src/Frontend/feature_tracker/super_include/SuperGlue.cpp  
  src/Frontend/feature_tracker/super_include/SuperPoint.cpp

  src/Frontend/feature_tracker/SPHORBfeature_tracker.cpp
  src/Frontend/feature_tracker/SPHORB/detector.cpp
  src/Frontend/feature_tracker/SPHORB/nonmax.cpp
  src/Frontend/feature_tracker/SPHORB/pfm.cpp
  src/Frontend/feature_tracker/SPHORB/utility.cpp
  src/Frontend/feature_tracker/SPHORB/SPHORB.cpp

  src/Frontend/sensors.cpp
  src/Frontend/lidar.cpp
  src/Frontend/MCVOfrontend.cpp

  src/Estimator/MCVOestimator.cpp
  src/Estimator/MCVOfeature_manager.cpp
  src/Estimator/parameters.cpp
  
  src/Estimator/factor/pose_local_parameterization.cpp
  src/Estimator/factor/projection_factor.cpp
  src/Estimator/factor/projection_td_factor.cpp
  src/Estimator/factor/projection_mc_factor.cpp
  src/Estimator/factor/marginalization_factor.cpp
  src/Estimator/factor/SE3AbsolutatePoseFactor.cpp

  src/utility/utility.cpp
  src/utility/visualization.cpp
  src/utility/CameraPoseVisualization.cpp

  src/Estimator/initial/solve_5pts.cpp
  src/Estimator/initial/initial_sfm.cpp
  src/Estimator/initial/initial_aligment.cpp
  src/Estimator/initial/alignment_factor.cpp
  src/Estimator/initial/initial_ex_rotation.cpp

  src/MCVOestimator_node.cpp
)
# add_dependencies(mcvofrontend_test ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(mcvo_estimator ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

# target_link_libraries(mcvofrontend_test
#   ${catkin_LIBRARIES}
#   ${TBB_LIBRARIES}
#   ${OpenCV_LIBS}
#   ${PCL_LIBRARIES}
#   ${CERES_LIBRARIES}
#   ${GLOG_LIBRARIES}
#   ${GFLAGS_LIBRARIES}
#   vpi)

target_link_libraries(mcvo_estimator
  ${catkin_LIBRARIES}
  ${TBB_LIBRARIES}
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
  ${TORCH_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  /lib/x86_64-linux-gnu/libuuid.so.1 
  glog
  vpi
  ${roslib_LIBRARIES}  )

# add_executable(
#   pose_graph3d
#   src/pose_graph3d.cpp
#   src/factor/FrameParameterization.cpp
#   src/factor/SE3RelativtePoseFactor.cpp
# )

# target_link_libraries(
#   pose_graph3d
#   ${CERES_LIBRARIES}
#   ${GLOG_LIBRARIES}
#   ${GFLAGS_LIBRARIES}
# )


