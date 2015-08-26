set(SOURCE 
    triangulate.h
	triangulate.cpp)

if(WIN32)
    set(OpenCV_DIR "C:/Users/sarah/Downloads/opencv/build")
	set(EIGEN3_INCLUDE_DIR "C:/Users/sarah/Downloads/eigen-eigen-bdd17ee3b1b3/eigen-eigen-bdd17ee3b1b3")
endif(WIN32)
if(UNIX)
	set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -o3 ")
endif(UNIX)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

add_library(sky3d ${SOURCE})

target_link_libraries(sky3d ${OpenCV_LIBS})
