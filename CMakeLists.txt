cmake_minimum_required(VERSION 3.20)

project(cudaSift 
LANGUAGES C CXX CUDA)
set(CMAKE_CXX_EXTENSIONS ON)
set(cudaSift_VERSION_MAJOR 2)
set(cudaSift_VERSION_MINOR 0)
set(cudaSift_VERSION_PATCH 0)

set(CPACK_PACKAGE_VERSION_MAJOR "${cudaSift_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${cudaSift_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${cudaSift_VERSION_PATCH}")
set(CPACK_GENERATOR "ZIP")
include(CPack)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(RemoveOption)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit )
if (NOT CUDAToolkit_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
  return()
else()
    message(STATUS "CUDA found \"${CUDAToolkit_FOUND}\". version \"${CUDAToolkit_VERSION}\" at \"${CUDAToolkit_INCLUDE_DIRS}\"")
    enable_language(CUDA)
endif(NOT CUDAToolkit_FOUND)
set(CLINE_WO_PEDANTIC "")

remove_specific_option_from_command_line(${CMAKE_CXX_FLAGS} "-pedantic" CLINE_WO_PEDANTIC)
set(CMAKE_CXX_FLAGS "${CLINE_WO_PEDANTIC}")
remove_specific_option_from_command_line(${CMAKE_C_FLAGS} "-pedantic" CLINE_WO_PEDANTIC)
set(CMAKE_C_FLAGS "${CLINE_WO_PEDANTIC}")

#if (WIN32)
#  set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS ")
#  list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;-O2;-DVERBOSE") 
#endif()
#if (UNIX)
#  if (APPLE)
#    set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2")
#    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;-O2;-DVERBOSE") 
#  else()
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse2 ")
#    list(APPEND CUDA_NVCC_FLAGS "-lineinfo;-ccbin;/usr/bin/gcc-6;--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT") 
#  endif()
#endif()

set(cuda_sources
  cudaImage.cu  
  cudaImage.h  
  cudaSiftH.cu 
  cudaSiftH.h  
  matching.cu  
  cudaSiftD.h  
  cudaSift.h  
  cudautils.h
)  

set(sources
  geomFuncs.cpp  
  mainSift.cpp  
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR} 
)

#SET(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(cudasift ${cuda_sources} ${sources} OPTIONS -arch=sm_35)

#cuda_add_executable(l2net l2netD.cu OPTIONS -arch=sm_35)

set_target_properties(cudasift PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
)

target_link_libraries(cudasift ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})
#  /usr/local/cuda/lib64/libcudadevrt.a ${OpenCV_LIBS} 
#)

if(NOT CUDA_ARCHITECTURES)
    #set(CUDA_ARCHITECTURES "Auto")
    set_property(TARGET cudasift PROPERTY CUDA_ARCHITECTURES "75")
endif()

install(FILES 
  ${cuda_sources} 
  ${sources}
  cudaSiftD.cu
  CMakeLists.txt
  Copyright.txt
  DESTINATION .
)
install(FILES data/left.pgm data/righ.pgm
  DESTINATION data
)
