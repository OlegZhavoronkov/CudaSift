cmake_minimum_required(VERSION 3.20)

project(cudaSift 
LANGUAGES C CXX)
set(CMAKE_CXX_EXTENSIONS OFF)
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
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)

set(CUDASIFT_CUDA_ARCH_NOT_SET "CUDASIFT_CUDA_ARCH_NOT_SET")
set(CUDASIFT_CUDA_ARCH_USED)
set(CUDASIFT_CUDA_ARCH ${CUDASIFT_CUDA_ARCH_NOT_SET} CACHE STRING "cuda arch for cuda sfp- eg. 75;8.6")

find_package(CUDAToolkit )
if (NOT CUDAToolkit_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
  return()
else()
    message(STATUS "CUDA found \"${CUDAToolkit_FOUND}\". version \"${CUDAToolkit_VERSION}\" at \"${CUDAToolkit_INCLUDE_DIRS}\" CMAKE_CUDA_COMPILER_LOADED \"${CMAKE_CUDA_COMPILER_LOADED}\"")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if(NOT CUDASIFT_CUDA_ARCH STREQUAL ${CUDASIFT_CUDA_ARCH_NOT_SET})
            message("CUDASIFT_CUDA_ARCH \"${CUDASIFT_CUDA_ARCH}\"")
            set(CMAKE_CUDA_ARCHITECTURES ${CUDASFP_CUDA_ARCH})
            set(CUDASIFT_CUDA_ARCH_USED ${CUDASIFT_CUDA_ARCH})
        else()
            include(FindCUDA/select_compute_arch)
            CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
            string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
            string(REPLACE "." "" CUDA_ARCH_LIST1 "${INSTALLED_GPU_CCS_2}")
            string(REPLACE " " ";" CUDA_ARCH_LIST "${CUDA_ARCH_LIST1}")
            set(CUDASIFT_CUDA_ARCH_USED ${CUDA_ARCH_LIST})
            set(CUDASIFT_CUDA_ARCH ${CUDA_ARCH_LIST})
#            set(CUDASFP_CUDA_ARCH "Auto")
#            set(CUDASFP_CUDA_ARCH_USED "Auto")
            message("CUDASIFT_CUDA_ARCH_USED defaulted to \"${CUDASIFT_CUDA_ARCH_USED}\"")
            #set(CMAKE_CUDA_ARCHITECTURES 75)
        endif()
        if("${CMAKE_CUDA_COMPILER}" STREQUAL "CMAKE_CUDA_COMPILER-NOTFOUND" OR "${CMAKE_CUDA_COMPILER}" STREQUAL "NOTFOUND")
            set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
        endif()
        set(CMAKE_CUDA_ARCHITECTURES ${CUDASIFT_CUDA_ARCH_USED})
    endif()
    message("enabling CUDA language \n$\{CMAKE_CUDA_ARCHITECTURES\} \"${CMAKE_CUDA_ARCHITECTURES}\" \n$\{CMAKE_CUDA_COMPILER\} \"${CMAKE_CUDA_COMPILER}\" \n$\{CUDAToolkit_NVCC_EXECUTABLE\} \"${CUDAToolkit_NVCC_EXECUTABLE}\"")
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

set( cuda_sift_sources
        ${CMAKE_CURRENT_LIST_DIR}/src/cudaSiftD.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/cudaSiftH.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/geomFuncs.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/match.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/matching.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/siftDeviceResidents.cuh
        ${CMAKE_CURRENT_LIST_DIR}/src/siftDeviceResidents.cu
)  

set (cuda_sift_hdrs
        ${CMAKE_CURRENT_LIST_DIR}/include/cudasift/cudaImage.h  
        ${CMAKE_CURRENT_LIST_DIR}/include/cudasift/cudaSiftH.h  
        ${CMAKE_CURRENT_LIST_DIR}/include/cudasift/cudaSiftD.h  
        ${CMAKE_CURRENT_LIST_DIR}/include/cudasift/cudaSift.h  
        ${CMAKE_CURRENT_LIST_DIR}/include/cudasift/cudautils.h
)

set(sift_sources
        ${CMAKE_CURRENT_LIST_DIR}/src/geomFuncs.cpp  
        ${CMAKE_CURRENT_LIST_DIR}/src/cudaSiftH.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/cudaImage.cpp
)

#include_directories(
#  ${CMAKE_CURRENT_SOURCE_DIR} 
#)



add_library(
    cudasift 
    SHARED ""
)

target_sources(cudasift 
    INTERFACE 
    ${cuda_sift_hdrs}
    PRIVATE
    ${cuda_sift_sources}
    ${sift_sources}
)

target_include_directories(cudasift 
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include/
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(cudasift PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
)

set(cudasift_dep_libs ${CUDA_cudart_LIBRARY} opencv_core)
message("$\{CUDA_cudart_LIBRARY\}\n\"${CUDA_cudart_LIBRARY}\" \n$\{OpenCV_LIBS\} \"${OpenCV_LIBS}\"\n$\{cudasift_dep_libs\}\n${cudasift_dep_libs}")
target_link_libraries(  cudasift 
PUBLIC
PRIVATE 
        ${cudasift_dep_libs}
)
#  /usr/local/cuda/lib64/libcudadevrt.a ${OpenCV_LIBS} 
#)

message( "CUDA_ARCHITECTURES \"${CUDA_ARCHITECTURES}\" CMAKE_CUDA_ARCHITECTURES \"${CMAKE_CUDA_ARCHITECTURES}\"\n$\{CUDASIFT_CUDA_ARCH_USED\}\n\"${CUDASIFT_CUDA_ARCH_USED}\"" )
set_property(TARGET cudasift PROPERTY CUDA_ARCHITECTURES ${CUDASIFT_CUDA_ARCH_USED})
set_property(TARGET cudasift PROPERTY CUDA_SEPARABLE_COMPILATION ON)
add_executable(cudasift_app 
    ${CMAKE_CURRENT_LIST_DIR}/app/mainSift.cpp
)
set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/app/mainSift.cpp PROPERTIES COMPILE_DEFINITIONS "TEST_DATA_FOLDER=\"${CMAKE_CURRENT_LIST_DIR}/data\"")
target_link_libraries(cudasift_app PRIVATE cudasift ${OpenCV_LIBS})

#install(FILES 
#  ${cuda_sources} 
#  ${sources}
#  cudaSiftD.cu
#  CMakeLists.txt
#  Copyright.txt
#  DESTINATION .
#)
#install(FILES data/left.pgm data/righ.pgm
#  DESTINATION data
#)
