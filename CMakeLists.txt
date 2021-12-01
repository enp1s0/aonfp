cmake_minimum_required(VERSION 3.18)
project(aonfp_cuda_copy LANGUAGES CXX CUDA)

find_package(CUDA 10.0 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# CUDA/CXX
foreach(lang CXX CUDA)
    set(CMAKE_${lang}_STANDARD          17)
    set(CMAKE_${lang}_STANDARD_REQUIRED ON)
    set(CMAKE_${lang}_EXTENSIONS        OFF)
endforeach()

# CUDA (1/2)
SET(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)

# Directories
set(INCDIR include)
set(SRCDIR src)

file(GLOB HEADERS "${INCDIR}/aonfp/*.hpp")
file(GLOB DETAIL_HEADERS "${INCDIR}/aonfp/detail/*.hpp")

add_library(aonfp_cuda_copy STATIC
	${SRCDIR}/cuda_copy.cu
	${HEADERS}
	${DETAIL_HEADERS}
	)

target_include_directories(aonfp_cuda_copy PUBLIC ${INCDIR})
target_link_libraries(aonfp_cuda_copy PRIVATE
	cuda
	)

set_target_properties(aonfp_cuda_copy PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

##########################################################################
# Installing
##########################################################################
install(TARGETS aonfp_cuda_copy
	LIBRARY DESTINATION lib
	)
install(
	DIRECTORY ${INCDIR} DESTINATION include
	)
