#ifndef __AONFP_DETAIL_MACRO_HPP__
#define __AONFP_DETAIL_MACRO_HPP__

#ifndef AONFP_HOST_DEVICE
 #if defined(__CUDA_ARCH__)
  #define AONFP_HOST_DEVICE __device__ __host__
 #else
  #define AONFP_HOST_DEVICE
 #endif
#endif

#endif
