#ifndef __AONFP_DETAIL_STANDARDFP_HPP__
#define __AONFP_DETAIL_STANDARDFP_HPP__

#ifndef AONFP_HOST_DEVICE
 #if defined(__CUDA_ARCH__)
  #define AONFP_HOST_DEVICE __device__ __host__
 #else
  #define AONFP_HOST_DEVICE
 #endif
#endif
#include <cmath>
#include <cstdint>

namespace aonfp {
namespace detail {
namespace standard_fp {
// If all people in the world used C++14 or later, these functions could be `template AONFP_HOST_DEVICE constexpr`.
template <class T>
AONFP_HOST_DEVICE constexpr unsigned get_exponent_size();
template <> AONFP_HOST_DEVICE constexpr unsigned get_exponent_size<float >() {return 8 ;}
template <> AONFP_HOST_DEVICE constexpr unsigned get_exponent_size<double>() {return 11;}

template <class T>
AONFP_HOST_DEVICE constexpr unsigned get_mantissa_size();
template <> AONFP_HOST_DEVICE constexpr unsigned get_mantissa_size<float >() {return 23;}
template <> AONFP_HOST_DEVICE constexpr unsigned get_mantissa_size<double>() {return 52;}

template <class T>
AONFP_HOST_DEVICE constexpr T get_nan(const T original_fp) {
	// TODO: sign
	return NAN;
}

template <class T>
AONFP_HOST_DEVICE constexpr T get_infinity(const T original_fp) {
	// TODO: sign
	return INFINITY;
}
} // namespace standard_fp
} // namespace detail
} // namespace aonfp
#endif
