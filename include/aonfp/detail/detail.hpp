#ifndef __AONFP_DETAIL_DETAIL_HPP__
#define __AONFP_DETAIL_DETAIL_HPP__

#ifndef AONFP_HOST_DEVICE
 #if defined(__CUDA_ARCH__)
  #define AONFP_HOST_DEVICE __device__ __host__
 #else
  #define AONFP_HOST_DEVICE
 #endif
#endif
#include <limits>
#include <cstdint>
#include <type_traits>
#include "standard_fp.hpp"

namespace aonfp {
namespace detail {

enum uo_flow_t {
	uo_flow_non,
	uo_flow_overflow,
	uo_flow_underflow,
};

struct aonfp_uint128_t {
	uint64_t x[2];
	aonfp_uint128_t operator=(const uint64_t v) {
		x[0] = v;
		x[1] = 0;
		return *this;
	}
};

template <class T>
struct bitstring_t {using type = T;};
template <> struct bitstring_t<float > {using type = uint32_t;};
template <> struct bitstring_t<double> {using type = uint64_t;};
template <> struct bitstring_t<const float > {using type = uint32_t;};
template <> struct bitstring_t<const double> {using type = uint64_t;};

AONFP_HOST_DEVICE constexpr unsigned long get_default_exponent_bias(const unsigned exponent_size) {
	return (1 << (exponent_size - 1)) - 1;
}

template <class T>
AONFP_HOST_DEVICE constexpr T get_exponent_bitstring(const T s_exp) {return ((static_cast<T>(1) << (sizeof(T) * 8 - 1)) - 1) & s_exp;}

template <class T>
AONFP_HOST_DEVICE constexpr T get_sign_bitstring(const T s_exp) {return s_exp & (static_cast<T>(1) << (sizeof(T) * 8 - 1));}

template <class T>
AONFP_HOST_DEVICE constexpr T get_nan_sign_exponent_bitstring() {return (static_cast<T>(1) << (sizeof(T) * 8 - 1)) - 1;}

template <class T>
AONFP_HOST_DEVICE constexpr T get_nan_mantissa_bitstring() {return ~static_cast<T>(0);}

template <class T>
AONFP_HOST_DEVICE constexpr T get_inf_sign_exponent_bitstring(T s_exp) {return get_nan_sign_exponent_bitstring<T>() | get_sign_bitstring(s_exp);}

template <class T>
AONFP_HOST_DEVICE constexpr T get_inf_mantissa_bitstring() {return static_cast<T>(0);};

template <class T>
AONFP_HOST_DEVICE constexpr T get_zero_sign_exponent_bitstring(T s_exp = 0) {return get_sign_bitstring(s_exp);}

template <class T>
AONFP_HOST_DEVICE constexpr T get_zero_mantissa_bitstring() {return static_cast<T>(0);};


// range getter
AONFP_HOST_DEVICE constexpr long get_max_exponent(const unsigned expopent_length) {return static_cast<long>(get_default_exponent_bias(expopent_length));}

AONFP_HOST_DEVICE constexpr long get_min_exponent(const unsigned exponent_length) {return 1 - static_cast<long>(get_default_exponent_bias(exponent_length));}

} //namespace detail
} //namespace aonfp
#endif
