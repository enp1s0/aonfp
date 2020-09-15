#ifndef __AONFP_DETAIL_Q_HPP__
#define __AONFP_DETAIL_Q_HPP__
#include "macro.hpp"

namespace aonfp {
namespace detail {
namespace q {
template <class T>
AONFP_HOST_DEVICE constexpr T get_exponent_bitstring(const T exp) {return exp;}

template <class T>
AONFP_HOST_DEVICE constexpr T get_sign_bitstring(const T s_mantissa) {return s_mantissa & (static_cast<T>(1) << (sizeof(T) * 8 - 1));}

template <class T>
AONFP_HOST_DEVICE constexpr T get_nan_exponent_bitstring() {return ~static_cast<T>(0);}

template <class T>
AONFP_HOST_DEVICE constexpr T get_nan_sign_mantissa_bitstring() {return ~(static_cast<T>(1) << (sizeof(T) * 8 - 1));}

template <class T>
AONFP_HOST_DEVICE constexpr T get_inf_exponent_bitstring() {return ~static_cast<T>(0);}

template <class T>
AONFP_HOST_DEVICE constexpr T get_inf_sign_mantissa_bitstring(const T s_mantissa) {return static_cast<T>(0) | get_sign_bitstring(s_mantissa);};

template <class T>
AONFP_HOST_DEVICE constexpr T get_zero_sign_exponent_bitstring() {return 0;}

template <class T>
AONFP_HOST_DEVICE constexpr T get_zero_mantissa_bitstring(const T s_mantissa) {return get_sign_bitstring(s_mantissa);};
} // namespace q
} // namespace detail
} // namespace aonfp
#endif
