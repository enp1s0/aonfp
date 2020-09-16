#ifndef __AONFP_DETAIL_DETAIL_HPP__
#define __AONFP_DETAIL_DETAIL_HPP__

#include <limits>
#include <cstdint>
#include <type_traits>
#include "standard_fp.hpp"
#include "aonfp_uint128_t.hpp"
#include "macro.hpp"

namespace aonfp {
namespace detail {

enum uo_flow_t {
	uo_flow_non,
	uo_flow_overflow,
	uo_flow_underflow,
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

template <class T>
AONFP_HOST_DEVICE constexpr T get_max();
template <> AONFP_HOST_DEVICE constexpr uint64_t get_max<uint64_t>() {return 0xfffffffffffffffflu;}
template <> AONFP_HOST_DEVICE constexpr uint32_t get_max<uint32_t>() {return 0xffffffffu;}
template <> AONFP_HOST_DEVICE constexpr uint16_t get_max<uint16_t>() {return 0xffff;}
template <> AONFP_HOST_DEVICE constexpr uint8_t  get_max<uint8_t >() {return 0xff;}


// range getter
AONFP_HOST_DEVICE constexpr long get_max_exponent(const unsigned expopent_length) {return static_cast<long>(get_default_exponent_bias(expopent_length));}

AONFP_HOST_DEVICE constexpr long get_min_exponent(const unsigned exponent_length) {return 1 - static_cast<long>(get_default_exponent_bias(exponent_length));}

// bit operations
template <class T>
AONFP_HOST_DEVICE inline unsigned num_of_bits(T bits);
template <>
AONFP_HOST_DEVICE inline unsigned num_of_bits<uint64_t>(uint64_t bits) {
	bits = (bits & 0x5555555555555555lu) + (bits >> 1  & 0x5555555555555555lu);
	bits = (bits & 0x3333333333333333lu) + (bits >> 2  & 0x3333333333333333lu);
	bits = (bits & 0x0f0f0f0f0f0f0f0flu) + (bits >> 4  & 0x0f0f0f0f0f0f0f0flu);
	bits = (bits & 0x00ff00ff00ff00fflu) + (bits >> 8  & 0x00ff00ff00ff00fflu);
	bits = (bits & 0x0000ffff0000fffflu) + (bits >> 16 & 0x0000ffff0000fffflu);
	bits = (bits & 0x00000000fffffffflu) + (bits >> 32 & 0x00000000fffffffflu);
	return bits;
}
template <>
AONFP_HOST_DEVICE inline unsigned num_of_bits<uint32_t>(uint32_t bits) {
	bits = (bits & 0x55555555u) + (bits >> 1  & 0x55555555u);
	bits = (bits & 0x33333333u) + (bits >> 2  & 0x33333333u);
	bits = (bits & 0x0f0f0f0fu) + (bits >> 4  & 0x0f0f0f0fu);
	bits = (bits & 0x00ff00ffu) + (bits >> 8  & 0x00ff00ffu);
	bits = (bits & 0x0000ffffu) + (bits >> 16 & 0x0000ffffu);
	return bits;
}
template <>
AONFP_HOST_DEVICE inline unsigned num_of_bits<uint16_t>(uint16_t bits) {
	bits = (bits & 0x5555u) + (bits >> 1 & 0x5555u);
	bits = (bits & 0x3333u) + (bits >> 2 & 0x3333u);
	bits = (bits & 0x0f0fu) + (bits >> 4 & 0x0f0fu);
	bits = (bits & 0x00ffu) + (bits >> 8 & 0x00ffu);
	return bits;
}
template <>
AONFP_HOST_DEVICE inline unsigned num_of_bits<uint8_t >(uint8_t  bits) {
	bits = (bits & 0x55u) + (bits >> 1 & 0x55u);
	bits = (bits & 0x33u) + (bits >> 2 & 0x33u);
	bits = (bits & 0x0fu) + (bits >> 4 & 0x0fu);
	return bits;
}

// get ntz
template <class T>
AONFP_HOST_DEVICE unsigned num_of_training_zero(const T v) {return num_of_bits((v & (-v)) -  1);}

// get nlz
template <class T>
AONFP_HOST_DEVICE inline unsigned num_of_leading_zero(T v);
template <>
AONFP_HOST_DEVICE inline unsigned num_of_leading_zero<uint64_t>(uint64_t v) {
	v = v | (v >> 1 );
	v = v | (v >> 2 );
	v = v | (v >> 4 );
	v = v | (v >> 8 );
	v = v | (v >> 16);
	v = v | (v >> 32);
	return num_of_bits<uint64_t>(~v);
}
template <>
AONFP_HOST_DEVICE inline unsigned num_of_leading_zero<uint32_t>(uint32_t v) {
	v = v | (v >> 1 );
	v = v | (v >> 2 );
	v = v | (v >> 4 );
	v = v | (v >> 8 );
	v = v | (v >> 16);
	return num_of_bits<uint32_t>(~v);
}
template <>
AONFP_HOST_DEVICE inline unsigned num_of_leading_zero<uint16_t>(uint16_t v) {
	v = v | (v >> 1 );
	v = v | (v >> 2 );
	v = v | (v >> 4 );
	v = v | (v >> 8 );
	return num_of_bits<uint16_t>(~v);
}
template <>
AONFP_HOST_DEVICE inline unsigned num_of_leading_zero<uint8_t >(uint8_t  v) {
	v = v | (v >> 1 );
	v = v | (v >> 2 );
	v = v | (v >> 4 );
	return num_of_bits<uint8_t >(~v);
}
} //namespace detail
} //namespace aonfp
#endif
