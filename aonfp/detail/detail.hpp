#ifndef __AONFP_DETAIL_DETAIL_HPP__
#define __AONFP_DETAIL_DETAIL_HPP__
#include <cstdint>
namespace aonfp {

namespace detail {

constexpr unsigned get_default_exponent_bias(const unsigned exponent_size) {
	return (1 << (exponent_size - 1)) - 1;
}

template <class T>
constexpr void set_nan_exponent(T& v);
template <> constexpr void set_nan_exponent<uint64_t>(uint64_t& v) {v = 0x7ffffffffffffffflu;}
template <> constexpr void set_nan_exponent<uint32_t>(uint32_t& v) {v = 0x7fffffff;}
template <> constexpr void set_nan_exponent<uint16_t>(uint16_t& v) {v = 0x7fff;}
template <> constexpr void set_nan_exponent<uint8_t >(uint8_t & v) {v = 0x7f;}

template <class T>
constexpr void set_nan_mantissa(T& v);
template <> constexpr void set_nan_mantissa<uint64_t>(uint64_t& v) {v = 0xfffffffffffffffflu;}
template <> constexpr void set_nan_mantissa<uint32_t>(uint32_t& v) {v = 0xffffffff;}
template <> constexpr void set_nan_mantissa<uint16_t>(uint16_t& v) {v = 0xffff;}
template <> constexpr void set_nan_mantissa<uint8_t >(uint8_t & v) {v = 0xff;}

template <class T>
constexpr void set_inf_exponent(T& v);
template <> constexpr void set_inf_exponent<uint64_t>(uint64_t& v) {v = 0x7ffffffffffffffflu | (v & 0x8000000000000000lu);}
template <> constexpr void set_inf_exponent<uint32_t>(uint32_t& v) {v = 0x7fffffff | (v & 0x80000000);}
template <> constexpr void set_inf_exponent<uint16_t>(uint16_t& v) {v = 0x7fff | (v & 0x8000);}
template <> constexpr void set_inf_exponent<uint8_t >(uint8_t & v) {v = 0x7f | (v & 0x80);}

template <class T>
constexpr void set_inf_mantissa(T& v) {v = 0;};

} //namespace detail
} //namespace aonfp
#endif
