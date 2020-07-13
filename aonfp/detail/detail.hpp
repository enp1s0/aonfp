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

} //namespace detail
} //namespace aonfp
#endif
