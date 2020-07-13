#ifndef __AONFP_DETAIL_DETAIL_HPP__
#define __AONFP_DETAIL_DETAIL_HPP__
#include <cstdint>
namespace aonfp {

namespace detail {

constexpr unsigned get_default_exponent_bias(const unsigned exponent_size) {
	return (1 << (exponent_size - 1)) - 1;
}

template <class T>
constexpr T get_nan_exponent(T& v);
template <> constexpr uint64_t get_nan_exponent<uint64_t>(uint64_t& v) {return 0x7ffffffffffffffflu;}
template <> constexpr uint32_t get_nan_exponent<uint32_t>(uint32_t& v) {return 0x7fffffff;}
template <> constexpr uint16_t get_nan_exponent<uint16_t>(uint16_t& v) {return 0x7fff;}
template <> constexpr uint8_t  get_nan_exponent<uint8_t >(uint8_t & v) {return 0x7f;}

template <class T>
constexpr T get_nan_mantissa(T& v);
template <> constexpr uint64_t get_nan_mantissa<uint64_t>(uint64_t& v) {return 0xfffffffffffffffflu;}
template <> constexpr uint32_t get_nan_mantissa<uint32_t>(uint32_t& v) {return 0xffffffff;}
template <> constexpr uint16_t get_nan_mantissa<uint16_t>(uint16_t& v) {return 0xffff;}
template <> constexpr uint8_t  get_nan_mantissa<uint8_t >(uint8_t & v) {return 0xff;}

template <class T>
constexpr T get_inf_exponent(T& v);
template <> constexpr uint64_t get_inf_exponent<uint64_t>(uint64_t& v) {return 0x7ffffffffffffffflu | (v & 0x8000000000000000lu);}
template <> constexpr uint32_t get_inf_exponent<uint32_t>(uint32_t& v) {return 0x7fffffff | (v & 0x80000000);}
template <> constexpr uint16_t get_inf_exponent<uint16_t>(uint16_t& v) {return 0x7fff | (v & 0x8000);}
template <> constexpr uint8_t  get_inf_exponent<uint8_t >(uint8_t & v) {return 0x7f | (v & 0x80);}

template <class T>
constexpr void get_inf_mantissa(T& v) {return static_cast<T>(0);};

template <class T>
inline void copy_mantissa(T& mantissa, const double v);
template <> inline void copy_mantissa<uint64_t>(uint64_t& mantissa, const double v) {
	mantissa = (*reinterpret_cast<const uint64_t*>(&v)) << 12;
}
template <> inline void copy_mantissa<uint32_t>(uint32_t& mantissa, const double v) {
	const auto mantissa_bs = (*reinterpret_cast<const uint64_t*>(&v)) & 0xffffffffffffflu;
	const auto r_s = (mantissa_bs & 0x80000lu) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	const auto digit_up = (mantissa_bs_a & 0x10000000000000lu) >> 52;
	mantissa = mantissa_bs_a >> (20 + digit_up);
}
template <> inline void copy_mantissa<uint16_t>(uint16_t& mantissa, const double v) {
	const auto mantissa_bs = (*reinterpret_cast<const uint64_t*>(&v)) & 0xffffffffffffflu;
	const auto r_s = (mantissa_bs & 0x800000000lu) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	const auto digit_up = (mantissa_bs_a & 0x10000000000000lu) >> 52;
	mantissa = mantissa_bs_a >> (36 + digit_up);
}
template <> inline void copy_mantissa<uint8_t>(uint8_t& mantissa, const double v) {
	const auto mantissa_bs = (*reinterpret_cast<const uint64_t*>(&v)) & 0xffffffffffffflu;
	const auto r_s = (mantissa_bs & 0x80000000000lu) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	const auto digit_up = (mantissa_bs_a & 0x10000000000000lu) >> 52;
	mantissa = mantissa_bs_a >> (44 + digit_up);
}

template <class T>
inline void copy_mantissa(T& mantissa, const float v);
template <> inline void copy_mantissa<uint64_t>(uint64_t& mantissa, const float v) {
	mantissa = (*reinterpret_cast<const uint64_t*>(&v)) << (9 + 32);
}
template <> inline void copy_mantissa<uint32_t>(uint32_t& mantissa, const float v) {
	mantissa = (*reinterpret_cast<const uint32_t*>(&v)) << 9;
}
template <> inline void copy_mantissa<uint16_t>(uint16_t& mantissa, const float v) {
	const auto mantissa_bs = (*reinterpret_cast<const uint32_t*>(&v)) & 0x7fffff;
	const auto r_s = (mantissa_bs & 0x40) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	const auto digit_up = (mantissa_bs_a & 0x800000) >> 23;
	mantissa = mantissa_bs_a >> (7 + digit_up);
}
template <> inline void copy_mantissa<uint8_t>(uint8_t& mantissa, const float v) {
	const auto mantissa_bs = (*reinterpret_cast<const uint32_t*>(&v)) & 0x7fffff;
	const auto r_s = (mantissa_bs & 0x8000) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	const auto digit_up = (mantissa_bs_a & 0x800000) >> 23;
	mantissa = mantissa_bs_a >> (15 + digit_up);
}

} //namespace detail
} //namespace aonfp
#endif
