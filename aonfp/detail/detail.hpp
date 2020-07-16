#ifndef __AONFP_DETAIL_DETAIL_HPP__
#define __AONFP_DETAIL_DETAIL_HPP__
#include <cstdint>
#include <type_traits>
namespace aonfp {

namespace detail {

enum uo_flow_t {
	uo_flow_non,
	uo_flow_overflow,
	uo_flow_underflow,
};

constexpr unsigned get_default_exponent_bias(const unsigned exponent_size) {
	return (1 << (exponent_size - 1)) - 1;
}

template <class T>
constexpr T get_nan_exponent(T v);
template <> constexpr uint64_t get_nan_exponent<uint64_t>(uint64_t v) {return 0x7ffffffffffffffflu;}
template <> constexpr uint32_t get_nan_exponent<uint32_t>(uint32_t v) {return 0x7fffffff;}
template <> constexpr uint16_t get_nan_exponent<uint16_t>(uint16_t v) {return 0x7fff;}
template <> constexpr uint8_t  get_nan_exponent<uint8_t >(uint8_t  v) {return 0x7f;}

template <class T>
constexpr T get_nan_mantissa(T v);
template <> constexpr uint64_t get_nan_mantissa<uint64_t>(uint64_t v) {return 0xfffffffffffffffflu;}
template <> constexpr uint32_t get_nan_mantissa<uint32_t>(uint32_t v) {return 0xffffffff;}
template <> constexpr uint16_t get_nan_mantissa<uint16_t>(uint16_t v) {return 0xffff;}
template <> constexpr uint8_t  get_nan_mantissa<uint8_t >(uint8_t  v) {return 0xff;}

template <class T>
constexpr T get_inf_exponent(T v);
template <> constexpr uint64_t get_inf_exponent<uint64_t>(uint64_t v) {return 0x7ffffffffffffffflu | (v & 0x8000000000000000lu);}
template <> constexpr uint32_t get_inf_exponent<uint32_t>(uint32_t v) {return 0x7fffffff | (v & 0x80000000);}
template <> constexpr uint16_t get_inf_exponent<uint16_t>(uint16_t v) {return 0x7fff | (v & 0x8000);}
template <> constexpr uint8_t  get_inf_exponent<uint8_t >(uint8_t  v) {return 0x7f | (v & 0x80);}

template <class T>
constexpr T get_inf_mantissa() {return static_cast<T>(0);};

template <class T>
constexpr T get_zero_exponent(T v);
template <> constexpr uint64_t get_zero_exponent<uint64_t>(uint64_t v) {return 0x8000000000000000lu & v;}
template <> constexpr uint32_t get_zero_exponent<uint32_t>(uint32_t v) {return 0x80000000 & v;}
template <> constexpr uint16_t get_zero_exponent<uint16_t>(uint16_t v) {return 0x8000 & v;}
template <> constexpr uint8_t  get_zero_exponent<uint8_t >(uint8_t  v) {return 0x80 & v;}

template <class T>
constexpr T get_zero_mantissa() {return static_cast<T>(0);};

template <class T>
inline T copy_mantissa(const double v, int& digit_up);
template <> inline uint64_t copy_mantissa<uint64_t>(const double v, int& digit_up) {
	digit_up = 0;
	return (*reinterpret_cast<const uint64_t*>(&v)) << 12;
}
template <> inline uint32_t copy_mantissa<uint32_t>(const double v, int& digit_up) {
	const auto mantissa_bs = (*reinterpret_cast<const uint64_t*>(&v)) & 0xffffffffffffflu;
	const auto r_s = (mantissa_bs & 0x80000lu) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	digit_up = (mantissa_bs_a & 0x10000000000000lu) >> 52;
	return mantissa_bs_a >> (20 + digit_up);
}
template <> inline uint16_t copy_mantissa<uint16_t>(const double v, int& digit_up) {
	const auto mantissa_bs = (*reinterpret_cast<const uint64_t*>(&v)) & 0xffffffffffffflu;
	const auto r_s = (mantissa_bs & 0x800000000lu) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	digit_up = (mantissa_bs_a & 0x10000000000000lu) >> 52;
	return mantissa_bs_a >> (36 + digit_up);
}
template <> inline uint8_t copy_mantissa<uint8_t>(const double v, int& digit_up) {
	const auto mantissa_bs = (*reinterpret_cast<const uint64_t*>(&v)) & 0xffffffffffffflu;
	const auto r_s = (mantissa_bs & 0x80000000000lu) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	digit_up = (mantissa_bs_a & 0x10000000000000lu) >> 52;
	return mantissa_bs_a >> (44 + digit_up);
}

template <class T>
inline T copy_mantissa(const float v, int& digit_up);
template <> inline uint64_t copy_mantissa<uint64_t>(const float v, int& digit_up) {
	digit_up = 0;
	return (*reinterpret_cast<const uint64_t*>(&v)) << (9 + 32);
}
template <> inline uint32_t copy_mantissa<uint32_t>(const float v, int& digit_up) {
	digit_up = 0;
	return (*reinterpret_cast<const uint32_t*>(&v)) << 9;
}
template <> inline uint16_t copy_mantissa<uint16_t>(const float v, int& digit_up) {
	const auto mantissa_bs = (*reinterpret_cast<const uint32_t*>(&v)) & 0x7fffff;
	const auto r_s = (mantissa_bs & 0x40) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	digit_up = (mantissa_bs_a & 0x800000) >> 23;
	return mantissa_bs_a >> (7 + digit_up);
}
template <> inline uint8_t copy_mantissa<uint8_t>(const float v, int& digit_up) {
	const auto mantissa_bs = (*reinterpret_cast<const uint32_t*>(&v)) & 0x7fffff;
	const auto r_s = (mantissa_bs & 0x8000) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	digit_up = (mantissa_bs_a & 0x800000) >> 23;
	return mantissa_bs_a >> (15 + digit_up);
}

template <class T>
inline T copy_sign_exponent(const double v, uo_flow_t& uo) {
	const auto sign = (*reinterpret_cast<const uint64_t*>(&v)) & 0x8000000000000000lu;
	const auto exponent = ((*reinterpret_cast<const uint64_t*>(&v)) & 0x7ff0000000000000lu) >> 52;
	const auto res_s = static_cast<T>((sign & 0x8000000000000000lu) >> (64 - 8 * sizeof(T)));
	const auto src_exponent = static_cast<typename std::make_signed<T>::type>(exponent) - detail::get_default_exponent_bias(11);
	const auto dst_exponent = static_cast<T>(src_exponent + detail::get_default_exponent_bias(sizeof(T) * 8 - 1));
	if (dst_exponent >> (sizeof(T) * 8 - 1)) {
		if (src_exponent >> (sizeof(T) * 8 - 1)) {
			uo = uo_flow_underflow;
		} else {
			uo = uo_flow_underflow;
		}
	}
	return res_s | dst_exponent;
}

} //namespace detail
} //namespace aonfp
#endif
