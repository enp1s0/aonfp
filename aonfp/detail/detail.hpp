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

template <class T>
struct bitstring_t {using type = T;};
template <> struct bitstring_t<float > {using type = uint32_t;};
template <> struct bitstring_t<double> {using type = uint64_t;};

template <class T>
union bitstring_union {
	T fp;
	bitstring_t<T> bitstring;
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
inline T copy_mantissa(const double v, int& move_up);
template <> inline uint64_t copy_mantissa<uint64_t>(const double v, int& move_up) {
	move_up = 0;
	return (*reinterpret_cast<const uint64_t*>(&v)) << 12;
}
template <> inline uint32_t copy_mantissa<uint32_t>(const double v, int& move_up) {
	const auto mantissa_bs = (*reinterpret_cast<const uint64_t*>(&v)) & 0xffffffffffffflu;
	const auto r_s = (mantissa_bs & 0x80000lu) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	move_up = (mantissa_bs_a & 0x10000000000000lu) >> 52;
	return mantissa_bs_a >> 20;
}
template <> inline uint16_t copy_mantissa<uint16_t>(const double v, int& move_up) {
	const auto mantissa_bs = (*reinterpret_cast<const uint64_t*>(&v)) & 0xffffffffffffflu;
	const auto r_s = (mantissa_bs & 0x800000000lu) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	move_up = (mantissa_bs_a & 0x10000000000000lu) >> 52;
	return mantissa_bs_a >> 36;
}
template <> inline uint8_t copy_mantissa<uint8_t>(const double v, int& move_up) {
	const auto mantissa_bs = (*reinterpret_cast<const uint64_t*>(&v)) & 0xffffffffffffflu;
	const auto r_s = (mantissa_bs & 0x80000000000lu) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	move_up = (mantissa_bs_a & 0x10000000000000lu) >> 52;
	return mantissa_bs_a >> 44;
}

template <class T>
inline T copy_mantissa(const float v, int& move_up);
template <> inline uint64_t copy_mantissa<uint64_t>(const float v, int& move_up) {
	move_up = 0;
	return (*reinterpret_cast<const uint64_t*>(&v)) << (9 + 32);
}
template <> inline uint32_t copy_mantissa<uint32_t>(const float v, int& move_up) {
	move_up = 0;
	return (*reinterpret_cast<const uint32_t*>(&v)) << 9;
}
template <> inline uint16_t copy_mantissa<uint16_t>(const float v, int& move_up) {
	const auto mantissa_bs = (*reinterpret_cast<const uint32_t*>(&v)) & 0x7fffff;
	const auto r_s = (mantissa_bs & 0x40) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	move_up = (mantissa_bs_a & 0x800000) >> 23;
	return mantissa_bs_a >> 7;
}
template <> inline uint8_t copy_mantissa<uint8_t>(const float v, int& move_up) {
	const auto mantissa_bs = (*reinterpret_cast<const uint32_t*>(&v)) & 0x7fffff;
	const auto r_s = (mantissa_bs & 0x8000) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	move_up = (mantissa_bs_a & 0x800000) >> 23;
	return mantissa_bs_a >> 15;
}

template <class T>
inline T copy_sign_exponent(const double v, const int move_up, uo_flow_t& uo) {
	const auto sign = (*reinterpret_cast<const uint64_t*>(&v)) & 0x8000000000000000lu;
	const auto exponent = (((*reinterpret_cast<const uint64_t*>(&v)) & 0x7ff0000000000000lu) >> 52) + move_up;
	const auto res_s = static_cast<T>((sign & 0x8000000000000000lu) >> (64 - 8 * sizeof(T)));
	const auto src_exponent = static_cast<typename std::make_signed<T>::type>(exponent) - detail::get_default_exponent_bias(11);
	const auto dst_exponent = static_cast<T>(src_exponent + detail::get_default_exponent_bias(sizeof(T) * 8 - 1));
	if (dst_exponent >> (sizeof(T) * 8 - 1)) {
		if (src_exponent > 0) {
			uo = uo_flow_overflow;
		} else {
			uo = uo_flow_underflow;
		}
	}
	uo = uo_flow_non;
	return res_s | dst_exponent;
}

template <class T>
inline T copy_sign_exponent(const float v, const int move_up, uo_flow_t& uo) {
	const auto sign = (*reinterpret_cast<const uint32_t*>(&v)) & 0x80000000lu;
	const auto exponent = (((*reinterpret_cast<const uint32_t*>(&v)) & 0x7f800000lu) >> 23) + move_up;
	const auto res_s = static_cast<T>((sign & 0x80000000lu) >> (32 - 8 * sizeof(T)));
	const auto src_exponent = static_cast<typename std::make_signed<T>::type>(exponent) - detail::get_default_exponent_bias(8);
	const auto dst_exponent = static_cast<T>(src_exponent + detail::get_default_exponent_bias(sizeof(T) * 8 - 1));
	if (dst_exponent >> (sizeof(T) * 8 - 1)) {
		if (src_exponent > 0) {
			uo = uo_flow_overflow;
		} else {
			uo = uo_flow_underflow;
		}
	}
	uo = uo_flow_non;
	return res_s | dst_exponent;
}

template <class T, class MANTISSA_T>
inline T copy_mantissa(const MANTISSA_T mantissa, int& move_up);

template <> inline double copy_mantissa<double, uint64_t>(const uint64_t m, int& move_up) {
	const auto shifted_m = m >> 12;
	const auto s = (m & 0x800lu) >> 11;
	const auto shifted_m_a = shifted_m + s;
	move_up = (shifted_m_a >> 52);
	return *reinterpret_cast<const double*>(&shifted_m_a);
}

template <> inline double copy_mantissa<double, uint32_t>(const uint32_t m, int& move_up) {
	const auto shifted_m = static_cast<uint64_t>(m) << 20;
	move_up = 0;
	return *reinterpret_cast<const double*>(&shifted_m);
}

template <> inline double copy_mantissa<double, uint16_t>(const uint16_t m, int& move_up) {
	const auto shifted_m = static_cast<uint64_t>(m) << 36;
	move_up = 0;
	return *reinterpret_cast<const double*>(&shifted_m);
}

template <> inline double copy_mantissa<double, uint8_t>(const uint8_t m, int& move_up) {
	const auto shifted_m = static_cast<uint64_t>(m) << 44;
	move_up = 0;
	return *reinterpret_cast<const double*>(&shifted_m);
}

template <> inline float copy_mantissa<float , uint64_t>(const uint64_t m, int& move_up) {
	const auto shifted_m = static_cast<uint32_t>(m >> 41);
	const auto s = static_cast<uint32_t>((m & 0x10000000000lu) >> 40);
	const auto shifted_m_a = shifted_m + s;
	move_up = (shifted_m_a >> 23);
	return *reinterpret_cast<const float*>(&shifted_m_a);
}

template <> inline float copy_mantissa<float , uint32_t>(const uint32_t m, int& move_up) {
	const auto shifted_m = m >> 9;
	const auto s = (m & 0x80) >> 8;
	const auto shifted_m_a = shifted_m + s;
	move_up = (shifted_m_a >> 23);
	return *reinterpret_cast<const float*>(&shifted_m_a);
}

template <> inline float copy_mantissa<float , uint16_t>(const uint16_t m, int& move_up) {
	const auto shifted_m = static_cast<uint32_t>(m) << 7;
	move_up = 0;
	return *reinterpret_cast<const float*>(&shifted_m);
}

template <> inline float copy_mantissa<float , uint8_t>(const uint8_t m, int& move_up) {
	const auto shifted_m = static_cast<uint32_t>(m) << 15;
	move_up = 0;
	return *reinterpret_cast<const float*>(&shifted_m);
}

} //namespace detail
} //namespace aonfp
#endif
