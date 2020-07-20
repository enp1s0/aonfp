#ifndef __AONFP_DETAIL_COMPOSE_HPP__
#define __AONFP_DETAIL_COMPOSE_HPP__
#ifndef AONFP_HOST_DEVICE
 #if defined(__CUDA_ARCH__)
  #define AONFP_HOST_DEVICE __device__ __host__
 #else
  #define AONFP_HOST_DEVICE
 #endif
#endif

#include "detail.hpp"
#include "standard_fp.hpp"

namespace aonfp {
namespace detail {

// composer / decomposer
template <class T>
AONFP_HOST_DEVICE inline T decompose_mantissa(const double v, int& move_up);
template <> AONFP_HOST_DEVICE inline uint64_t decompose_mantissa<uint64_t>(const double v, int& move_up) {
	move_up = 0;
	return (*reinterpret_cast<const uint64_t*>(&v)) << 12;
}

template <> AONFP_HOST_DEVICE inline uint32_t decompose_mantissa<uint32_t>(const double v, int& move_up) {
	const auto mantissa_bs = (*reinterpret_cast<const uint64_t*>(&v)) & 0xffffffffffffflu;
	const auto r_s = (mantissa_bs & 0x80000lu) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	move_up = (mantissa_bs_a & 0x10000000000000lu) >> 52;
	return mantissa_bs_a >> 20;
}

template <> AONFP_HOST_DEVICE inline uint16_t decompose_mantissa<uint16_t>(const double v, int& move_up) {
	const auto mantissa_bs = (*reinterpret_cast<const uint64_t*>(&v)) & 0xffffffffffffflu;
	const auto r_s = (mantissa_bs & 0x800000000lu) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	move_up = (mantissa_bs_a & 0x10000000000000lu) >> 52;
	return mantissa_bs_a >> 36;
}

template <> AONFP_HOST_DEVICE inline uint8_t decompose_mantissa<uint8_t>(const double v, int& move_up) {
	const auto mantissa_bs = (*reinterpret_cast<const uint64_t*>(&v)) & 0xffffffffffffflu;
	const auto r_s = (mantissa_bs & 0x80000000000lu) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	move_up = (mantissa_bs_a & 0x10000000000000lu) >> 52;
	return mantissa_bs_a >> 44;
}

template <class T>
AONFP_HOST_DEVICE inline T decompose_mantissa(const float v, int& move_up);
template <> AONFP_HOST_DEVICE inline uint64_t decompose_mantissa<uint64_t>(const float v, int& move_up) {
	move_up = 0;
	return (*reinterpret_cast<const uint64_t*>(&v)) << (9 + 32);
}

template <> AONFP_HOST_DEVICE inline uint32_t decompose_mantissa<uint32_t>(const float v, int& move_up) {
	move_up = 0;
	return (*reinterpret_cast<const uint32_t*>(&v)) << 9;
}

template <> AONFP_HOST_DEVICE inline uint16_t decompose_mantissa<uint16_t>(const float v, int& move_up) {
	const auto mantissa_bs = (*reinterpret_cast<const uint32_t*>(&v)) & 0x7fffff;
	const auto r_s = (mantissa_bs & 0x40) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	move_up = (mantissa_bs_a & 0x800000) >> 23;
	return mantissa_bs_a >> 7;
}

template <> AONFP_HOST_DEVICE inline uint8_t decompose_mantissa<uint8_t>(const float v, int& move_up) {
	const auto mantissa_bs = (*reinterpret_cast<const uint32_t*>(&v)) & 0x7fffff;
	const auto r_s = (mantissa_bs & 0x8000) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	move_up = (mantissa_bs_a & 0x800000) >> 23;
	return mantissa_bs_a >> 15;
}

template <class T>
AONFP_HOST_DEVICE inline T decompose_sign_exponent(const double v, const int move_up, uo_flow_t& uo) {
	const auto sign = (*reinterpret_cast<const uint64_t*>(&v)) & 0x8000000000000000lu;
	const auto exponent = (((*reinterpret_cast<const uint64_t*>(&v)) & 0x7ff0000000000000lu) >> 52) + move_up;
	const auto res_s = static_cast<T>((sign & 0x8000000000000000lu) >> (64 - 8 * sizeof(T)));
	if (!exponent) {
		return res_s;
	}
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
AONFP_HOST_DEVICE inline T decompose_sign_exponent(const float v, const int move_up, uo_flow_t& uo) {
	const auto sign = ((*reinterpret_cast<const uint32_t*>(&v)) & 0x80000000) >> 31;
	const auto exponent = (((*reinterpret_cast<const uint32_t*>(&v)) & 0x7f800000) >> 23) + move_up;
	const auto res_s = static_cast<T>(sign) << (sizeof(T) * 8 - 1);
	if (!exponent) {
		return res_s;
	}
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
AONFP_HOST_DEVICE inline T compose_mantissa(const MANTISSA_T mantissa, const T src_fp, int& move_up) {
	const auto shifted_m = mantissa >> (sizeof(MANTISSA_T) * 8 - standard_fp::get_mantissa_size<T>());
	const auto s = (mantissa >> (sizeof(MANTISSA_T) * 8 - standard_fp::get_mantissa_size<T>() - 1) & 0x1);
	const auto shifted_m_a = shifted_m + s;
	move_up = (shifted_m_a >> standard_fp::get_mantissa_size<T>());
	const auto full_bitstring = shifted_m_a | (*reinterpret_cast<const MANTISSA_T*>(&src_fp) & (~((static_cast<MANTISSA_T>(1) << standard_fp::get_exponent_size<T>()) - 1)));
	return *reinterpret_cast<const T*>(&full_bitstring);
}

template <> AONFP_HOST_DEVICE inline double compose_mantissa<double, uint32_t>(const uint32_t m, const double src_fp, int& move_up) {
	const auto shifted_m = static_cast<uint64_t>(m) << 20;
	move_up = 0;
	const auto full_bitstring = shifted_m | (*reinterpret_cast<const uint64_t*>(&src_fp) & 0xfff0000000000000lu);
	return *reinterpret_cast<const double*>(&full_bitstring);
}

template <> AONFP_HOST_DEVICE inline double compose_mantissa<double, uint16_t>(const uint16_t m, const double src_fp, int& move_up) {
	const auto shifted_m = static_cast<uint64_t>(m) << 36;
	move_up = 0;
	const auto full_bitstring = shifted_m | (*reinterpret_cast<const uint64_t*>(&src_fp) & 0xfff0000000000000lu);
	return *reinterpret_cast<const double*>(&full_bitstring);
}

template <> AONFP_HOST_DEVICE inline double compose_mantissa<double, uint8_t>(const uint8_t m, const double src_fp, int& move_up) {
	const auto shifted_m = static_cast<uint64_t>(m) << 44;
	move_up = 0;
	const auto full_bitstring = shifted_m | (*reinterpret_cast<const uint64_t*>(&src_fp) & 0xfff0000000000000lu);
	return *reinterpret_cast<const double*>(&full_bitstring);
}

template <> AONFP_HOST_DEVICE inline float compose_mantissa<float , uint16_t>(const uint16_t m, const float src_fp, int& move_up) {
	const auto shifted_m = static_cast<uint32_t>(m) << 7;
	move_up = 0;
	const auto full_bitstring = shifted_m | (*reinterpret_cast<const uint32_t*>(&src_fp) & 0xff800000);
	return *reinterpret_cast<const float*>(&full_bitstring);
}

template <> AONFP_HOST_DEVICE inline float compose_mantissa<float , uint8_t>(const uint8_t m, const float src_fp, int& move_up) {
	const auto shifted_m = static_cast<uint32_t>(m) << 15;
	move_up = 0;
	const auto full_bitstring = shifted_m | (*reinterpret_cast<const uint32_t*>(&src_fp) & 0xff800000);
	return *reinterpret_cast<const float*>(&full_bitstring);
}


template <class T, class S_EXP_T>
AONFP_HOST_DEVICE inline T compose_sign_exponent(const S_EXP_T s_exp, const T src_fp, const int move_up) {
	const auto exponent_bitstring = get_exponent_bitstring(s_exp);
	auto dst_exponent = static_cast<typename bitstring_t<T>::type>(0);
	if (exponent_bitstring) {
		const auto exponent = static_cast<typename std::make_signed<S_EXP_T>::type>(static_cast<typename std::make_signed<S_EXP_T>::type>(exponent_bitstring) - get_default_exponent_bias(sizeof(S_EXP_T) * 8 - 1) + move_up);
		if (exponent >= get_max_exponent(standard_fp::get_exponent_size<T>())) {
			return standard_fp::get_infinity<T>(src_fp);
		}
		if (exponent <= get_min_exponent(standard_fp::get_exponent_size<T>())) {
			return static_cast<T>(0);
		}
		dst_exponent = static_cast<typename bitstring_t<T>::type>(exponent + get_default_exponent_bias(standard_fp::get_exponent_size<T>()));
	}
	const auto dst_exponent_bitstring = dst_exponent << standard_fp::get_mantissa_size<T>();

	const auto dst_sign_bitstring = static_cast<typename bitstring_t<T>::type>(get_sign_bitstring(s_exp) >> (sizeof(S_EXP_T) * 8 - 1)) << (sizeof(T) * 8 - 1);

	const auto dst_mantissa_bitstring = *reinterpret_cast<const typename bitstring_t<T>::type*>(&src_fp) & ((static_cast<typename bitstring_t<T>::type>(1) << standard_fp::get_mantissa_size<T>()) - 1);

	const auto full_bitstring = dst_sign_bitstring | dst_exponent_bitstring | dst_mantissa_bitstring;

	return *reinterpret_cast<const T*>(&full_bitstring);
}

} // namespace detail
} // namespace aonfp

#endif
