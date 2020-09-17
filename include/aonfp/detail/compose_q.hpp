#ifndef __AONFP_DETAIL_COMPOSE_Q_HPP__
#define __AONFP_DETAIL_COMPOSE_Q_HPP__

#include "detail.hpp"
#include "standard_fp.hpp"
#include "macro.hpp"

namespace aonfp {
namespace detail {
namespace q {
// composer / decomposer
template <class T, class S>
AONFP_HOST_DEVICE inline T decompose_sign_mantissa(const S v, int& move_up) {
	constexpr unsigned ieee_mantissa_size = aonfp::detail::standard_fp::get_mantissa_size<S>();
	constexpr unsigned aonfp_mantissa_size = sizeof(T) * 8;

	using ieee_bitstring_t = typename aonfp::detail::bitstring_t<S>::type;
	constexpr auto mantissa_mask = (static_cast<ieee_bitstring_t>(1) << aonfp::detail::standard_fp::get_mantissa_size<S>()) - 1;

	const auto ieee_bitstring = *reinterpret_cast<const ieee_bitstring_t*>(&v);
	const auto ieee_mantissa_bitstring = (ieee_bitstring & mantissa_mask) << (1 + aonfp::detail::standard_fp::get_exponent_size<S>());
	move_up = 0;

	T result_mantissa, result_sign;
	if (ieee_mantissa_size <= aonfp_mantissa_size) {
		result_mantissa = (static_cast<T>(ieee_mantissa_bitstring) << (aonfp_mantissa_size - sizeof(ieee_bitstring_t) * 8)) >> 1;
		result_sign = static_cast<T>(ieee_bitstring >> (sizeof(ieee_bitstring_t) * 8 - 1)) << (sizeof(T) * 8 - 1);
	} else {
		// rounding
		constexpr unsigned shift_size = (sizeof(ieee_bitstring_t) - sizeof(T)) * 8 + 1;
		const ieee_bitstring_t cut_msb = ieee_mantissa_bitstring & (static_cast<ieee_bitstring_t>(1) << (shift_size - 1));

		const ieee_bitstring_t ma = (ieee_mantissa_bitstring + (cut_msb << 1)) >> shift_size;

		if (ma == 0 && cut_msb != 0) {
			move_up = 1;
		}
		result_mantissa = static_cast<T>(ma);
		result_sign = static_cast<T>(ieee_bitstring >> (sizeof(ieee_bitstring_t) * 8 - 1)) << (sizeof(T) * 8 - 1);
	}
	return result_mantissa | result_sign;
}

template <class T, class S>
AONFP_HOST_DEVICE inline T decompose_exponent(const S v, const int move_up, uo_flow_t& uo) {
	using ieee_bitstring_t = typename aonfp::detail::bitstring_t<S>::type;
	const auto exponent = (((*reinterpret_cast<const ieee_bitstring_t*>(&v)) >> aonfp::detail::standard_fp::get_mantissa_size<S>())
			& ((static_cast<ieee_bitstring_t>(1) << aonfp::detail::standard_fp::get_exponent_size<S>()) - 1)) // mantissa mask
		+ move_up;
	if (!exponent) {
		return 0;
	}
	const auto src_exponent = static_cast<typename std::make_signed<T>::type>(exponent) - detail::get_default_exponent_bias(aonfp::detail::standard_fp::get_exponent_size<S>());
	const T dst_exponent = aonfp::detail::get_max<T>() + src_exponent;
	if (src_exponent > 0) {
		uo = uo_flow_overflow;
	}
	uo = uo_flow_non;
	return dst_exponent;
}

template <class T, class S_MANTISSA_T>
AONFP_HOST_DEVICE inline T compose_sign_mantissa(const S_MANTISSA_T mantissa_q, const T src_fp, int& move_up) {
	using ieee_bitstring_t = typename aonfp::detail::bitstring_t<T>::type;
	const auto mantissa = mantissa_q << 1;
	const auto sign = mantissa_q >> (sizeof(S_MANTISSA_T) * 8 - 1);

	// src exp
	constexpr auto src_mantissa_size = aonfp::detail::standard_fp::get_mantissa_size<T>();
	const auto src_exp = (((*reinterpret_cast<const ieee_bitstring_t*>(&src_fp)) << 1) >> (1 + src_mantissa_size) << src_mantissa_size);

	ieee_bitstring_t full_bitstring;
	if (sizeof(S_MANTISSA_T) * 8 > aonfp::detail::standard_fp::get_mantissa_size<T>()) {
		const auto shifted_m = mantissa >> (sizeof(S_MANTISSA_T) * 8 - standard_fp::get_mantissa_size<T>());
		const auto s = (mantissa >> (sizeof(S_MANTISSA_T) * 8 - 1 - standard_fp::get_mantissa_size<T>()) & 0x1);
		const auto shifted_m_a = shifted_m + s;
		move_up = (shifted_m_a >> standard_fp::get_mantissa_size<T>());
		full_bitstring = shifted_m_a | src_exp | ((static_cast<ieee_bitstring_t>(sign) << (sizeof(ieee_bitstring_t) * 8 - 1)));
	} else {
		const auto shifted_m = static_cast<ieee_bitstring_t>(mantissa) << (aonfp::detail::standard_fp::get_mantissa_size<T>() - sizeof(S_MANTISSA_T) * 8);
		move_up = 0;
		full_bitstring = shifted_m | src_exp | ((static_cast<ieee_bitstring_t>(sign) << (sizeof(ieee_bitstring_t) * 8 - 1)));
	}
	return *reinterpret_cast<const T*>(&full_bitstring);
}

template <class T, class EXP_T>
AONFP_HOST_DEVICE inline T compose_exponent(const EXP_T ex, const T src_fp, const int move_up) {
	if (ex == 0) {
		return static_cast<T>(0);
	}
	using ieee_bitstring_t = typename bitstring_t<T>::type;
	const auto e = static_cast<long>(ex) - aonfp::detail::get_max<EXP_T>() - move_up;
	auto dst_exponent = static_cast<typename bitstring_t<T>::type>(0);
	if (e >= -(1 << static_cast<int>(aonfp::detail::standard_fp::get_exponent_size<T>()))) {
		const auto bias = aonfp::detail::get_default_exponent_bias(aonfp::detail::standard_fp::get_exponent_size<T>());
		dst_exponent = e + bias;
	}

	const auto dst_exponent_bitstring = static_cast<ieee_bitstring_t>(dst_exponent) << standard_fp::get_mantissa_size<T>();

	const auto dst_mantissa_bitstring = *reinterpret_cast<const typename bitstring_t<T>::type*>(&src_fp) &
		(((static_cast<ieee_bitstring_t>(1) << standard_fp::get_mantissa_size<T>()) - 1) | (static_cast<ieee_bitstring_t>(1) << (sizeof(ieee_bitstring_t) * 8 - 1)));

	const auto full_bitstring = dst_exponent_bitstring | dst_mantissa_bitstring;

	return *reinterpret_cast<const T*>(&full_bitstring);
}
} // namespace q
} // namespace detail
} // namespace aonfp

#endif
