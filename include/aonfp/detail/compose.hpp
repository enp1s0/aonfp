#ifndef __AONFP_DETAIL_COMPOSE_HPP__
#define __AONFP_DETAIL_COMPOSE_HPP__
#include "detail.hpp"
#include "standard_fp.hpp"
#include "macro.hpp"

namespace aonfp {
namespace detail {

// composer / decomposer
template <class T, class S>
AONFP_HOST_DEVICE inline T decompose_mantissa(const S v, int& move_up) {
	constexpr unsigned ieee_mantissa_size = aonfp::detail::standard_fp::get_mantissa_size<S>();
	constexpr unsigned aonfp_mantissa_size = sizeof(T) * 8;

	using ieee_bitstring_t = typename aonfp::detail::bitstring_t<S>::type;
	constexpr auto mantissa_mask = (static_cast<ieee_bitstring_t>(1) << aonfp::detail::standard_fp::get_mantissa_size<S>()) - 1;

	const auto ieee_mantissa_bitstring = ((*reinterpret_cast<const ieee_bitstring_t*>(&v)) & mantissa_mask) << (1 + aonfp::detail::standard_fp::get_exponent_size<S>());
	move_up = 0;

	if (ieee_mantissa_size <= aonfp_mantissa_size) {
		return static_cast<T>(ieee_mantissa_bitstring) << ((sizeof(T) - sizeof(ieee_bitstring_t)) * 8);
	} else {
		// rounding
		constexpr unsigned shift_size = (sizeof(ieee_bitstring_t) - sizeof(T)) * 8;
		const ieee_bitstring_t cut_msb = ieee_mantissa_bitstring & (static_cast<ieee_bitstring_t>(1) << (shift_size - 1));

		const ieee_bitstring_t ma = (ieee_mantissa_bitstring + (cut_msb << 1)) >> shift_size;

		if (ma == 0 && cut_msb != 0) {
			move_up = 1;
		}
		return static_cast<T>(ma);
	}
}

template <class T, class S>
AONFP_HOST_DEVICE inline T decompose_sign_exponent(const S v, const int move_up, uo_flow_t& uo) {
	using ieee_bitstring_t = typename aonfp::detail::bitstring_t<S>::type;
	const auto sign = (*reinterpret_cast<const ieee_bitstring_t*>(&v)) >> (sizeof(ieee_bitstring_t) * 8 - 1);
	const auto exponent = (((*reinterpret_cast<const ieee_bitstring_t*>(&v)) >> aonfp::detail::standard_fp::get_mantissa_size<S>())
			& ((static_cast<ieee_bitstring_t>(1) << aonfp::detail::standard_fp::get_exponent_size<S>()) - 1)) // mantissa mask
		+ move_up;
	const auto res_s = static_cast<T>(sign) << (sizeof(T) * 8 - 1);
	if (!exponent) {
		return res_s;
	}
	const auto src_exponent = static_cast<typename std::make_signed<T>::type>(exponent) - detail::get_default_exponent_bias(aonfp::detail::standard_fp::get_exponent_size<S>());
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
	using ieee_bitstring_t = typename aonfp::detail::bitstring_t<T>::type;

	if (sizeof(MANTISSA_T) * 8 > aonfp::detail::standard_fp::get_mantissa_size<T>()) {
		const auto shifted_m = mantissa >> (sizeof(MANTISSA_T) * 8 - standard_fp::get_mantissa_size<T>());
		const auto s = (mantissa >> (sizeof(MANTISSA_T) * 8 - standard_fp::get_mantissa_size<T>() - 1) & 0x1);
		const auto shifted_m_a = shifted_m + s;
		move_up = (shifted_m_a >> standard_fp::get_mantissa_size<T>());
		const auto full_bitstring = shifted_m_a | (*reinterpret_cast<const MANTISSA_T*>(&src_fp) & (~((static_cast<MANTISSA_T>(1) << standard_fp::get_mantissa_size<T>()) - 1)));
		return *reinterpret_cast<const T*>(&full_bitstring);
	} else {
		const auto shifted_m = static_cast<ieee_bitstring_t>(mantissa) << (aonfp::detail::standard_fp::get_mantissa_size<T>() - sizeof(MANTISSA_T) * 8);
		move_up = 0;
		const auto full_bitstring = shifted_m |
			(*reinterpret_cast<const ieee_bitstring_t*>(&src_fp) & ((~static_cast<ieee_bitstring_t>(0)) - ((static_cast<ieee_bitstring_t>(1) << aonfp::detail::standard_fp::get_mantissa_size<T>()) - 1)));
		return *reinterpret_cast<const T*>(&full_bitstring);
	}
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
