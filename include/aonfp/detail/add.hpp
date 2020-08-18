#ifndef __AONFP_DETAIL_ADD_HPP__
#define __AONFP_DETAIL_ADD_HPP__
#include <type_traits>
#include "detail.hpp"

namespace aonfp {
namespace detail {

template <class DST_S_EXP_T, class DST_MANTISSA_T, class SRC_S_EXP_T, class SRC_MANTISSA_T>
AONFP_HOST_DEVICE inline void add(DST_S_EXP_T& dst_s_exp, DST_MANTISSA_T& dst_mantissa,
		const SRC_S_EXP_T src_s_exp_a, const SRC_MANTISSA_T src_mantissa_a,
		const SRC_S_EXP_T src_s_exp_b, const SRC_MANTISSA_T src_mantissa_b) {
	const auto exp_a = detail::get_exponent_bitstring(src_s_exp_a);
	const auto exp_b = detail::get_exponent_bitstring(src_s_exp_b);

	using compute_mantissa_t = typename std::conditional<sizeof(DST_MANTISSA_T) >= sizeof(SRC_MANTISSA_T), DST_MANTISSA_T, SRC_MANTISSA_T>::type;

	const auto base_exp = exp_a > exp_b ? exp_a : exp_b;
	const int shift = 2 * base_exp - exp_a - exp_b;

	compute_mantissa_t base_mantissa;
	compute_mantissa_t adding_mantissa;

	bool negative = false;

	if (exp_a > exp_b) {
		base_mantissa = src_mantissa_a;
		adding_mantissa = src_mantissa_b;
	} else if (exp_b > exp_a) {
		base_mantissa = src_mantissa_b;
		adding_mantissa = src_mantissa_a;
	} else {
		base_mantissa = std::max(src_mantissa_a, src_mantissa_b);
		adding_mantissa = std::min(src_mantissa_a, src_mantissa_b);
	}

	adding_mantissa <<= ((sizeof(compute_mantissa_t) - sizeof(SRC_MANTISSA_T)) * 8);
	base_mantissa <<= ((sizeof(compute_mantissa_t) - sizeof(SRC_MANTISSA_T)) * 8);

	adding_mantissa = (adding_mantissa >> 1) | (static_cast<compute_mantissa_t>(1) << (sizeof(compute_mantissa_t) * 8 - 1));
	base_mantissa = (base_mantissa >> 1) | (static_cast<compute_mantissa_t>(1) << (sizeof(compute_mantissa_t) * 8 - 1));

	// For getting carry
	adding_mantissa >>= 1;
	base_mantissa >>= 1;

	adding_mantissa >>= shift;

	const auto sign_a = detail::get_sign_bitstring(src_s_exp_a);
	const auto sign_b = detail::get_sign_bitstring(src_s_exp_b);

	compute_mantissa_t tmp_dst_mantissa;

	if ((sign_a ^ sign_b) != 0) {
		tmp_dst_mantissa = base_mantissa - adding_mantissa;
	} else {
		tmp_dst_mantissa = base_mantissa + adding_mantissa;
	}

	const auto carry = detail::num_of_leading_zero<compute_mantissa_t>(tmp_dst_mantissa);

	dst_mantissa = (tmp_dst_mantissa << (carry + 1)) >> ((sizeof(compute_mantissa_t) - sizeof(DST_MANTISSA_T)) * 8);

	// exponential
	auto dst_exp = (static_cast<long>(base_exp) - detail::get_default_exponent_bias(sizeof(SRC_S_EXP_T) * 8 - 1) + detail::get_default_exponent_bias(sizeof(DST_S_EXP_T) * 8 - 1) - carry + 1);

	if (dst_exp < 0 || ((sign_a ^ sign_b) && src_mantissa_a == src_mantissa_b && exp_a == exp_b)) {
		dst_s_exp = detail::get_zero_sign_exponent_bitstring<DST_S_EXP_T>();
		dst_mantissa = detail::get_zero_mantissa_bitstring<DST_MANTISSA_T>();
		return;
	}

	if (sign_a && sign_b) {
		negative = true;
	} else if (sign_a) {
		if (exp_a > exp_b || (exp_a == exp_b && src_mantissa_a > src_mantissa_b)) {
			negative = true;
		}
	} else if (sign_b) {
		if (exp_b > exp_a || (exp_a == exp_b && src_mantissa_b > src_mantissa_a)) {
			negative = true;
		}
	}

	dst_s_exp = dst_exp;
	if (negative) {
		dst_s_exp |= static_cast<DST_S_EXP_T>(1) << (sizeof(DST_S_EXP_T) * 8 - 1);
	}
}
} // namespace detail
} // namespace aonfp
#endif
