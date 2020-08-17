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
	constexpr auto base_shift = 2 * sizeof(compute_mantissa_t) - sizeof(DST_MANTISSA_T) - sizeof(SRC_MANTISSA_T);

	const auto base_exp = exp_a > exp_b ? exp_a : exp_b;
	const int shift = 2 * base_exp - exp_a - exp_b;

	compute_mantissa_t base_mantissa;
	compute_mantissa_t adding_mantissa;

	if (exp_a > exp_b) {
		base_mantissa = src_mantissa_a;
		adding_mantissa = src_mantissa_b;
	} else {
		base_mantissa = src_mantissa_b;
		adding_mantissa = src_mantissa_a;
	}

	adding_mantissa = (adding_mantissa >> 1) | (static_cast<compute_mantissa_t>(1) << (sizeof(compute_mantissa_t) * 8 - 1));
	base_mantissa = (base_mantissa >> 1) | (static_cast<compute_mantissa_t>(1) << (sizeof(compute_mantissa_t) * 8 - 1));

	// For getting carry
	adding_mantissa >>= 1;
	base_mantissa >>= 1;

	std::printf("shift = %d\n", shift);
	adding_mantissa >>= shift;

	const auto sign_a = detail::get_sign_bitstring(src_s_exp_a);
	const auto sign_b = detail::get_sign_bitstring(src_s_exp_b);

	compute_mantissa_t tmp_dst_mantissa;

	aonfp::detail::utils::print_bin(base_mantissa, true);
	aonfp::detail::utils::print_bin(adding_mantissa, true);

	if ((sign_a ^ sign_b) != 0) {
		tmp_dst_mantissa = base_mantissa - adding_mantissa;
	} else {
		tmp_dst_mantissa = base_mantissa + adding_mantissa;
	}

	const auto carry = tmp_dst_mantissa >> (sizeof(compute_mantissa_t) * 8 - 1);

	dst_mantissa = tmp_dst_mantissa << (2 - carry);

	// exponential
	if ((sign_a ^ sign_b) != 0) {
	} else {
	
	}
}
} // namespace detail
} // namespace aonfp
#endif
