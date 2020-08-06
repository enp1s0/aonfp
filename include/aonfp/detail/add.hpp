#ifndef __AONFP_DETAIL_ADD_HPP__
#define __AONFP_DETAIL_ADD_HPP__
#include <type_traits>
#include "detail.hpp"

namespace aonfp {
namespace detail {
template <class RESULT_T>
struct add_compute_t {using type = RESULT_T;};
template <> struct add_compute_t<uint64_t> {using type = uint64_t;};
template <> struct add_compute_t<uint32_t> {using type = uint64_t;};
template <> struct add_compute_t<uint16_t> {using type = uint32_t;};
template <> struct add_compute_t<uint8_t > {using type = uint16_t;};

template <class DST_S_EXP_T, class DST_MANTISSA_T, class SRC_S_EXP_T, class SRC_MANTISSA_T>
AONFP_HOST_DEVICE inline void add_abs(DST_S_EXP_T& dst_s_exp, DST_MANTISSA_T& dst_mantissa,
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

	base_mantissa = (base_mantissa >> (base_shift - 2)) | (static_cast<compute_mantissa_t>(1) << (sizeof(compute_mantissa_t) * 8 - 2));
	adding_mantissa = (adding_mantissa >> (base_shift - 2)) | (static_cast<compute_mantissa_t>(1) << (sizeof(compute_mantissa_t) * 8 - 2));

	adding_mantissa >>= shift;

	const auto tmp_dst_mantissa = base_mantissa + adding_mantissa;

	const auto mlb2 = static_cast<uint32_t>(tmp_dst_mantissa >> (sizeof(compute_mantissa_t) * 8 - 2));
	const auto a0 = mlb2 & 0b01;
	const auto a1 = mlb2 & 0b10;
	const auto b0 = ((a1 >> 1) < a0) ? 0 : 1;
	const auto b1 = (~(a1)) & 0b10;
	const auto shifted = b0 | b1;

	dst_mantissa = tmp_dst_mantissa << shifted;

	const auto exp_c = base_exp - shifted + 2 + detail::get_default_exponent_bias(sizeof(DST_S_EXP_T) * 8 - 1);
	if (static_cast<DST_S_EXP_T>(exp_c) > static_cast<DST_S_EXP_T>(aonfp::detail::get_max_exponent(sizeof(DST_S_EXP_T) * 8 - 1) << 1)) {
		dst_mantissa = aonfp::detail::get_inf_mantissa_bitstring<DST_MANTISSA_T>();
		dst_s_exp = aonfp::detail::get_inf_sign_exponent_bitstring<DST_S_EXP_T>(0);
		return;
	}
	dst_s_exp = exp_c;
}
} // namespace detail
} // namespace aonfp
#endif
