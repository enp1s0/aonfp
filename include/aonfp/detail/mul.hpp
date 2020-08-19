#ifndef __AONFP_DETAIL_MUL_HPP__
#define __AONFP_DETAIL_MUL_HPP__
#include <type_traits>
#include "detail.hpp"
#include "compose.hpp"
#include "macro.hpp"

namespace aonfp {
namespace detail {
template <class RESULT_T>
struct mul_compute_t {using type = RESULT_T;};
template <> struct mul_compute_t<uint64_t> {using type = uint64_t;};
template <> struct mul_compute_t<uint32_t> {using type = uint64_t;};
template <> struct mul_compute_t<uint16_t> {using type = uint32_t;};
template <> struct mul_compute_t<uint8_t > {using type = uint16_t;};

template <class T>
AONFP_HOST_DEVICE inline typename mul_compute_t<T>::type mul_mantissa(const T mantissa_a, const T mantissa_b, uint32_t &shifted) {
	using c_t = typename mul_compute_t<T>::type;
	const auto w_mantissa_a = (static_cast<c_t>(1) << (sizeof(T) * 8 - 1)) | (static_cast<c_t>(mantissa_a) >> 1);
	const auto w_mantissa_b = (static_cast<c_t>(1) << (sizeof(T) * 8 - 2)) | (static_cast<c_t>(mantissa_b) >> 2);
	const auto w_mantissa_ab = w_mantissa_a * w_mantissa_b;
	const auto mlb2 = static_cast<uint32_t>(w_mantissa_ab >> (sizeof(c_t) * 8 - 2));
	const auto a0 = mlb2 & 0b01;
	const auto a1 = mlb2 & 0b10;
	const auto b0 = ((a1 >> 1) < a0) ? 0 : 1;
	const auto b1 = (~(a1)) & 0b10;
	shifted = b0 | b1;

	return w_mantissa_ab << shifted;
}

#ifndef __CUDA_ARCH__
inline uint64_t __mul64hi(const uint64_t a, const uint64_t b) {
	const auto a_lo = static_cast<uint64_t>(static_cast<uint32_t>(a));
	const auto a_hi = a >> 32;
	const auto b_lo = static_cast<uint64_t>(static_cast<uint32_t>(b));
	const auto b_hi = b >> 32;

	const auto ab_hi = a_hi * b_hi;
	const auto ab_mid = a_hi * b_lo;
	const auto ba_mid = b_hi * a_lo;
	const auto ab_lo = a_lo * b_lo;

	const auto carry_bit = (static_cast<uint64_t>(static_cast<uint32_t>(ab_mid)) + static_cast<uint64_t>(static_cast<uint32_t>(ba_mid)) + (ab_lo >> 32)) >> 32;

	const auto hi = ab_hi + (ab_mid >> 32) + (ba_mid >> 32) + carry_bit;

	return hi;
}
#endif

template <>
AONFP_HOST_DEVICE inline typename mul_compute_t<uint64_t>::type mul_mantissa(const uint64_t mantissa_a, const uint64_t mantissa_b, uint32_t &shifted) {
	const auto w_mantissa_a = (1lu << (sizeof(uint64_t) * 8 - 1)) | (mantissa_a >> 1);
	const auto w_mantissa_b = (1lu << (sizeof(uint64_t) * 8 - 2)) | (mantissa_b >> 2);

	aonfp_uint128_t ab;
	ab.x[0] = w_mantissa_a * w_mantissa_b;
	ab.x[1] = __mul64hi(w_mantissa_a, w_mantissa_b);

	const auto mlb2 = static_cast<uint32_t>(ab.x[1] >> (sizeof(uint64_t) * 8 - 2));
	const auto a0 = mlb2 & 0b01;
	const auto a1 = mlb2 & 0b10;
	const auto b0 = ((a1 >> 1) < a0) ? 0 : 1;
	const auto b1 = (~(a1)) & 0b10;
	shifted = b0 | b1;

	ab.x[1] = (ab.x[1] << shifted) | (ab.x[0] >> (sizeof(uint64_t) * 8 - shifted));

	return ab.x[1];
}

template <class DST_T, class SRC_T>
AONFP_HOST_DEVICE DST_T resize_mantissa(const SRC_T src_mantissa, uint32_t &shifted) {
	using c_t = typename std::conditional<sizeof(SRC_T) >= sizeof(DST_T), SRC_T, DST_T>::type;
	const auto c_src_mantissa = static_cast<c_t>(src_mantissa);

	if (sizeof(DST_T) >= sizeof(SRC_T)) {
		shifted = 0;
		return c_src_mantissa << ((sizeof(DST_T) - sizeof(SRC_T)) * 8);
	} else {
		const auto c = ((c_src_mantissa & (static_cast<c_t>(1) << ((sizeof(SRC_T) - sizeof(DST_T)) * 8 - 1))) >> (sizeof(SRC_T) - sizeof(DST_T) - 1) & 0x1);

		const auto m0 = c_src_mantissa >> ((sizeof(SRC_T) - sizeof(DST_T)) * 8);

		const auto m1 = m0 + c;
		shifted = static_cast<uint32_t>(m1 >> (sizeof(DST_T) * 8));
		return static_cast<DST_T>(m1 >> shifted);
	}
}

template <class DST_S_EXP_T, class DST_MANTISSA_T, class SRC_S_EXP_T, class SRC_MANTISSA_T>
AONFP_HOST_DEVICE inline void mul(DST_S_EXP_T& dst_s_exp, DST_MANTISSA_T& dst_mantissa,
		const SRC_S_EXP_T src_s_exp_a, const SRC_MANTISSA_T src_mantissa_a,
		const SRC_S_EXP_T src_s_exp_b, const SRC_MANTISSA_T src_mantissa_b) {

	uint32_t shifted_0;
	const auto full_mantissa = mul_mantissa(src_mantissa_a, src_mantissa_b, shifted_0);

	uint32_t shifted_1;
	dst_mantissa = resize_mantissa<DST_MANTISSA_T>(full_mantissa, shifted_1);

	const auto exp_a = detail::get_exponent_bitstring(src_s_exp_a) - static_cast<long>(detail::get_default_exponent_bias(sizeof(SRC_S_EXP_T) * 8 - 1));
	const auto exp_b = detail::get_exponent_bitstring(src_s_exp_b) - static_cast<long>(detail::get_default_exponent_bias(sizeof(SRC_S_EXP_T) * 8 - 1));

	const auto exp = exp_a + exp_b - shifted_0 + 3 + shifted_1;

	const auto exponent = static_cast<typename std::make_signed<DST_S_EXP_T>::type>(exp + detail::get_default_exponent_bias(sizeof(DST_S_EXP_T) * 8 - 1));

	if (exponent < 0) {
		dst_mantissa = aonfp::detail::get_zero_mantissa_bitstring<DST_MANTISSA_T>();
		dst_s_exp = aonfp::detail::get_zero_sign_exponent_bitstring<DST_S_EXP_T>(0);
		return;
	} else if (static_cast<DST_S_EXP_T>(exponent) > static_cast<DST_S_EXP_T>(aonfp::detail::get_max_exponent(sizeof(DST_S_EXP_T) * 8 - 1) << 1)) {
		dst_mantissa = aonfp::detail::get_inf_mantissa_bitstring<DST_MANTISSA_T>();
		dst_s_exp = aonfp::detail::get_inf_sign_exponent_bitstring<DST_S_EXP_T>(0);
		return;
	}
	const auto s = (detail::get_sign_bitstring(src_s_exp_a) ^ detail::get_sign_bitstring(src_s_exp_b)) >> (sizeof(SRC_S_EXP_T) * 8 - 1);
	dst_s_exp = exponent | (static_cast<DST_S_EXP_T>(s) << (sizeof(DST_S_EXP_T) * 8 - 1));
}
} // namespace detail
} // namespace aonfp
#endif
