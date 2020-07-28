#ifndef __AONFP_DETAIL_OPERATORS_HPP__
#define __AONFP_DETAIL_OPERATORS_HPP__
#include "../aonfp.hpp"
#include "detail.hpp"
#include "compose.hpp"
namespace aonfp {
namespace detail {
struct aonfp_uint128_t {
	uint64_t x[2];
};
template <class RESULT_T>
struct mul_compute_t {using type = RESULT_T;};
template <> struct mul_compute_t<uint64_t> {using type = aonfp_uint128_t;};
template <> struct mul_compute_t<uint32_t> {using type = uint64_t;};
template <> struct mul_compute_t<uint16_t> {using type = uint32_t;};
template <> struct mul_compute_t<uint8_t > {using type = uint16_t;};

template <class T>
AONFP_HOST_DEVICE typename mul_compute_t<T>::type mul_mantissa(const T mantissa_a, const T mantissa_b, uint32_t &shifted) {
	using c_t = typename mul_compute_t<T>::type;
	const auto w_mantissa_a = (static_cast<c_t>(1) << (sizeof(T) * 8 - 1)) | (static_cast<c_t>(mantissa_a) >> 1);
	const auto w_mantissa_b = (static_cast<c_t>(1) << (sizeof(T) * 8 - 2)) | (static_cast<c_t>(mantissa_b) >> 2);
	const auto w_mantissa_ab = w_mantissa_a * w_mantissa_b;
	const auto mlb2 = static_cast<uint32_t>(w_mantissa_ab >> (sizeof(c_t) * 8 - 2));
	const auto a0 = mlb2 & 0b01;
	const auto a1 = mlb2 & 0b10;
	const auto b0 = ((a1 >> 1) < a0) ? 0 : 1;
	const auto b1 = ~a1;
	shifted = b0 | (b1 << 1);

	return w_mantissa_ab << shifted;
}

#ifndef __CUDA_ARCH__
uint64_t __mul64hi(const uint64_t a, const uint64_t b) {
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
AONFP_HOST_DEVICE typename mul_compute_t<uint64_t>::type mul_mantissa(const uint64_t mantissa_a, const uint64_t mantissa_b, uint32_t &shifted) {
	const auto w_mantissa_a = (1lu << (sizeof(uint64_t) * 8 - 1)) | (mantissa_a >> 1);
	const auto w_mantissa_b = (1lu << (sizeof(uint64_t) * 8 - 2)) | (mantissa_a >> 2);

	mul_compute_t<uint64_t>::type ab;
	ab.x[0] = w_mantissa_a * w_mantissa_b;
	ab.x[1] = __mul64hi(w_mantissa_a, w_mantissa_b);

	const auto mlb2 = static_cast<uint32_t>(ab.x[1] >> (sizeof(uint64_t) * 8 - 2));
	const auto a0 = mlb2 & 0b01;
	const auto a1 = mlb2 & 0b10;
	const auto b0 = ((a1 >> 1) < a0) ? 0 : 1;
	const auto b1 = ~a1;
	shifted = b0 | (b1 << 1);

	ab.x[1] = (ab.x[1] << shifted) | (ab.x[0] >> (sizeof(uint64_t) * 8 - shifted));
	ab.x[0] = ab.x[0] << shifted;

	return ab;
}
} // namespace detail
} // namespace aonfp
#endif
