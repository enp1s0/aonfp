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
	const auto bitstrings = *reinterpret_cast<const uint64_t*>(&v);
	return ((bitstrings & 0xffffffffffffflu) << 11) | (bitstrings & 0x8000000000000000lu);
}

template <> AONFP_HOST_DEVICE inline uint32_t decompose_mantissa<uint32_t>(const double v, int& move_up) {
	const auto mantissa_bs = (*reinterpret_cast<const uint64_t*>(&v)) & 0xffffffffffffflu;
	const auto sign = (*reinterpret_cast<const uint64_t*>(&v)) >> 63;
	const auto r_s = (mantissa_bs & 0x100000lu) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	move_up = (mantissa_bs_a & 0x10000000000000lu) >> 52;
	return (mantissa_bs_a >> 21) | (static_cast<uint32_t>(sign) << 31);
}

template <> AONFP_HOST_DEVICE inline uint16_t decompose_mantissa<uint16_t>(const double v, int& move_up) {
	const auto mantissa_bs = (*reinterpret_cast<const uint64_t*>(&v)) & 0xffffffffffffflu;
	const auto sign = (*reinterpret_cast<const uint64_t*>(&v)) >> 63;
	const auto r_s = (mantissa_bs & 0x1000000000lu) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	move_up = (mantissa_bs_a & 0x10000000000000lu) >> 52;
	return (mantissa_bs_a >> 37) | (static_cast<uint16_t>(sign) << 15);
}

template <> AONFP_HOST_DEVICE inline uint8_t decompose_mantissa<uint8_t>(const double v, int& move_up) {
	const auto mantissa_bs = (*reinterpret_cast<const uint64_t*>(&v)) & 0xffffffffffffflu;
	const auto sign = (*reinterpret_cast<const uint64_t*>(&v)) >> 63;
	const auto r_s = (mantissa_bs & 0x100000000000lu) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	move_up = (mantissa_bs_a & 0x10000000000000lu) >> 52;
	return (mantissa_bs_a >> 45) | (static_cast<uint8_t>(sign) << 7);
}

template <class T>
AONFP_HOST_DEVICE inline T decompose_mantissa(const float v, int& move_up);
template <> AONFP_HOST_DEVICE inline uint64_t decompose_mantissa<uint64_t>(const float v, int& move_up) {
	move_up = 0;
	const auto bitstrings = *reinterpret_cast<const uint32_t*>(&v);
	const auto sign = (*reinterpret_cast<const uint32_t*>(&v)) >> 31;

	const auto mantissa = static_cast<uint64_t>(bitstrings & 0x7fffff) << (32 + 9 - 1);
	return mantissa | (static_cast<uint64_t>(sign) << 63);
}

template <> AONFP_HOST_DEVICE inline uint32_t decompose_mantissa<uint32_t>(const float v, int& move_up) {
	move_up = 0;
	const auto bitstrings = *reinterpret_cast<const uint32_t*>(&v);
	const auto sign = (*reinterpret_cast<const uint32_t*>(&v)) >> 31;

	const auto mantissa = (bitstrings & 0x7fffff) << (9 - 1);
	return mantissa | (sign << 31);
}

template <> AONFP_HOST_DEVICE inline uint16_t decompose_mantissa<uint16_t>(const float v, int& move_up) {
	const auto mantissa_bs = (*reinterpret_cast<const uint32_t*>(&v)) & 0x7fffff;
	const auto sign = (*reinterpret_cast<const uint32_t*>(&v)) >> 31;
	const auto r_s = (mantissa_bs & 0x40) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	move_up = (mantissa_bs_a & 0x800000) >> 23;
	return (mantissa_bs_a >> 8) | (static_cast<uint16_t>(sign) << 15);
}

template <> AONFP_HOST_DEVICE inline uint8_t decompose_mantissa<uint8_t>(const float v, int& move_up) {
	const auto mantissa_bs = (*reinterpret_cast<const uint32_t*>(&v)) & 0x7fffff;
	const auto sign = (*reinterpret_cast<const uint32_t*>(&v)) >> 31;
	const auto r_s = (mantissa_bs & 0x8000) << 1;
	const auto mantissa_bs_a = mantissa_bs + r_s;
	move_up = (mantissa_bs_a & 0x800000) >> 23;
	return (mantissa_bs_a >> 14) | (static_cast<uint8_t>(sign) << 7);
}
} // namespace detail
} // namespace aonfp

#endif
