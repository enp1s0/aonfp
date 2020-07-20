#ifndef __AONFP_AONFP_HPP__
#define __AONFP_AONFP_HPP__
#ifndef AONFP_HOST_DEVICE
 #if defined(__CUDA_ARCH__)
  #define AONFP_HOST_DEVICE __device__ __host__
 #else
  #define AONFP_HOST_DEVICE
 #endif
#endif

#include "detail/detail.hpp"
#include "detail/compose.hpp"
#include "detail/standard_fp.hpp"

namespace aonfp {
template <class T, class S_EXP_T, class MANTISSA_T>
AONFP_HOST_DEVICE inline void decompose(S_EXP_T& s_exp, MANTISSA_T& mantissa, const T v) {
	int move_up;
	mantissa = detail::decompose_mantissa<MANTISSA_T>(v, move_up);
	detail::uo_flow_t uo;
	s_exp = detail::decompose_sign_exponent<S_EXP_T>(v, move_up, uo);

	if (uo == detail::uo_flow_overflow) {
		s_exp = detail::get_inf_sign_exponent_bitstring<S_EXP_T>(s_exp);
		mantissa = detail::get_inf_mantissa_bitstring<MANTISSA_T>();
		return;
	}

	if (uo == detail::uo_flow_underflow) {
		s_exp = detail::get_zero_sign_exponent_bitstring<S_EXP_T>(s_exp);
		mantissa = detail::get_zero_mantissa_bitstring<MANTISSA_T>();
		return;
	}
}

template <class T, class S_EXP_T, class MANTISSA_T>
AONFP_HOST_DEVICE inline T compose(const S_EXP_T s_exp, const MANTISSA_T mantissa) {
	int move_up;
	const auto fp_mantissa = detail::compose_mantissa<T, MANTISSA_T>(mantissa, 1, move_up);
	return detail::compose_sign_exponent<T, S_EXP_T>(s_exp, fp_mantissa, move_up);
}

template <class SRC_S_EXP_T, class SRC_MANTISSA_T, class DST_S_EXP_T, class DST_MANTISSA_T>
AONFP_HOST_DEVICE void mul(DST_S_EXP_T* const dst_s_exp, DST_MANTISSA_T* const dst_mantissa,
		const SRC_S_EXP_T src_s_exp_A, const SRC_MANTISSA_T src_mantissa_A,
		const SRC_S_EXP_T src_s_exp_B, const SRC_MANTISSA_T src_mantissa_B) {

}

} //namespace aonfp
#endif
