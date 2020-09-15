#ifndef __AONFP_Q_HPP__
#define __AONFP_Q_HPP__
#include "detail/compose_q.hpp"
#include "detail/detail_q.hpp"

namespace aonfp {
namespace q {
namespace aonfp {
template <class T, class EXP_T, class S_MANTISSA_T>
AONFP_HOST_DEVICE inline void decompose(EXP_T& exp, S_MANTISSA_T& s_mantissa, const T v) {
	int move_up;
	s_mantissa = detail::q::decompose_sign_mantissa<S_MANTISSA_T>(v, move_up);
	detail::uo_flow_t uo;
	exp = detail::q::decompose_exponent<EXP_T>(v, move_up, uo);

	if (uo == detail::uo_flow_overflow) {
		exp = detail::get_inf_sign_exponent_bitstring<EXP_T>();
		s_mantissa = detail::q::get_inf_sign_mantissa_bitstring<S_MANTISSA_T>(s_mantissa);
		return;
	}

	if (uo == detail::uo_flow_underflow) {
		exp = detail::get_zero_sign_exponent_bitstring<EXP_T>();
		s_mantissa = detail::get_zero_mantissa_bitstring<S_MANTISSA_T>(s_mantissa);
		return;
	}
}
} // namespace q
} // namespace aonfp
#endif
