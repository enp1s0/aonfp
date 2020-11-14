#ifndef __AONFP_Q_HPP__
#define __AONFP_Q_HPP__
#include "detail/compose_q.hpp"
#include "detail/detail_q.hpp"

namespace aonfp {
namespace q {
template <class EXPONENT_T, class S_MANTISSA_T>
struct aonfp_t {
    using exponent_t = EXPONENT_T;
    using s_mantissa_t = S_MANTISSA_T;
};

template <class T, class EXP_T, class S_MANTISSA_T>
AONFP_HOST_DEVICE inline void decompose(EXP_T& exp, S_MANTISSA_T& s_mantissa, const T v) {
	int move_up;
	s_mantissa = detail::q::decompose_sign_mantissa<S_MANTISSA_T>(v, move_up);
	detail::uo_flow_t uo;
	exp = detail::q::decompose_exponent<EXP_T>(v, move_up, uo);

	if (uo == detail::uo_flow_overflow) {
		exp = detail::q::get_inf_exponent_bitstring<EXP_T>();
		s_mantissa = detail::q::get_inf_sign_mantissa_bitstring<S_MANTISSA_T>(s_mantissa);
		return;
	}

	if (uo == detail::uo_flow_underflow) {
		exp = detail::q::get_zero_exponent_bitstring<EXP_T>();
		s_mantissa = detail::q::get_zero_sign_mantissa_bitstring<S_MANTISSA_T>(s_mantissa);
		return;
	}
}

template <class T, class EXP_T, class S_MANTISSA_T>
AONFP_HOST_DEVICE inline T compose(const EXP_T exp, const S_MANTISSA_T mantissa) {
	int move_up;
	const auto fp_mantissa = detail::q::compose_sign_mantissa<T, S_MANTISSA_T>(mantissa, 1, move_up);
	const auto fp = detail::q::compose_exponent<T, EXP_T>(exp, fp_mantissa, move_up);

	using ieee_bitstring_t = typename aonfp::detail::bitstring_t<T>::type;
	if (((*reinterpret_cast<const ieee_bitstring_t*>(&fp)) << 1) == 0 && exp) {
		if (mantissa >> (sizeof(S_MANTISSA_T) * 8 - 1)) {
			return static_cast<T>(-1);
		} else {
			return static_cast<T>(1);
		}
	}
	return fp;
}
} // namespace q
} // namespace aonfp
#endif
