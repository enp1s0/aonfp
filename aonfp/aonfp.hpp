#ifndef __AONFP_AONFP_HPP__
#define __AONFP_AONFP_HPP__
#include "detail/detail.hpp"
#include "detail/standard_fp.hpp"

namespace aonfp {
template <class T, class S_EXP_T, class MANTISSA_T>
inline void convert(S_EXP_T& s_exp, MANTISSA_T& mantissa, const T v) {
	detail::uo_flow_t uo;
	detail::copy_sign_exponent<S_EXP_T>(s_exp, uo);

	if (uo == detail::uo_flow_overflow) {
		s_exp = detail::get_inf_exponent<S_EXP_T>(0);
		mantissa = detail::get_inf_mantissa<MANTISSA_T>();
		return;
	}

	if (uo == detail::uo_flow_underflow) {
		s_exp = detail::get_inf_exponent<S_EXP_T>(s_exp);
		mantissa = detail::get_inf_mantissa<MANTISSA_T>();
		return;
	}
}
} //namespace aonfp
#endif
