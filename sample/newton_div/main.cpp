#include <aonfp/aonfp.hpp>
#include <aonfp/detail/utils.hpp>
#include <iostream>
#include <random>
#include <cmath>
#include <limits>

constexpr std::size_t C = 1lu << 20;

template <class S_EXP_T, class MANTISSA_T>
void newton_r(S_EXP_T& src_s_exp, MANTISSA_T& src_mantissa, const S_EXP_T q_s_exp, const MANTISSA_T q_mantissa) {
	// const value
	S_EXP_T two_s_exp;
	MANTISSA_T two_mantissa;
	aonfp::decompose(two_s_exp, two_mantissa, 2.0);

	S_EXP_T tmp_s_exp;
	MANTISSA_T tmp_mantissa;
	aonfp::mul(
		tmp_s_exp, tmp_mantissa,
		src_s_exp, src_mantissa,
		q_s_exp, q_mantissa
		);
	aonfp::sub(
		tmp_s_exp, tmp_mantissa,
		two_s_exp, two_mantissa,
		tmp_s_exp, tmp_mantissa
		);

	aonfp::mul(
		src_s_exp, src_mantissa,
		src_s_exp, src_mantissa,
		tmp_s_exp, tmp_mantissa
		);
}

template <class T, class S_EXP_T, class MANTISSA_T>
void test_newton_div() {
	std::mt19937 mt(std::random_device{}());

	std::uniform_real_distribution<T> dist(-10000, 10000);

	const auto threshold_base = 64 * std::pow(10.0f, -std::log10(2.0) * std::min<unsigned>(sizeof(MANTISSA_T) * 8, aonfp::detail::standard_fp::get_mantissa_size<T>()));

	std::size_t passed = 0;

	for (std::size_t c = 0; c < C; c++) {
		const auto test_value = dist(mt);

		S_EXP_T s_exp;
		MANTISSA_T mantissa;
		aonfp::decompose(s_exp, mantissa, test_value);

		S_EXP_T r_s_exp;
		MANTISSA_T r_mantissa;
		aonfp::decompose(r_s_exp, r_mantissa, 1.2 / test_value);

		for (unsigned i = 0; i < 10; i++) {
			newton_r(r_s_exp, r_mantissa, s_exp, mantissa);
		}

		const auto decomposed_value = aonfp::compose<T>(r_s_exp, r_mantissa);

		const auto error = std::abs(static_cast<double>(1.0 / test_value) - decomposed_value);
		const auto threshold = threshold_base * std::abs(static_cast<double>(1.0 / test_value));

		if (error > threshold) {
			std::printf("SE : ");aonfp::detail::utils::print_hex(s_exp, true);
			std::printf("M  : ");aonfp::detail::utils::print_hex(mantissa, true);
			std::printf("{org = %e, cvt = %e} error = %e > [threshold:%e]\n", 1.0 / test_value, decomposed_value, error, threshold);
		} else {
			passed++;
		}
	}
	std::printf("TEST {%10s, ES %10s, M %10s} : %lu / %lu (%3.3f %%)\n",
				aonfp::detail::utils::get_type_name_string<T>(),
				aonfp::detail::utils::get_type_name_string<S_EXP_T>(),
				aonfp::detail::utils::get_type_name_string<MANTISSA_T>(),
				passed, C,
				static_cast<double>(passed) / C * 100.0
				);
}

int main() {
	test_newton_div<double, uint64_t, uint64_t>();
	test_newton_div<double, uint64_t, uint32_t>();
	test_newton_div<double, uint64_t, uint16_t>();
	test_newton_div<double, uint64_t, uint8_t>();
	test_newton_div<double, uint32_t, uint64_t>();
	test_newton_div<double, uint32_t, uint32_t>();
	test_newton_div<double, uint32_t, uint16_t>();
	test_newton_div<double, uint32_t, uint8_t>();
	test_newton_div<double, uint16_t, uint64_t>();
	test_newton_div<double, uint16_t, uint32_t>();
	test_newton_div<double, uint16_t, uint16_t>();
	test_newton_div<double, uint16_t, uint8_t>();
	test_newton_div<double, uint8_t, uint64_t>();
	test_newton_div<double, uint8_t, uint32_t>();
	test_newton_div<double, uint8_t, uint16_t>();
	test_newton_div<double, uint8_t, uint8_t>();

	test_newton_div<float, uint64_t, uint64_t>();
	test_newton_div<float, uint64_t, uint32_t>();
	test_newton_div<float, uint64_t, uint16_t>();
	test_newton_div<float, uint64_t, uint8_t>();
	test_newton_div<float, uint32_t, uint64_t>();
	test_newton_div<float, uint32_t, uint32_t>();
	test_newton_div<float, uint32_t, uint16_t>();
	test_newton_div<float, uint32_t, uint8_t>();
	test_newton_div<float, uint16_t, uint64_t>();
	test_newton_div<float, uint16_t, uint32_t>();
	test_newton_div<float, uint16_t, uint16_t>();
	test_newton_div<float, uint16_t, uint8_t>();
	test_newton_div<float, uint8_t, uint64_t>();
	test_newton_div<float, uint8_t, uint32_t>();
	test_newton_div<float, uint8_t, uint16_t>();
	test_newton_div<float, uint8_t, uint8_t>();
}
