#include <aonfp/aonfp.hpp>
#include <aonfp/detail/utils.hpp>
#include <iostream>
#include <random>
#include <cmath>
#include <limits>

constexpr std::size_t C = 1lu << 10;

template <class DST_S_EXP_T, class DST_MANTISSA_T, class SRC_S_EXP_T, class SRC_MANTISSA_T>
void test_mul() {
	std::mt19937 mt(std::random_device{}());

	std::uniform_real_distribution<double> dist(-10000, 10000);

	const auto threshold_base = 16 * std::pow(10.0f, -std::log10(2.0) * std::min<unsigned>(sizeof(DST_MANTISSA_T) * 8, aonfp::detail::standard_fp::get_mantissa_size<double>()));

	std::size_t passed = 0;

	for (std::size_t c = 0; c < C; c++) {
		const auto test_value_a = dist(mt);
		const auto test_value_b = dist(mt);

		SRC_S_EXP_T s_exp_a, s_exp_b;
		SRC_MANTISSA_T mantissa_a, mantissa_b;
		aonfp::decompose(s_exp_a, mantissa_a, test_value_a);
		aonfp::decompose(s_exp_b, mantissa_b, test_value_b);

		DST_S_EXP_T s_exp_ab;
		DST_MANTISSA_T mantissa_ab;
		aonfp::mul(s_exp_ab, mantissa_ab, s_exp_a, mantissa_a, s_exp_b, mantissa_b);

		const auto test_value_ab = aonfp::compose<double>(s_exp_ab, mantissa_ab);

		const auto correct_ans = test_value_a * test_value_b;
		const auto error = std::abs(correct_ans - test_value_ab);
		const auto threshold = threshold_base * std::abs(correct_ans);

		if (error > threshold) {
			std::printf("SE_A : ");aonfp::detail::utils::print_hex(s_exp_a, true);
			std::printf("M_A  : ");aonfp::detail::utils::print_hex(mantissa_a, true);
			std::printf("SE_B : ");aonfp::detail::utils::print_hex(s_exp_b, true);
			std::printf("M_B  : ");aonfp::detail::utils::print_hex(mantissa_b, true);
			std::printf("{ab = %e, cor = %e} error = %e > [threshold:%e]\n", test_value_ab, correct_ans, error, threshold);
		} else {
			passed++;
		}
	}
	std::printf("TEST {ES %10s, M %10s} -> {ES %10s, M %10s} : %lu / %lu (%3.3f %%)\n",
				aonfp::detail::utils::get_type_name_string<SRC_S_EXP_T>(),
				aonfp::detail::utils::get_type_name_string<SRC_MANTISSA_T>(),
				aonfp::detail::utils::get_type_name_string<DST_S_EXP_T>(),
				aonfp::detail::utils::get_type_name_string<DST_MANTISSA_T>(),
				passed, C,
				static_cast<double>(passed) / C * 100.0
				);
}

int main() {
	test_mul<uint64_t, uint64_t, uint64_t, uint64_t>();
	test_mul<uint64_t, uint32_t, uint64_t, uint32_t>();
	test_mul<uint64_t, uint16_t, uint64_t, uint16_t>();
	test_mul<uint64_t, uint8_t , uint64_t, uint8_t >();
	test_mul<uint32_t, uint64_t, uint32_t, uint64_t>();
	test_mul<uint32_t, uint32_t, uint32_t, uint32_t>();
	test_mul<uint32_t, uint16_t, uint32_t, uint16_t>();
	test_mul<uint32_t, uint8_t , uint32_t, uint8_t >();
	test_mul<uint16_t, uint64_t, uint16_t, uint64_t>();
	test_mul<uint16_t, uint32_t, uint16_t, uint32_t>();
	test_mul<uint16_t, uint16_t, uint16_t, uint16_t>();
	test_mul<uint16_t, uint8_t , uint16_t, uint8_t >();
	test_mul<uint8_t , uint64_t, uint8_t , uint64_t>();
	test_mul<uint8_t , uint32_t, uint8_t , uint32_t>();
	test_mul<uint8_t , uint16_t, uint8_t , uint16_t>();
	test_mul<uint8_t , uint8_t , uint8_t , uint8_t >();
}
