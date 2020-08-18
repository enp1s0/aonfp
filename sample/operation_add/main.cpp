#include <aonfp/aonfp.hpp>
#include <aonfp/detail/utils.hpp>
#include <iostream>
#include <random>
#include <cmath>
#include <limits>

constexpr std::size_t C = 1lu << 5;

template <class DST_S_EXP_T, class DST_MANTISSA_T, class SRC_S_EXP_T, class SRC_MANTISSA_T>
void test_add() {
	std::mt19937 mt(std::random_device{}());

	std::uniform_real_distribution<double> dist(-1, 1);

	const auto threshold_base = std::pow(10.0, 5.0 - std::log10(2.0) * std::min<unsigned>(sizeof(DST_MANTISSA_T), sizeof(SRC_MANTISSA_T)) * 8);

	std::size_t passed = 0;

	for (std::size_t c = 0; c < C; c++) {
		const auto org_test_value_a = dist(mt);
		const auto org_test_value_b = dist(mt);

		SRC_S_EXP_T s_exp_a, s_exp_b;
		SRC_MANTISSA_T mantissa_a, mantissa_b;
		aonfp::decompose(s_exp_a, mantissa_a, org_test_value_a);
		aonfp::decompose(s_exp_b, mantissa_b, org_test_value_b);

		DST_S_EXP_T s_exp_ab;
		DST_MANTISSA_T mantissa_ab;
		aonfp::add(s_exp_ab, mantissa_ab, s_exp_a, mantissa_a, s_exp_b, mantissa_b);

		const auto test_value_a = aonfp::compose<double>(s_exp_a, mantissa_a);
		const auto test_value_b = aonfp::compose<double>(s_exp_b, mantissa_b);
		const auto test_value_ab = aonfp::compose<double>(s_exp_ab, mantissa_ab);

		const auto correct_ans = test_value_a + test_value_b;
		const auto error = std::abs(correct_ans - test_value_ab);
		const auto threshold = threshold_base * std::abs(correct_ans);// * (std::abs(test_value_a) + std::abs(test_value_b));

		DST_MANTISSA_T correct_mantissa;
		DST_S_EXP_T correct_s_exp;
		aonfp::decompose(correct_s_exp, correct_mantissa, correct_ans);

		if (error > threshold) {
			std::printf("a    : %e\n", test_value_a);
			std::printf("b    : %e\n", test_value_b);
			std::printf("SE_A : 0b");aonfp::detail::utils::print_bin<SRC_S_EXP_T>(s_exp_a, true);
			std::printf("M_A  : 0b");aonfp::detail::utils::print_bin<SRC_MANTISSA_T>(mantissa_a, true);
			std::printf("SE_B : 0b");aonfp::detail::utils::print_bin<SRC_S_EXP_T>(s_exp_b, true);
			std::printf("M_B  : 0b");aonfp::detail::utils::print_bin<SRC_MANTISSA_T>(mantissa_b, true);
			std::printf("SE_AB: 0b");aonfp::detail::utils::print_bin<DST_S_EXP_T>(s_exp_ab, true);
			std::printf("M_AB : 0b");aonfp::detail::utils::print_bin<DST_MANTISSA_T>(mantissa_ab, true);
			std::printf("M_C  : 0b");aonfp::detail::utils::print_bin(correct_mantissa, true);
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

#define TEST_ADD_0(a, b, c) \
	test_add<uint64_t, a, b, c>(); \
	test_add<uint32_t, a, b, c>(); \
	test_add<uint16_t, a, b, c>(); \
	test_add<uint8_t , a, b, c>();

#define TEST_ADD_1(b, c) \
	TEST_ADD_0(uint64_t, b, c); \
	TEST_ADD_0(uint32_t, b, c); \
	TEST_ADD_0(uint16_t, b, c); \
	TEST_ADD_0(uint8_t , b, c);

#define TEST_ADD_2(c) \
	TEST_ADD_1(uint64_t, c); \
	TEST_ADD_1(uint32_t, c); \
	TEST_ADD_1(uint16_t, c); \
	TEST_ADD_1(uint8_t , c);

int main() {
	TEST_ADD_2(uint64_t);
	TEST_ADD_2(uint32_t);
	TEST_ADD_2(uint16_t);
	TEST_ADD_2(uint8_t );
}
