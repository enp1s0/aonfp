#include <aonfp/aonfp.hpp>
#include <aonfp/detail/utils.hpp>
#include <iostream>
#include <random>
#include <cmath>
#include <limits>

constexpr std::size_t C = 1lu << 20;

template <class T, class S_EXP_T, class MANTISSA_T>
void test_compose_decompose() {
	std::mt19937 mt(std::random_device{}());

	std::uniform_real_distribution<T> dist(-1, 1);

	const auto threshold = 1.5 * std::pow(10.0f, -std::log10(2.0) * std::min<unsigned>(sizeof(MANTISSA_T) * 8, aonfp::detail::standard_fp::get_mantissa_size<T>()));

	std::size_t passed = 0;

	for (std::size_t c = 0; c < C; c++) {
		const auto test_value = dist(mt);

		S_EXP_T s_exp;
		MANTISSA_T mantissa;
		aonfp::decompose(s_exp, mantissa, test_value);

		const auto decomposed_value = aonfp::compose<T>(s_exp, mantissa);

		const auto error = std::abs(static_cast<double>(test_value) - decomposed_value);

		if (error > threshold) {
			std::printf("SE : ");aonfp::detail::utils::print_hex(s_exp, true);
			std::printf("M  : ");aonfp::detail::utils::print_hex(mantissa, true);
			std::printf("{org = %e, cvt = %e} error = %e > [threshold:%e]\n", test_value, decomposed_value, error, threshold);
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
	test_compose_decompose<double, uint64_t, uint64_t>();
	test_compose_decompose<double, uint64_t, uint32_t>();
	test_compose_decompose<double, uint64_t, uint16_t>();
	test_compose_decompose<double, uint64_t, uint8_t>();
	test_compose_decompose<double, uint32_t, uint64_t>();
	test_compose_decompose<double, uint32_t, uint32_t>();
	test_compose_decompose<double, uint32_t, uint16_t>();
	test_compose_decompose<double, uint32_t, uint8_t>();
	test_compose_decompose<double, uint16_t, uint64_t>();
	test_compose_decompose<double, uint16_t, uint32_t>();
	test_compose_decompose<double, uint16_t, uint16_t>();
	test_compose_decompose<double, uint16_t, uint8_t>();
	test_compose_decompose<double, uint8_t, uint64_t>();
	test_compose_decompose<double, uint8_t, uint32_t>();
	test_compose_decompose<double, uint8_t, uint16_t>();
	test_compose_decompose<double, uint8_t, uint8_t>();

	test_compose_decompose<float, uint64_t, uint64_t>();
	test_compose_decompose<float, uint64_t, uint32_t>();
	test_compose_decompose<float, uint64_t, uint16_t>();
	test_compose_decompose<float, uint64_t, uint8_t>();
	test_compose_decompose<float, uint32_t, uint64_t>();
	test_compose_decompose<float, uint32_t, uint32_t>();
	test_compose_decompose<float, uint32_t, uint16_t>();
	test_compose_decompose<float, uint32_t, uint8_t>();
	test_compose_decompose<float, uint16_t, uint64_t>();
	test_compose_decompose<float, uint16_t, uint32_t>();
	test_compose_decompose<float, uint16_t, uint16_t>();
	test_compose_decompose<float, uint16_t, uint8_t>();
	test_compose_decompose<float, uint8_t, uint64_t>();
	test_compose_decompose<float, uint8_t, uint32_t>();
	test_compose_decompose<float, uint8_t, uint16_t>();
	test_compose_decompose<float, uint8_t, uint8_t>();
}
