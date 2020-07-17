#include <aonfp/aonfp.hpp>
#include <iostream>
#include <random>
#include <cmath>
#include <limits>

constexpr std::size_t C = 1lu << 3;

template <class T>
std::string get_type_name();
template <> std::string get_type_name<float   >() {return "float";}
template <> std::string get_type_name<double  >() {return "double";}
template <> std::string get_type_name<uint64_t>() {return "uint64_t";}
template <> std::string get_type_name<uint32_t>() {return "uint32_t";}
template <> std::string get_type_name<uint16_t>() {return "uint16_t";}
template <> std::string get_type_name<uint8_t >() {return "uint8_t";}

template <class T, class S_EXP_T, class MANTISSA_T>
void test_compose_decompose() {
	std::mt19937 mt(std::random_device{}());

	std::uniform_real_distribution<T> dist(-1, 1);

	const auto threshold = std::pow(10.0f, -std::log10(2.0) * std::min<unsigned>(sizeof(MANTISSA_T) * 8, aonfp::detail::standard_fp::get_mantissa_size<T>()));

	std::size_t passed = 0;

	for (std::size_t c = 0; c < C; c++) {
		const auto test_value = dist(mt);

		S_EXP_T s_exp;
		MANTISSA_T mantissa;
		aonfp::decompose(s_exp, mantissa, test_value);

		const auto decomposed_value = aonfp::compose<T>(s_exp, mantissa);

		const auto error = std::abs(static_cast<double>(test_value) - decomposed_value);

		if (error > threshold) {
			std::printf("{org = %e, cvt = %e} error = %e > [threshold:%e]\n", test_value, decomposed_value, error, threshold);
		} else {
			passed++;
		}
	}
	std::printf("TEST {%10s, ES %10s, M %10s} : %lu / %lu is passed\n",
				get_type_name<T>().c_str(),
				get_type_name<S_EXP_T>().c_str(),
				get_type_name<MANTISSA_T>().c_str(),
				passed, C
				);
}

int main() {
	test_compose_decompose<double, uint32_t, uint32_t>();
}
