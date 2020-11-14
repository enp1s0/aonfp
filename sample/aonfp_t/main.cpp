#include <aonfp/aonfp.hpp>
#include <aonfp/detail/utils.hpp>

template <class AONFP_T>
class print_aonfp {
	typename AONFP_T::mantissa_t m;
	typename AONFP_T::s_exponent_t se;
public:
	print_aonfp(const double v) {
		aonfp::decompose(se, m, v);
	}

	void print() {
		std::printf(" e:");aonfp::detail::utils::print_bin(m , true);
		std::printf("se:");aonfp::detail::utils::print_bin(se, true);
	}
};

int main() {
	print_aonfp<aonfp::aonfp_t<uint16_t, uint16_t>>{0.65}.print();
}
