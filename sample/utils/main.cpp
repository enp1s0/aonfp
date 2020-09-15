#include <aonfp/detail/utils.hpp>

template <class T>
void print_fp(const T v) {
	std::printf("type = %s\n", aonfp::detail::utils::get_type_name_string<T>());
	std::printf("fp    : %e\n", v);
	std::printf("bin   : "); aonfp::detail::utils::print_bin(v, 1);
	std::printf("hex   : "); aonfp::detail::utils::print_hex(v, 1);
}

int main() {
	print_fp(1.234);
	print_fp(1.234f);
}
