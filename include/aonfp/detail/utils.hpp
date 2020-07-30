#ifndef __AONFP_DETAIL_UTILS_HPP__
#define __AONFP_DETAIL_UTILS_HPP__
#include <iostream>
namespace aonfp {
namespace detail {
namespace utils {
template <class T>
void print_hex(const T v, const bool line_break = false);
template <> void print_hex<uint64_t>(const uint64_t v, const bool line_break) {printf("0x%016lx", v);if(line_break)printf("\n");}
template <> void print_hex<uint32_t>(const uint32_t v, const bool line_break) {printf("0x%08x", v);if(line_break)printf("\n");}
template <> void print_hex<uint16_t>(const uint16_t v, const bool line_break) {printf("0x%04x", v);if(line_break)printf("\n");}
template <> void print_hex<uint8_t >(const uint8_t  v, const bool line_break) {printf("0x%02x", v);if(line_break)printf("\n");}

template <class T>
inline void print_bin(const T v, const bool line_break = false) {
	for (int i = sizeof(T) * 8 - 1; i >= 0; i--) {
		std::printf("%d", static_cast<int>(v >> i) & 0x1);
	}
	if (line_break) {
		std::printf("\n");
	}
}

template <class T>
const char* get_type_name_string();
template <> const char* get_type_name_string<float   >() {return "float";}
template <> const char* get_type_name_string<double  >() {return "double";}
template <> const char* get_type_name_string<uint64_t>() {return "uint64_t";}
template <> const char* get_type_name_string<uint32_t>() {return "uint32_t";}
template <> const char* get_type_name_string<uint16_t>() {return "uint16_t";}
template <> const char* get_type_name_string<uint8_t >() {return "uint8_t";}
} // namespace utils
} // namespace detail
} // namespace aonfp
#endif
