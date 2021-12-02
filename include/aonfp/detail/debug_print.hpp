#ifndef __AONFP_DETAIL_DEBUG_PRINT_HPP__
#define __AONFP_DETAIL_DEBUG_PRINT_HPP__

#include <string>
#include <cstdlib>

namespace aonfp {
namespace debug {
inline void print(const std::string str, FILE* out = stderr) {
	const char* env_name = "AONFP_DEBUG";
	const char* env_var = std::getenv(env_name);
	if (env_var == nullptr || std::string(env_var) == "0") {
		return;
	}

	std::fprintf(out, "[AONFP]: %s\n", str.c_str());
}
} // namespace debug
} // namespace aonfp

#endif
