#ifndef __AONFP_DETAIL_UINT128_HPP__
#define __AONFP_DETAIL_UINT128_HPP__
#include <cstdint>

namespace aonfp {
namespace detail {
struct aonfp_uint128_t {
	uint64_t x[2];
	aonfp_uint128_t operator=(const uint64_t v) {
		x[0] = v;
		x[1] = 0;
		return *this;
	}
};
}
}
#endif
