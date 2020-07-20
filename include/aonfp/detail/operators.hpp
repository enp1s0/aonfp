#ifndef __AONFP_DETAIL_OPERATORS_HPP__
#define __AONFP_DETAIL_OPERATORS_HPP__
#include "detail.hpp"
namespace aonfp {
namespace detail {
template <class RESULT_T>
struct mul_compute_t {using type = RESULT_T;};
template <> struct mul_compute_t<uint32_t> {using type = uint64_t;};
template <> struct mul_compute_t<uint16_t> {using type = uint32_t;};
template <> struct mul_compute_t<uint8_t > {using type = uint16_t;};

} // namespace detail
} // namespace aonfp
#endif
