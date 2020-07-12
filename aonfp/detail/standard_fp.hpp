#ifndef __AONFP_DETAIL_STANDARDFP_HPP__
#define __AONFP_DETAIL_STANDARDFP_HPP__
#include <cstdint>

namespace aonfp {
namespace detail {
// If all people in the world used C++14 or later, these functions could be `template constexpr`.
template <class T>
constexpr unsigned get_exponent_size();
template <> constexpr unsigned get_exponent_size<float >() {return 8 ;}
template <> constexpr unsigned get_exponent_size<double>() {return 11;}

template <class T>
constexpr unsigned get_mantissa_size();
template <> constexpr unsigned get_mantissa_size<float >() {return 23;}
template <> constexpr unsigned get_mantissa_size<double>() {return 52;}
}
} // namespace aonfp
#endif
