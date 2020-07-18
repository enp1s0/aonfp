# AONFP (Architecture Oriented Non-standard Floating Point)

## Description
AONFP is a header only C++ library for floating point data compression.

## Supported data format

### Standard floating point

- `float` (IEEE 754 Binary32)
- `double` (IEEE 754 Binary64)

### Decomposed data type
Data types for `sign_exponent` and `mantissa` are

- `uint64`
- `uint32`
- `uint16`
- `uint8`

respectively.


## Sample code
```cpp
const auto value = static_cast<double>(1.234);
uint8_t s_exp;
uint64_t mantissa;

// decompose
aonfp::decompose(s_exp, mantissa, value);

// compose
const auto decomposed_value = aonfp::compose<double>(s_exp, mantissa);
```

### CUDA extension
AONFP has a function which copies an AONFP format array in host memory to device memory while converting to IEEE format.
And also it has a reverse function.

![cuda-copy](docs/aonfp_cuda_copy.svg)

To use this feature, you need to build a static library and link it to your application.

#### Build
#### Link
