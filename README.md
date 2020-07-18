# AONFP (Architecture Oriented Non-standard Floating Point)

## Supported

### Standard floating point

- `float`
- `double`

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
