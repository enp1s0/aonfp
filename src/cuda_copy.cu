#include <aonfp/aonfp.hpp>
#include <aonfp/cuda_copy.hpp>

namespace {
template <class T, class S_EXP_T, class MANTISSA_T>
__global__ void copy_to_device_kernel(T *const dst_ptr, const S_EXP_T *const s_exp_ptr, const MANTISSA_T *const mantissa_ptr, const std::size_t N) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N) {
		return;
	}
	const auto s_exp = s_exp_ptr[tid];
	const auto mantissa = mantissa_ptr[tid];

	dst_ptr[tid] = aonfp::compose<T>(s_exp, mantissa);
}

template <class S_EXP_T, class MANTISSA_T, class T>
__global__ void copy_to_host_kernel(S_EXP_T *const s_exp_ptr, MANTISSA_T *const mantissa_ptr, const T* const src_ptr, const std::size_t N) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N) {
		return;
	}
	S_EXP_T s_exp;
	MANTISSA_T mantissa;

	aonfp::decompose(s_exp, mantissa, src_ptr[tid]);

	s_exp_ptr[tid] = s_exp;
	mantissa_ptr[tid] = mantissa;
}
} // namespace

template <class T, class S_EXP_T, class MANTISSA_T>
int aonfp::cuda::copy_to_device(T *const dst_ptr, const S_EXP_T *const s_exp_ptr, const MANTISSA_T *const mantissa_ptr, const std::size_t N, cudaStream_t stream) {
	constexpr std::size_t block_size = 256;
	copy_to_device_kernel<T, S_EXP_T, MANTISSA_T><<<(N + block_size - 1) / block_size, block_size, 0, stream>>>(dst_ptr, s_exp_ptr, mantissa_ptr, N);
	if (cudaGetLastError() == cudaSuccess) {
		return 0;
	} else {
		return 1;
	}
}

template <class S_EXP_T, class MANTISSA_T, class T>
int aonfp::cuda::copy_to_host(S_EXP_T *const s_exp_ptr, MANTISSA_T *const mantissa_ptr, const T *const src_ptr, const std::size_t N, cudaStream_t stream) {
	constexpr std::size_t block_size = 256;
	copy_to_host_kernel<T, S_EXP_T, MANTISSA_T><<<(N + block_size - 1) / block_size, block_size, 0, stream>>>(s_exp_ptr, mantissa_ptr, src_ptr, N);
	if (cudaGetLastError() == cudaSuccess) {
		return 0;
	} else {
		return 1;
	}
}
