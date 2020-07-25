#ifndef __AONFP_CUDA_COPY_HPP__
#define __AONFP_CUDA_COPY_HPP__
namespace aonfp {
namespace cuda {
template <class T, class S_EXP_T, class MANTISSA_T>
int copy_to_device(T* const dst_ptr, const S_EXP_T* const s_exp_ptr, const MANTISSA_T* const mantissa_ptr, const std::size_t N, const bool set_cpu_affinity_frag = true, cudaStream_t stream = 0);

template <class S_EXP_T, class MANTISSA_T, class T>
int copy_to_host(S_EXP_T* const s_exp_ptr, MANTISSA_T* const mantissa_ptr, const T* const dst_ptr, const std::size_t N, const bool set_cpu_affinity_frag = true, cudaStream_t stream = 0);
} // namespace cuda
} // namespace aonfp
#endif
