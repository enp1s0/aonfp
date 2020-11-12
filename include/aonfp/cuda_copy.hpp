#ifndef __AONFP_CUDA_COPY_HPP__
#define __AONFP_CUDA_COPY_HPP__
namespace aonfp {
namespace cuda {
template <class T, class S_EXP_T, class MANTISSA_T>
int copy_to_device(T* const dst_ptr, const S_EXP_T* const s_exp_ptr, const MANTISSA_T* const mantissa_ptr, const std::size_t N, const unsigned block_size = 256, const bool set_cpu_affinity_frag = true, cudaStream_t stream = 0);

template <class S_EXP_T, class MANTISSA_T, class T>
int copy_to_host(S_EXP_T* const s_exp_ptr, MANTISSA_T* const mantissa_ptr, const T* const dst_ptr, const std::size_t N, const unsigned block_size = 256, const bool set_cpu_affinity_frag = true, cudaStream_t stream = 0);
} // namespace cuda
namespace q {
namespace cuda {
template <class T, class EXP_T, class S_MANTISSA_T>
int copy_to_device(T* const dst_ptr, const EXP_T* const exp_ptr, const S_MANTISSA_T* const s_mantissa_ptr, const std::size_t N, const unsigned block_size = 256, const bool set_cpu_affinity_frag = true, cudaStream_t stream = 0);

template <class EXP_T, class S_MANTISSA_T, class T>
int copy_to_host(EXP_T* const exp_ptr, S_MANTISSA_T* const s_mantissa_ptr, const T* const dst_ptr, const std::size_t N, const unsigned block_size = 256, const bool set_cpu_affinity_frag = true, cudaStream_t stream = 0);
} // namespace q
} // namespace cuda
} // namespace aonfp
#endif
