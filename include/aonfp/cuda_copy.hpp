#ifndef __AONFP_CUDA_COPY_HPP__
#define __AONFP_CUDA_COPY_HPP__
#include <cstdint>
namespace aonfp {
namespace cuda {
template <class T, class S_EXP_T, class MANTISSA_T>
int copy_to_device(
		T* const dst_ptr, unsigned inc_dst,
		const S_EXP_T* const s_exp_ptr, const unsigned inc_src_s_exp,
		const MANTISSA_T* const mantissa_ptr, const unsigned inc_src_mantissa,
		const std::size_t N,
		const unsigned block_size = 256,
		const bool set_cpu_affinity_frag = true,
		cudaStream_t stream = 0);

template <class S_EXP_T, class MANTISSA_T, class T>
int copy_to_host(
		S_EXP_T* const s_exp_ptr, const unsigned inc_dst_s_exp,
		MANTISSA_T* const mantissa_ptr, const unsigned inc_dst_mantissa,
		const T* const src_ptr, const unsigned inc_src,
		const std::size_t N,
		const unsigned block_size = 256,
		const bool set_cpu_affinity_frag = true,
		cudaStream_t stream = 0);
} // namespace cuda
namespace q {
namespace cuda {
template <class T, class EXP_T, class S_MANTISSA_T>
int copy_to_device(
		T* const dst_ptr, unsigned inc_dst,
		const EXP_T* const exp_ptr, const unsigned inc_src_exp,
		const S_MANTISSA_T* const s_mantissa_ptr, const unsigned inc_src_s_mantissa,
		const std::size_t N,
		const unsigned block_size = 256,
		const bool set_cpu_affinity_frag = true,
		cudaStream_t stream = 0);

template <class EXP_T, class S_MANTISSA_T, class T>
int copy_to_host(
		EXP_T* const exp_ptr, const unsigned inc_dst_exp,
		S_MANTISSA_T* const s_mantissa_ptr, const unsigned inc_dst_s_mantissa,
		const T* const src_ptr, const unsigned inc_src,
		const std::size_t N,
		const unsigned block_size = 256,
		const bool set_cpu_affinity_frag = true,
		cudaStream_t stream = 0);
} // namespace q
} // namespace cuda
} // namespace aonfp
#endif
