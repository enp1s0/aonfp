#include <algorithm>
#include <exception>
#include <iomanip>
#include <sstream>
#include <string>
#include <sched.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <aonfp/aonfp.hpp>
#include <aonfp/q.hpp>
#include <aonfp/cuda_copy.hpp>

namespace {

void cuda_check_error(
		const cudaError_t stat,
		const std::string func
		) {
	if (stat != cudaSuccess) {
		throw std::runtime_error(std::string("[AONFP ERROR]: ") + cudaGetErrorString(stat) + " @" + func);
	}
}
#define CUDA_CHECK_ERROR(stat) cuda_check_error((stat), __func__)

std::string get_cuda_path(const int device_id) {
	constexpr std::size_t busid_size = sizeof("0000:00:00.0");
	constexpr std::size_t busid_reduced_size = sizeof("0000:00");

	char busid[busid_size];
	CUDA_CHECK_ERROR(cudaDeviceGetPCIBusId(busid, busid_size, device_id));
	std::string busid_str = [](std::string str) -> std::string {
		std::transform(str.begin(), str.end(), str.begin(),
				[](const unsigned c) {return std::tolower(c);});
		return str;
	}(busid);
	const std::string path = "/sys/class/pci_bus/" + busid_str.substr(0, busid_reduced_size - 1) + "/../../" + busid_str;

	const auto real_path = realpath(path.c_str(), nullptr);
	if (real_path == nullptr) {
		throw std::runtime_error("[AONFP ERROR]: Could not find real path of " + path);
	}

	return std::string{real_path};
}

cpu_set_t str_to_cpuset(const std::string str) {
	constexpr unsigned cpuset_n_uint32 = sizeof(cpu_set_t) / sizeof(uint32_t);
	uint32_t cpumask[cpuset_n_uint32];
	auto m = cpuset_n_uint32 - 1;
	cpumask[m] = 0u;
	for (unsigned o = 0; o < str.length(); o++) {
		const char c = str.c_str()[o];
		if (c == ',') {
			m--;
			cpumask[m] = 0u;
		} else {
			const auto v = [](const char c) -> int32_t {
				const int v = c - '0';
				if (0 <= c && c <= 9) return v;
				if (9 < c) return 10 + v - 'a';
				return -1;
			}(c);
			if (v == -1) break;
			cpumask[m] <<= 4;
			cpumask[m] += v;
		}
	}

	cpu_set_t mask;
	for (unsigned a = 0; m < cpuset_n_uint32; a++, m++) {
		memcpy(reinterpret_cast<uint32_t*>(&mask) + a, cpumask + m, sizeof(cpu_set_t));
	}

	return mask;
}

cpu_set_t get_cpu_gpu_affinity(const int device_id) {
	cpu_set_t mask;
	memset(&mask, 0, sizeof(cpu_set_t));

	const auto cuda_path = get_cuda_path(device_id);
	const auto path = cuda_path + "/local_cpus";

	int fd = open(path.c_str(), O_RDONLY);
	if (fd < 0) {
		throw std::runtime_error("[AONFP ERROR]: Could not open " + path);
	}

	char affinity_str[sizeof(cpu_set_t) * 2];
	const auto s = read(fd, affinity_str, sizeof(cpu_set_t) * 2);
	if (s > 0) {
		mask = str_to_cpuset(affinity_str);
	}
	close(fd);
	return mask;
}

void set_cpu_affinity(const int device_id) {
	cpu_set_t mask;
	sched_getaffinity(0, sizeof(cpu_set_t), &mask);

	const auto gpu_mask = get_cpu_gpu_affinity(device_id);

	cpu_set_t final_mask;
	CPU_AND(&final_mask, &mask, &gpu_mask);

	if (CPU_COUNT(&final_mask)) {
		sched_setaffinity(0, sizeof(cpu_set_t), &final_mask);
	}
}

template <class T, class S_EXP_T, class MANTISSA_T>
__global__ void copy_to_device_kernel(T *const dst_ptr, const unsigned inc_dst, const S_EXP_T *const s_exp_ptr, const unsigned inc_src_s_exp, const MANTISSA_T *const mantissa_ptr, const unsigned inc_src_mantissa, const std::size_t N) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N) {
		return;
	}
	const auto s_exp = s_exp_ptr[tid * inc_src_s_exp];
	const auto mantissa = mantissa_ptr[tid * inc_src_mantissa];

	dst_ptr[tid * inc_dst] = aonfp::compose<T>(s_exp, mantissa);
}

template <class S_EXP_T, class MANTISSA_T, class T>
__global__ void copy_to_host_kernel(S_EXP_T *const s_exp_ptr, const unsigned inc_dst_s_exp, MANTISSA_T *const mantissa_ptr, const unsigned inc_dst_mantissa, const T* const src_ptr, const unsigned inc_src, const std::size_t N) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N) {
		return;
	}
	S_EXP_T s_exp;
	MANTISSA_T mantissa;

	aonfp::decompose(s_exp, mantissa, src_ptr[tid * inc_src]);

	s_exp_ptr[tid * inc_dst_s_exp] = s_exp;
	mantissa_ptr[tid * inc_dst_mantissa] = mantissa;
}

template <class T, class S_EXP_T, class S_MANTISSA_T>
__global__ void copy_to_device_q_kernel(T *const dst_ptr, const unsigned inc_dst, const S_EXP_T *const exp_ptr, const unsigned inc_src_exp, const S_MANTISSA_T *const s_mantissa_ptr, const unsigned inc_src_s_mantissa, const std::size_t N) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N) {
		return;
	}
	const auto exp = exp_ptr[tid * inc_src_exp];
	const auto s_mantissa = s_mantissa_ptr[tid * inc_src_s_mantissa];

	dst_ptr[tid * inc_dst] = aonfp::q::compose<T>(exp, s_mantissa);
}

template <class EXP_T, class S_MANTISSA_T, class T>
__global__ void copy_to_host_q_kernel(EXP_T *const exp_ptr, const unsigned inc_dst_exp, S_MANTISSA_T *const s_mantissa_ptr, const unsigned inc_dst_s_mantissa, const T* const src_ptr, const unsigned inc_src, const std::size_t N) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N) {
		return;
	}
	EXP_T exp;
	S_MANTISSA_T s_mantissa;

	aonfp::q::decompose(exp, s_mantissa, src_ptr[tid * inc_src]);

	exp_ptr[tid * inc_dst_exp] = exp;
	s_mantissa_ptr[tid * inc_dst_s_mantissa] = s_mantissa;
}
} // namespace

template <class T, class S_EXP_T, class MANTISSA_T>
int aonfp::cuda::copy_to_device(
		T* const dst_ptr, unsigned inc_dst,
		const S_EXP_T* const s_exp_ptr, const unsigned inc_src_s_exp,
		const MANTISSA_T* const mantissa_ptr, const unsigned inc_src_mantissa,
		const std::size_t N,
		const unsigned block_size,
		const bool set_cpu_affinity_frag,
		cudaStream_t stream) {
	if (set_cpu_affinity_frag) {
		// Set CPU affinity for good performance
		cudaPointerAttributes p_attributes;
		CUDA_CHECK_ERROR(cudaPointerGetAttributes(&p_attributes, dst_ptr));
		set_cpu_affinity(p_attributes.device);
	}

	copy_to_device_kernel<T, S_EXP_T, MANTISSA_T><<<(N + block_size - 1) / block_size, block_size, 0, stream>>>(
			dst_ptr, inc_dst,
			s_exp_ptr, inc_src_s_exp,
			mantissa_ptr, inc_src_mantissa,
			N);

	if (cudaGetLastError() == cudaSuccess) {
		return 0;
	} else {
		CUDA_CHECK_ERROR(cudaGetLastError());
		return 1;
	}
}

template <class S_EXP_T, class MANTISSA_T, class T>
int aonfp::cuda::copy_to_host(
		S_EXP_T* const s_exp_ptr, const unsigned inc_dst_s_exp,
		MANTISSA_T* const mantissa_ptr, const unsigned inc_dst_mantissa,
		const T* const src_ptr, const unsigned inc_src,
		const std::size_t N,
		const unsigned block_size,
		const bool set_cpu_affinity_frag,
		cudaStream_t stream) {
	if (set_cpu_affinity_frag) {
		// Set CPU affinity for good performance
		cudaPointerAttributes p_attributes;
		CUDA_CHECK_ERROR(cudaPointerGetAttributes(&p_attributes, src_ptr));
		set_cpu_affinity(p_attributes.device);
	}

	copy_to_host_kernel<S_EXP_T, MANTISSA_T, T><<<(N + block_size - 1) / block_size, block_size, 0, stream>>>(s_exp_ptr, inc_dst_s_exp, mantissa_ptr, inc_dst_mantissa, src_ptr, inc_src, N);
	if (cudaGetLastError() == cudaSuccess) {
		return 0;
	} else {
		CUDA_CHECK_ERROR(cudaGetLastError());
		return 1;
	}
}

template int aonfp::cuda::copy_to_device<double, uint64_t, uint64_t>(double* const, const unsigned, const uint64_t* const, const unsigned, const uint64_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint32_t, uint64_t>(double* const, const unsigned, const uint32_t* const, const unsigned, const uint64_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint16_t, uint64_t>(double* const, const unsigned, const uint16_t* const, const unsigned, const uint64_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint8_t , uint64_t>(double* const, const unsigned, const uint8_t * const, const unsigned, const uint64_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint64_t, uint32_t>(double* const, const unsigned, const uint64_t* const, const unsigned, const uint32_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint32_t, uint32_t>(double* const, const unsigned, const uint32_t* const, const unsigned, const uint32_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint16_t, uint32_t>(double* const, const unsigned, const uint16_t* const, const unsigned, const uint32_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint8_t , uint32_t>(double* const, const unsigned, const uint8_t * const, const unsigned, const uint32_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint64_t, uint16_t>(double* const, const unsigned, const uint64_t* const, const unsigned, const uint16_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint32_t, uint16_t>(double* const, const unsigned, const uint32_t* const, const unsigned, const uint16_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint16_t, uint16_t>(double* const, const unsigned, const uint16_t* const, const unsigned, const uint16_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint8_t , uint16_t>(double* const, const unsigned, const uint8_t * const, const unsigned, const uint16_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint64_t, uint8_t >(double* const, const unsigned, const uint64_t* const, const unsigned, const uint8_t * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint32_t, uint8_t >(double* const, const unsigned, const uint32_t* const, const unsigned, const uint8_t * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint16_t, uint8_t >(double* const, const unsigned, const uint16_t* const, const unsigned, const uint8_t * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint8_t , uint8_t >(double* const, const unsigned, const uint8_t * const, const unsigned, const uint8_t * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);

template int aonfp::cuda::copy_to_device<float , uint64_t, uint64_t>(float * const, const unsigned, const uint64_t* const, const unsigned, const uint64_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint32_t, uint64_t>(float * const, const unsigned, const uint32_t* const, const unsigned, const uint64_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint16_t, uint64_t>(float * const, const unsigned, const uint16_t* const, const unsigned, const uint64_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint8_t , uint64_t>(float * const, const unsigned, const uint8_t * const, const unsigned, const uint64_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint64_t, uint32_t>(float * const, const unsigned, const uint64_t* const, const unsigned, const uint32_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint32_t, uint32_t>(float * const, const unsigned, const uint32_t* const, const unsigned, const uint32_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint16_t, uint32_t>(float * const, const unsigned, const uint16_t* const, const unsigned, const uint32_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint8_t , uint32_t>(float * const, const unsigned, const uint8_t * const, const unsigned, const uint32_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint64_t, uint16_t>(float * const, const unsigned, const uint64_t* const, const unsigned, const uint16_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint32_t, uint16_t>(float * const, const unsigned, const uint32_t* const, const unsigned, const uint16_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint16_t, uint16_t>(float * const, const unsigned, const uint16_t* const, const unsigned, const uint16_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint8_t , uint16_t>(float * const, const unsigned, const uint8_t * const, const unsigned, const uint16_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint64_t, uint8_t >(float * const, const unsigned, const uint64_t* const, const unsigned, const uint8_t * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint32_t, uint8_t >(float * const, const unsigned, const uint32_t* const, const unsigned, const uint8_t * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint16_t, uint8_t >(float * const, const unsigned, const uint16_t* const, const unsigned, const uint8_t * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint8_t , uint8_t >(float * const, const unsigned, const uint8_t * const, const unsigned, const uint8_t * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);

template int aonfp::cuda::copy_to_host<uint64_t, uint64_t, double>(uint64_t* const, const unsigned, uint64_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint64_t, double>(uint32_t* const, const unsigned, uint64_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint64_t, double>(uint16_t* const, const unsigned, uint64_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint64_t, double>(uint8_t * const, const unsigned, uint64_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint64_t, uint32_t, double>(uint64_t* const, const unsigned, uint32_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint32_t, double>(uint32_t* const, const unsigned, uint32_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint32_t, double>(uint16_t* const, const unsigned, uint32_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint32_t, double>(uint8_t * const, const unsigned, uint32_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint64_t, uint16_t, double>(uint64_t* const, const unsigned, uint16_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint16_t, double>(uint32_t* const, const unsigned, uint16_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint16_t, double>(uint16_t* const, const unsigned, uint16_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint16_t, double>(uint8_t * const, const unsigned, uint16_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint64_t, uint8_t , double>(uint64_t* const, const unsigned, uint8_t * const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint8_t , double>(uint32_t* const, const unsigned, uint8_t * const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint8_t , double>(uint16_t* const, const unsigned, uint8_t * const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint8_t , double>(uint8_t * const, const unsigned, uint8_t * const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);

template int aonfp::cuda::copy_to_host<uint64_t, uint64_t, float >(uint64_t* const, const unsigned, uint64_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint64_t, float >(uint32_t* const, const unsigned, uint64_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint64_t, float >(uint16_t* const, const unsigned, uint64_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint64_t, float >(uint8_t * const, const unsigned, uint64_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint64_t, uint32_t, float >(uint64_t* const, const unsigned, uint32_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint32_t, float >(uint32_t* const, const unsigned, uint32_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint32_t, float >(uint16_t* const, const unsigned, uint32_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint32_t, float >(uint8_t * const, const unsigned, uint32_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint64_t, uint16_t, float >(uint64_t* const, const unsigned, uint16_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint16_t, float >(uint32_t* const, const unsigned, uint16_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint16_t, float >(uint16_t* const, const unsigned, uint16_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint16_t, float >(uint8_t * const, const unsigned, uint16_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint64_t, uint8_t , float >(uint64_t* const, const unsigned, uint8_t * const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint8_t , float >(uint32_t* const, const unsigned, uint8_t * const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint8_t , float >(uint16_t* const, const unsigned, uint8_t * const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint8_t , float >(uint8_t * const, const unsigned, uint8_t * const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);

// For q
template <class T, class EXP_T, class S_MANTISSA_T>
int aonfp::q::cuda::copy_to_device(
		T* const dst_ptr, unsigned inc_dst,
		const EXP_T* const exp_ptr, const unsigned inc_src_exp,
		const S_MANTISSA_T* const s_mantissa_ptr, const unsigned inc_src_s_mantissa,
		const std::size_t N,
		const unsigned block_size,
		const bool set_cpu_affinity_frag,
		cudaStream_t stream) {
	if (set_cpu_affinity_frag) {
		// Set CPU affinity for good performance
		cudaPointerAttributes p_attributes;
		CUDA_CHECK_ERROR(cudaPointerGetAttributes(&p_attributes, dst_ptr));
		set_cpu_affinity(p_attributes.device);
	}

	copy_to_device_q_kernel<T, EXP_T, S_MANTISSA_T><<<(N + block_size - 1) / block_size, block_size, 0, stream>>>(dst_ptr, inc_dst, exp_ptr, inc_src_exp, s_mantissa_ptr, inc_src_s_mantissa, N);
	if (cudaGetLastError() == cudaSuccess) {
		return 0;
	} else {
		CUDA_CHECK_ERROR(cudaGetLastError());
		return 1;
	}
}

template <class EXP_T, class S_MANTISSA_T, class T>
int aonfp::q::cuda::copy_to_host(
		EXP_T* const exp_ptr, const unsigned inc_dst_exp,
		S_MANTISSA_T* const s_mantissa_ptr, const unsigned inc_dst_s_mantissa,
		const T* const src_ptr, const unsigned inc_src,
		const std::size_t N,
		const unsigned block_size,
		const bool set_cpu_affinity_frag,
		cudaStream_t stream) {
	if (set_cpu_affinity_frag) {
		// Set CPU affinity for good performance
		cudaPointerAttributes p_attributes;
		CUDA_CHECK_ERROR(cudaPointerGetAttributes(&p_attributes, src_ptr));
		set_cpu_affinity(p_attributes.device);
	}

	copy_to_host_q_kernel<EXP_T, S_MANTISSA_T, T><<<(N + block_size - 1) / block_size, block_size, 0, stream>>>(exp_ptr, inc_dst_exp, s_mantissa_ptr, inc_dst_s_mantissa, src_ptr, inc_src, N);
	if (cudaGetLastError() == cudaSuccess) {
		return 0;
	} else {
		CUDA_CHECK_ERROR(cudaGetLastError());
		return 1;
	}
}

template int aonfp::q::cuda::copy_to_device<double, uint64_t, uint64_t>(double* const, const unsigned, const uint64_t* const, const unsigned, const uint64_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint32_t, uint64_t>(double* const, const unsigned, const uint32_t* const, const unsigned, const uint64_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint16_t, uint64_t>(double* const, const unsigned, const uint16_t* const, const unsigned, const uint64_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint8_t , uint64_t>(double* const, const unsigned, const uint8_t * const, const unsigned, const uint64_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint64_t, uint32_t>(double* const, const unsigned, const uint64_t* const, const unsigned, const uint32_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint32_t, uint32_t>(double* const, const unsigned, const uint32_t* const, const unsigned, const uint32_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint16_t, uint32_t>(double* const, const unsigned, const uint16_t* const, const unsigned, const uint32_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint8_t , uint32_t>(double* const, const unsigned, const uint8_t * const, const unsigned, const uint32_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint64_t, uint16_t>(double* const, const unsigned, const uint64_t* const, const unsigned, const uint16_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint32_t, uint16_t>(double* const, const unsigned, const uint32_t* const, const unsigned, const uint16_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint16_t, uint16_t>(double* const, const unsigned, const uint16_t* const, const unsigned, const uint16_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint8_t , uint16_t>(double* const, const unsigned, const uint8_t * const, const unsigned, const uint16_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint64_t, uint8_t >(double* const, const unsigned, const uint64_t* const, const unsigned, const uint8_t * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint32_t, uint8_t >(double* const, const unsigned, const uint32_t* const, const unsigned, const uint8_t * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint16_t, uint8_t >(double* const, const unsigned, const uint16_t* const, const unsigned, const uint8_t * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint8_t , uint8_t >(double* const, const unsigned, const uint8_t * const, const unsigned, const uint8_t * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);

template int aonfp::q::cuda::copy_to_device<float , uint64_t, uint64_t>(float * const, const unsigned, const uint64_t* const, const unsigned, const uint64_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint32_t, uint64_t>(float * const, const unsigned, const uint32_t* const, const unsigned, const uint64_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint16_t, uint64_t>(float * const, const unsigned, const uint16_t* const, const unsigned, const uint64_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint8_t , uint64_t>(float * const, const unsigned, const uint8_t * const, const unsigned, const uint64_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint64_t, uint32_t>(float * const, const unsigned, const uint64_t* const, const unsigned, const uint32_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint32_t, uint32_t>(float * const, const unsigned, const uint32_t* const, const unsigned, const uint32_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint16_t, uint32_t>(float * const, const unsigned, const uint16_t* const, const unsigned, const uint32_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint8_t , uint32_t>(float * const, const unsigned, const uint8_t * const, const unsigned, const uint32_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint64_t, uint16_t>(float * const, const unsigned, const uint64_t* const, const unsigned, const uint16_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint32_t, uint16_t>(float * const, const unsigned, const uint32_t* const, const unsigned, const uint16_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint16_t, uint16_t>(float * const, const unsigned, const uint16_t* const, const unsigned, const uint16_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint8_t , uint16_t>(float * const, const unsigned, const uint8_t * const, const unsigned, const uint16_t* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint64_t, uint8_t >(float * const, const unsigned, const uint64_t* const, const unsigned, const uint8_t * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint32_t, uint8_t >(float * const, const unsigned, const uint32_t* const, const unsigned, const uint8_t * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint16_t, uint8_t >(float * const, const unsigned, const uint16_t* const, const unsigned, const uint8_t * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint8_t , uint8_t >(float * const, const unsigned, const uint8_t * const, const unsigned, const uint8_t * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);

template int aonfp::q::cuda::copy_to_host<uint64_t, uint64_t, double>(uint64_t* const, const unsigned, uint64_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint32_t, uint64_t, double>(uint32_t* const, const unsigned, uint64_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint16_t, uint64_t, double>(uint16_t* const, const unsigned, uint64_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint8_t , uint64_t, double>(uint8_t * const, const unsigned, uint64_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint64_t, uint32_t, double>(uint64_t* const, const unsigned, uint32_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint32_t, uint32_t, double>(uint32_t* const, const unsigned, uint32_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint16_t, uint32_t, double>(uint16_t* const, const unsigned, uint32_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint8_t , uint32_t, double>(uint8_t * const, const unsigned, uint32_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint64_t, uint16_t, double>(uint64_t* const, const unsigned, uint16_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint32_t, uint16_t, double>(uint32_t* const, const unsigned, uint16_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint16_t, uint16_t, double>(uint16_t* const, const unsigned, uint16_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint8_t , uint16_t, double>(uint8_t * const, const unsigned, uint16_t* const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint64_t, uint8_t , double>(uint64_t* const, const unsigned, uint8_t * const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint32_t, uint8_t , double>(uint32_t* const, const unsigned, uint8_t * const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint16_t, uint8_t , double>(uint16_t* const, const unsigned, uint8_t * const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint8_t , uint8_t , double>(uint8_t * const, const unsigned, uint8_t * const, const unsigned, const double* const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);

template int aonfp::q::cuda::copy_to_host<uint64_t, uint64_t, float >(uint64_t* const, const unsigned, uint64_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint32_t, uint64_t, float >(uint32_t* const, const unsigned, uint64_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint16_t, uint64_t, float >(uint16_t* const, const unsigned, uint64_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint8_t , uint64_t, float >(uint8_t * const, const unsigned, uint64_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint64_t, uint32_t, float >(uint64_t* const, const unsigned, uint32_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint32_t, uint32_t, float >(uint32_t* const, const unsigned, uint32_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint16_t, uint32_t, float >(uint16_t* const, const unsigned, uint32_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint8_t , uint32_t, float >(uint8_t * const, const unsigned, uint32_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint64_t, uint16_t, float >(uint64_t* const, const unsigned, uint16_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint32_t, uint16_t, float >(uint32_t* const, const unsigned, uint16_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint16_t, uint16_t, float >(uint16_t* const, const unsigned, uint16_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint8_t , uint16_t, float >(uint8_t * const, const unsigned, uint16_t* const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint64_t, uint8_t , float >(uint64_t* const, const unsigned, uint8_t * const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint32_t, uint8_t , float >(uint32_t* const, const unsigned, uint8_t * const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint16_t, uint8_t , float >(uint16_t* const, const unsigned, uint8_t * const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint8_t , uint8_t , float >(uint8_t * const, const unsigned, uint8_t * const, const unsigned, const float * const, const unsigned, const std::size_t, const unsigned, const bool, cudaStream_t);
