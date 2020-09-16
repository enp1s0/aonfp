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

std::string get_cuda_path(const int device_id) {
	constexpr std::size_t busid_size = sizeof("0000:00:00.0");
	constexpr std::size_t busid_reduced_size = sizeof("0000:00");

	char busid[busid_size];
	cudaDeviceGetPCIBusId(busid, busid_size, device_id);
	std::string busid_str = [](std::string str) -> std::string {
		std::transform(str.begin(), str.end(), str.begin(),
				[](const unsigned c) {return std::tolower(c);});
		return str;
	}(busid);
	const std::string path = "/sys/class/pci_bus/" + busid_str.substr(0, busid_reduced_size - 1) + "/../../" + busid_str;

	const auto real_path = realpath(path.c_str(), nullptr);
	if (real_path == nullptr) {
		throw std::runtime_error("Could not find real path of " + path);
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

std::string cpuset_to_str(const cpu_set_t cpuset) {
	unsigned c = 0;
	auto m8 = reinterpret_cast<const uint8_t*>(&cpuset);

	std::string str;

	for (int o = sizeof(cpu_set_t) - 1; o >= 0; o--) {
		if (c == 0 && m8[o] == 0)
			continue;
		char cc[3] = {0};
		str += cc;
		c += 2;
		if (o && o % 4 == 0) {
			str += ",";
			c++;
		}
	}
	return str;
}

cpu_set_t get_cpu_gpu_affinity(const int device_id) {
	cpu_set_t mask;
	memset(&mask, 0, sizeof(cpu_set_t));

	const auto cuda_path = get_cuda_path(device_id);
	const auto path = cuda_path + "/local_cpus";

	int fd = open(path.c_str(), O_RDONLY);
	if (fd < 0) {
		throw std::runtime_error("Could not open " + path);
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
		const auto affinity_str = cpuset_to_str(final_mask);
		sched_setaffinity(0, sizeof(cpu_set_t), &final_mask);
	}
}

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

template <class T, class S_EXP_T, class MANTISSA_T>
__global__ void copy_to_device_q_kernel(T *const dst_ptr, const S_EXP_T *const s_exp_ptr, const MANTISSA_T *const mantissa_ptr, const std::size_t N) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N) {
		return;
	}
	const auto s_exp = s_exp_ptr[tid];
	const auto mantissa = mantissa_ptr[tid];

	dst_ptr[tid] = aonfp::q::compose<T>(s_exp, mantissa);
}

template <class S_EXP_T, class MANTISSA_T, class T>
__global__ void copy_to_host_q_kernel(S_EXP_T *const s_exp_ptr, MANTISSA_T *const mantissa_ptr, const T* const src_ptr, const std::size_t N) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N) {
		return;
	}
	S_EXP_T s_exp;
	MANTISSA_T mantissa;

	aonfp::q::decompose(s_exp, mantissa, src_ptr[tid]);

	s_exp_ptr[tid] = s_exp;
	mantissa_ptr[tid] = mantissa;
}
} // namespace

template <class T, class S_EXP_T, class MANTISSA_T>
int aonfp::cuda::copy_to_device(T *const dst_ptr, const S_EXP_T *const s_exp_ptr, const MANTISSA_T *const mantissa_ptr, const std::size_t N, const bool set_cpu_affinity_frag, cudaStream_t stream) {
	if (set_cpu_affinity_frag) {
		// Set CPU affinity for good performance
		cudaPointerAttributes p_attributes;
		cudaPointerGetAttributes(&p_attributes, dst_ptr);
		set_cpu_affinity(p_attributes.device);
	}

	constexpr std::size_t block_size = 1024;
	copy_to_device_kernel<T, S_EXP_T, MANTISSA_T><<<(N + block_size - 1) / block_size, block_size, 0, stream>>>(dst_ptr, s_exp_ptr, mantissa_ptr, N);
	if (cudaGetLastError() == cudaSuccess) {
		return 0;
	} else {
		return 1;
	}
}

template <class S_EXP_T, class MANTISSA_T, class T>
int aonfp::cuda::copy_to_host(S_EXP_T *const s_exp_ptr, MANTISSA_T *const mantissa_ptr, const T *const src_ptr, const std::size_t N, const bool set_cpu_affinity_frag, cudaStream_t stream) {
	if (set_cpu_affinity_frag) {
		// Set CPU affinity for good performance
		cudaPointerAttributes p_attributes;
		cudaPointerGetAttributes(&p_attributes, src_ptr);
		set_cpu_affinity(p_attributes.device);
	}

	constexpr std::size_t block_size = 1024;
	copy_to_host_kernel<S_EXP_T, MANTISSA_T, T><<<(N + block_size - 1) / block_size, block_size, 0, stream>>>(s_exp_ptr, mantissa_ptr, src_ptr, N);
	if (cudaGetLastError() == cudaSuccess) {
		return 0;
	} else {
		return 1;
	}
}

template int aonfp::cuda::copy_to_device<double, uint64_t, uint64_t>(double* const, const uint64_t* const, const uint64_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint32_t, uint64_t>(double* const, const uint32_t* const, const uint64_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint16_t, uint64_t>(double* const, const uint16_t* const, const uint64_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint8_t , uint64_t>(double* const, const uint8_t * const, const uint64_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint64_t, uint32_t>(double* const, const uint64_t* const, const uint32_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint32_t, uint32_t>(double* const, const uint32_t* const, const uint32_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint16_t, uint32_t>(double* const, const uint16_t* const, const uint32_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint8_t , uint32_t>(double* const, const uint8_t * const, const uint32_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint64_t, uint16_t>(double* const, const uint64_t* const, const uint16_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint32_t, uint16_t>(double* const, const uint32_t* const, const uint16_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint16_t, uint16_t>(double* const, const uint16_t* const, const uint16_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint8_t , uint16_t>(double* const, const uint8_t * const, const uint16_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint64_t, uint8_t >(double* const, const uint64_t* const, const uint8_t * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint32_t, uint8_t >(double* const, const uint32_t* const, const uint8_t * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint16_t, uint8_t >(double* const, const uint16_t* const, const uint8_t * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint8_t , uint8_t >(double* const, const uint8_t * const, const uint8_t * const, const std::size_t, const bool, cudaStream_t);

template int aonfp::cuda::copy_to_device<float , uint64_t, uint64_t>(float * const, const uint64_t* const, const uint64_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint32_t, uint64_t>(float * const, const uint32_t* const, const uint64_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint16_t, uint64_t>(float * const, const uint16_t* const, const uint64_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint8_t , uint64_t>(float * const, const uint8_t * const, const uint64_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint64_t, uint32_t>(float * const, const uint64_t* const, const uint32_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint32_t, uint32_t>(float * const, const uint32_t* const, const uint32_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint16_t, uint32_t>(float * const, const uint16_t* const, const uint32_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint8_t , uint32_t>(float * const, const uint8_t * const, const uint32_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint64_t, uint16_t>(float * const, const uint64_t* const, const uint16_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint32_t, uint16_t>(float * const, const uint32_t* const, const uint16_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint16_t, uint16_t>(float * const, const uint16_t* const, const uint16_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint8_t , uint16_t>(float * const, const uint8_t * const, const uint16_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint64_t, uint8_t >(float * const, const uint64_t* const, const uint8_t * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint32_t, uint8_t >(float * const, const uint32_t* const, const uint8_t * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint16_t, uint8_t >(float * const, const uint16_t* const, const uint8_t * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint8_t , uint8_t >(float * const, const uint8_t * const, const uint8_t * const, const std::size_t, const bool, cudaStream_t);

template int aonfp::cuda::copy_to_host<uint64_t, uint64_t, double>(uint64_t* const, uint64_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint64_t, double>(uint32_t* const, uint64_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint64_t, double>(uint16_t* const, uint64_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint64_t, double>(uint8_t * const, uint64_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint64_t, uint32_t, double>(uint64_t* const, uint32_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint32_t, double>(uint32_t* const, uint32_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint32_t, double>(uint16_t* const, uint32_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint32_t, double>(uint8_t * const, uint32_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint64_t, uint16_t, double>(uint64_t* const, uint16_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint16_t, double>(uint32_t* const, uint16_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint16_t, double>(uint16_t* const, uint16_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint16_t, double>(uint8_t * const, uint16_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint64_t, uint8_t , double>(uint64_t* const, uint8_t * const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint8_t , double>(uint32_t* const, uint8_t * const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint8_t , double>(uint16_t* const, uint8_t * const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint8_t , double>(uint8_t * const, uint8_t * const, const double* const, const std::size_t, const bool, cudaStream_t);

template int aonfp::cuda::copy_to_host<uint64_t, uint64_t, float >(uint64_t* const, uint64_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint64_t, float >(uint32_t* const, uint64_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint64_t, float >(uint16_t* const, uint64_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint64_t, float >(uint8_t * const, uint64_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint64_t, uint32_t, float >(uint64_t* const, uint32_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint32_t, float >(uint32_t* const, uint32_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint32_t, float >(uint16_t* const, uint32_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint32_t, float >(uint8_t * const, uint32_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint64_t, uint16_t, float >(uint64_t* const, uint16_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint16_t, float >(uint32_t* const, uint16_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint16_t, float >(uint16_t* const, uint16_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint16_t, float >(uint8_t * const, uint16_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint64_t, uint8_t , float >(uint64_t* const, uint8_t * const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint8_t , float >(uint32_t* const, uint8_t * const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint8_t , float >(uint16_t* const, uint8_t * const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint8_t , float >(uint8_t * const, uint8_t * const, const float * const, const std::size_t, const bool, cudaStream_t);

// For q
template <class T, class EXP_T, class S_MANTISSA_T>
int aonfp::q::cuda::copy_to_device(T *const dst_ptr, const EXP_T *const exp_ptr, const S_MANTISSA_T *const s_mantissa_ptr, const std::size_t N, const bool set_cpu_affinity_frag, cudaStream_t stream) {
	if (set_cpu_affinity_frag) {
		// Set CPU affinity for good performance
		cudaPointerAttributes p_attributes;
		cudaPointerGetAttributes(&p_attributes, dst_ptr);
		set_cpu_affinity(p_attributes.device);
	}

	constexpr std::size_t block_size = 1024;
	copy_to_device_q_kernel<T, EXP_T, S_MANTISSA_T><<<(N + block_size - 1) / block_size, block_size, 0, stream>>>(dst_ptr, exp_ptr, s_mantissa_ptr, N);
	if (cudaGetLastError() == cudaSuccess) {
		return 0;
	} else {
		return 1;
	}
}

template <class EXP_T, class S_MANTISSA_T, class T>
int aonfp::q::cuda::copy_to_host(EXP_T *const exp_ptr, S_MANTISSA_T *const s_mantissa_ptr, const T *const src_ptr, const std::size_t N, const bool set_cpu_affinity_frag, cudaStream_t stream) {
	if (set_cpu_affinity_frag) {
		// Set CPU affinity for good performance
		cudaPointerAttributes p_attributes;
		cudaPointerGetAttributes(&p_attributes, src_ptr);
		set_cpu_affinity(p_attributes.device);
	}

	constexpr std::size_t block_size = 1024;
	copy_to_host_q_kernel<EXP_T, S_MANTISSA_T, T><<<(N + block_size - 1) / block_size, block_size, 0, stream>>>(exp_ptr, s_mantissa_ptr, src_ptr, N);
	if (cudaGetLastError() == cudaSuccess) {
		return 0;
	} else {
		return 1;
	}
}

template int aonfp::q::cuda::copy_to_device<double, uint64_t, uint64_t>(double* const, const uint64_t* const, const uint64_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint32_t, uint64_t>(double* const, const uint32_t* const, const uint64_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint16_t, uint64_t>(double* const, const uint16_t* const, const uint64_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint8_t , uint64_t>(double* const, const uint8_t * const, const uint64_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint64_t, uint32_t>(double* const, const uint64_t* const, const uint32_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint32_t, uint32_t>(double* const, const uint32_t* const, const uint32_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint16_t, uint32_t>(double* const, const uint16_t* const, const uint32_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint8_t , uint32_t>(double* const, const uint8_t * const, const uint32_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint64_t, uint16_t>(double* const, const uint64_t* const, const uint16_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint32_t, uint16_t>(double* const, const uint32_t* const, const uint16_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint16_t, uint16_t>(double* const, const uint16_t* const, const uint16_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint8_t , uint16_t>(double* const, const uint8_t * const, const uint16_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint64_t, uint8_t >(double* const, const uint64_t* const, const uint8_t * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint32_t, uint8_t >(double* const, const uint32_t* const, const uint8_t * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint16_t, uint8_t >(double* const, const uint16_t* const, const uint8_t * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<double, uint8_t , uint8_t >(double* const, const uint8_t * const, const uint8_t * const, const std::size_t, const bool, cudaStream_t);

template int aonfp::q::cuda::copy_to_device<float , uint64_t, uint64_t>(float * const, const uint64_t* const, const uint64_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint32_t, uint64_t>(float * const, const uint32_t* const, const uint64_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint16_t, uint64_t>(float * const, const uint16_t* const, const uint64_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint8_t , uint64_t>(float * const, const uint8_t * const, const uint64_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint64_t, uint32_t>(float * const, const uint64_t* const, const uint32_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint32_t, uint32_t>(float * const, const uint32_t* const, const uint32_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint16_t, uint32_t>(float * const, const uint16_t* const, const uint32_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint8_t , uint32_t>(float * const, const uint8_t * const, const uint32_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint64_t, uint16_t>(float * const, const uint64_t* const, const uint16_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint32_t, uint16_t>(float * const, const uint32_t* const, const uint16_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint16_t, uint16_t>(float * const, const uint16_t* const, const uint16_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint8_t , uint16_t>(float * const, const uint8_t * const, const uint16_t* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint64_t, uint8_t >(float * const, const uint64_t* const, const uint8_t * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint32_t, uint8_t >(float * const, const uint32_t* const, const uint8_t * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint16_t, uint8_t >(float * const, const uint16_t* const, const uint8_t * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_device<float , uint8_t , uint8_t >(float * const, const uint8_t * const, const uint8_t * const, const std::size_t, const bool, cudaStream_t);

template int aonfp::q::cuda::copy_to_host<uint64_t, uint64_t, double>(uint64_t* const, uint64_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint32_t, uint64_t, double>(uint32_t* const, uint64_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint16_t, uint64_t, double>(uint16_t* const, uint64_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint8_t , uint64_t, double>(uint8_t * const, uint64_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint64_t, uint32_t, double>(uint64_t* const, uint32_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint32_t, uint32_t, double>(uint32_t* const, uint32_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint16_t, uint32_t, double>(uint16_t* const, uint32_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint8_t , uint32_t, double>(uint8_t * const, uint32_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint64_t, uint16_t, double>(uint64_t* const, uint16_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint32_t, uint16_t, double>(uint32_t* const, uint16_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint16_t, uint16_t, double>(uint16_t* const, uint16_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint8_t , uint16_t, double>(uint8_t * const, uint16_t* const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint64_t, uint8_t , double>(uint64_t* const, uint8_t * const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint32_t, uint8_t , double>(uint32_t* const, uint8_t * const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint16_t, uint8_t , double>(uint16_t* const, uint8_t * const, const double* const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint8_t , uint8_t , double>(uint8_t * const, uint8_t * const, const double* const, const std::size_t, const bool, cudaStream_t);

template int aonfp::q::cuda::copy_to_host<uint64_t, uint64_t, float >(uint64_t* const, uint64_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint32_t, uint64_t, float >(uint32_t* const, uint64_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint16_t, uint64_t, float >(uint16_t* const, uint64_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint8_t , uint64_t, float >(uint8_t * const, uint64_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint64_t, uint32_t, float >(uint64_t* const, uint32_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint32_t, uint32_t, float >(uint32_t* const, uint32_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint16_t, uint32_t, float >(uint16_t* const, uint32_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint8_t , uint32_t, float >(uint8_t * const, uint32_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint64_t, uint16_t, float >(uint64_t* const, uint16_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint32_t, uint16_t, float >(uint32_t* const, uint16_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint16_t, uint16_t, float >(uint16_t* const, uint16_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint8_t , uint16_t, float >(uint8_t * const, uint16_t* const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint64_t, uint8_t , float >(uint64_t* const, uint8_t * const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint32_t, uint8_t , float >(uint32_t* const, uint8_t * const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint16_t, uint8_t , float >(uint16_t* const, uint8_t * const, const float * const, const std::size_t, const bool, cudaStream_t);
template int aonfp::q::cuda::copy_to_host<uint8_t , uint8_t , float >(uint8_t * const, uint8_t * const, const float * const, const std::size_t, const bool, cudaStream_t);
