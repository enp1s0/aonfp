#include <algorithm>
#include <string>
#include <exception>
#include <sched.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <aonfp/aonfp.hpp>
#include <aonfp/cuda_copy.hpp>

namespace {

std::string get_cuda_path(const int device_id) {
	constexpr std::size_t busid_size = sizeof("0000:00:00.0");
	constexpr std::size_t busid_reduced_size = sizeof("0000:00");

	char busid[busid_size];
	cudaDeviceGetPCIBusId(busid, busid_size, device_id);
	std::string busid_str = [](std::string str) -> std::string {
		std::transform(str.begin(), str.end(), str.begin(),
				[](unsigned c) {return std::tolower(c);});
		return str;
	}(busid);
	const std::string path = "/sys/class/pci_bus/" + busid_str.substr(0, busid_reduced_size) + "/../../" + busid_str;

	const auto real_path = realpath(path.c_str(), nullptr);
	if (real_path == nullptr) {
		throw std::runtime_error("Could not find real path of " + path);
	}

	return std::string{real_path};
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
	constexpr std::size_t block_size = 1024;
	copy_to_device_kernel<T, S_EXP_T, MANTISSA_T><<<(N + block_size - 1) / block_size, block_size, 0, stream>>>(dst_ptr, s_exp_ptr, mantissa_ptr, N);
	if (cudaGetLastError() == cudaSuccess) {
		return 0;
	} else {
		return 1;
	}
}

template <class S_EXP_T, class MANTISSA_T, class T>
int aonfp::cuda::copy_to_host(S_EXP_T *const s_exp_ptr, MANTISSA_T *const mantissa_ptr, const T *const src_ptr, const std::size_t N, cudaStream_t stream) {
	constexpr std::size_t block_size = 1024;
	copy_to_host_kernel<S_EXP_T, MANTISSA_T, T><<<(N + block_size - 1) / block_size, block_size, 0, stream>>>(s_exp_ptr, mantissa_ptr, src_ptr, N);
	if (cudaGetLastError() == cudaSuccess) {
		return 0;
	} else {
		return 1;
	}
}

template int aonfp::cuda::copy_to_device<double, uint64_t, uint64_t>(double* const, const uint64_t* const, const uint64_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint32_t, uint64_t>(double* const, const uint32_t* const, const uint64_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint16_t, uint64_t>(double* const, const uint16_t* const, const uint64_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint8_t , uint64_t>(double* const, const uint8_t * const, const uint64_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint64_t, uint32_t>(double* const, const uint64_t* const, const uint32_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint32_t, uint32_t>(double* const, const uint32_t* const, const uint32_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint16_t, uint32_t>(double* const, const uint16_t* const, const uint32_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint8_t , uint32_t>(double* const, const uint8_t * const, const uint32_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint64_t, uint16_t>(double* const, const uint64_t* const, const uint16_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint32_t, uint16_t>(double* const, const uint32_t* const, const uint16_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint16_t, uint16_t>(double* const, const uint16_t* const, const uint16_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint8_t , uint16_t>(double* const, const uint8_t * const, const uint16_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint64_t, uint8_t >(double* const, const uint64_t* const, const uint8_t * const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint32_t, uint8_t >(double* const, const uint32_t* const, const uint8_t * const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint16_t, uint8_t >(double* const, const uint16_t* const, const uint8_t * const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<double, uint8_t , uint8_t >(double* const, const uint8_t * const, const uint8_t * const, const std::size_t, cudaStream_t);

template int aonfp::cuda::copy_to_device<float , uint64_t, uint64_t>(float * const, const uint64_t* const, const uint64_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint32_t, uint64_t>(float * const, const uint32_t* const, const uint64_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint16_t, uint64_t>(float * const, const uint16_t* const, const uint64_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint8_t , uint64_t>(float * const, const uint8_t * const, const uint64_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint64_t, uint32_t>(float * const, const uint64_t* const, const uint32_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint32_t, uint32_t>(float * const, const uint32_t* const, const uint32_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint16_t, uint32_t>(float * const, const uint16_t* const, const uint32_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint8_t , uint32_t>(float * const, const uint8_t * const, const uint32_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint64_t, uint16_t>(float * const, const uint64_t* const, const uint16_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint32_t, uint16_t>(float * const, const uint32_t* const, const uint16_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint16_t, uint16_t>(float * const, const uint16_t* const, const uint16_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint8_t , uint16_t>(float * const, const uint8_t * const, const uint16_t* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint64_t, uint8_t >(float * const, const uint64_t* const, const uint8_t * const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint32_t, uint8_t >(float * const, const uint32_t* const, const uint8_t * const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint16_t, uint8_t >(float * const, const uint16_t* const, const uint8_t * const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_device<float , uint8_t , uint8_t >(float * const, const uint8_t * const, const uint8_t * const, const std::size_t, cudaStream_t);

template int aonfp::cuda::copy_to_host<uint64_t, uint64_t, double>(uint64_t* const, uint64_t* const, const double* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint64_t, double>(uint32_t* const, uint64_t* const, const double* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint64_t, double>(uint16_t* const, uint64_t* const, const double* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint64_t, double>(uint8_t * const, uint64_t* const, const double* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint64_t, uint32_t, double>(uint64_t* const, uint32_t* const, const double* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint32_t, double>(uint32_t* const, uint32_t* const, const double* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint32_t, double>(uint16_t* const, uint32_t* const, const double* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint32_t, double>(uint8_t * const, uint32_t* const, const double* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint64_t, uint16_t, double>(uint64_t* const, uint16_t* const, const double* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint16_t, double>(uint32_t* const, uint16_t* const, const double* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint16_t, double>(uint16_t* const, uint16_t* const, const double* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint16_t, double>(uint8_t * const, uint16_t* const, const double* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint64_t, uint8_t , double>(uint64_t* const, uint8_t * const, const double* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint8_t , double>(uint32_t* const, uint8_t * const, const double* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint8_t , double>(uint16_t* const, uint8_t * const, const double* const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint8_t , double>(uint8_t * const, uint8_t * const, const double* const, const std::size_t, cudaStream_t);

template int aonfp::cuda::copy_to_host<uint64_t, uint64_t, float >(uint64_t* const, uint64_t* const, const float * const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint64_t, float >(uint32_t* const, uint64_t* const, const float * const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint64_t, float >(uint16_t* const, uint64_t* const, const float * const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint64_t, float >(uint8_t * const, uint64_t* const, const float * const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint64_t, uint32_t, float >(uint64_t* const, uint32_t* const, const float * const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint32_t, float >(uint32_t* const, uint32_t* const, const float * const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint32_t, float >(uint16_t* const, uint32_t* const, const float * const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint32_t, float >(uint8_t * const, uint32_t* const, const float * const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint64_t, uint16_t, float >(uint64_t* const, uint16_t* const, const float * const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint16_t, float >(uint32_t* const, uint16_t* const, const float * const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint16_t, float >(uint16_t* const, uint16_t* const, const float * const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint16_t, float >(uint8_t * const, uint16_t* const, const float * const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint64_t, uint8_t , float >(uint64_t* const, uint8_t * const, const float * const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint32_t, uint8_t , float >(uint32_t* const, uint8_t * const, const float * const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint16_t, uint8_t , float >(uint16_t* const, uint8_t * const, const float * const, const std::size_t, cudaStream_t);
template int aonfp::cuda::copy_to_host<uint8_t , uint8_t , float >(uint8_t * const, uint8_t * const, const float * const, const std::size_t, cudaStream_t);
