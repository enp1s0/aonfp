#include <iostream>
#include <chrono>
#include <random>
#include <aonfp/aonfp.hpp>
#include <aonfp/cuda_copy.hpp>

using DEVICE_T = double;
using S_EXP_T = uint32_t;
using MANTISSA_T = uint32_t;

constexpr std::size_t N = 1lu << 20;

__global__ void add_one_kernel(DEVICE_T *ptr) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= N) {
		return;
	}

	ptr[tid] += static_cast<DEVICE_T>(1);
}

int main() {
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<DEVICE_T> dist(-1000.0, 1000.0);

	DEVICE_T* device_array;
	S_EXP_T* src_s_exp_array;
	S_EXP_T* dst_s_exp_array;
	MANTISSA_T* src_mantissa_array;
	MANTISSA_T* dst_mantissa_array;

	cudaMalloc(&device_array, sizeof(DEVICE_T) * N);
	cudaMallocHost(&src_s_exp_array, sizeof(S_EXP_T) * N);
	cudaMallocHost(&dst_s_exp_array, sizeof(S_EXP_T) * N);
	cudaMallocHost(&src_mantissa_array, sizeof(MANTISSA_T) * N);
	cudaMallocHost(&dst_mantissa_array, sizeof(MANTISSA_T) * N);

	for (std::size_t i = 0; i < N; i++) {
		aonfp::decompose(src_s_exp_array[i], src_mantissa_array[i], dist(mt));
	}

	{
		cudaDeviceSynchronize();
		const auto start_clock = std::chrono::system_clock::now();
		aonfp::cuda::copy_to_device(device_array, src_s_exp_array, src_mantissa_array, N);
		cudaDeviceSynchronize();
		const auto end_clock = std::chrono::system_clock::now();

		const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1.e-6;

		std::printf("Host => Device : %e [GiB/s]\n", N * (sizeof(S_EXP_T) + sizeof(MANTISSA_T)) / elapsed_time / (1lu << 30));
	}

	constexpr std::size_t block_size = 1lu << 8;
	add_one_kernel<<<(N + block_size + 1) / block_size, block_size>>>(device_array);

	{
		cudaDeviceSynchronize();
		const auto start_clock = std::chrono::system_clock::now();
		aonfp::cuda::copy_to_host(dst_s_exp_array, dst_mantissa_array, device_array, N);
		cudaDeviceSynchronize();
		const auto end_clock = std::chrono::system_clock::now();

		const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1.e-6;

		std::printf("Device => Host : %e [GiB/s]\n", N * (sizeof(S_EXP_T) + sizeof(MANTISSA_T)) / elapsed_time / (1lu << 30));
	}

	double max_error = 0;
	for (std::size_t i = 0; i < N; i++) {
		const auto src = aonfp::compose<double>(src_s_exp_array[i], src_mantissa_array[i]);
		const auto dst = aonfp::compose<double>(dst_s_exp_array[i], dst_mantissa_array[i]);
		const auto error = std::abs((src + 1.0) - dst);
		max_error = std::max(max_error, error);
	}
	std::printf("max_error : %e\n", max_error);

	cudaFreeHost(src_s_exp_array);
	cudaFreeHost(src_mantissa_array);
	cudaFreeHost(dst_s_exp_array);
	cudaFreeHost(dst_mantissa_array);

	// Memcpy bandwidth test
	DEVICE_T* host_device_array;
	cudaMallocHost(&host_device_array, sizeof(DEVICE_T) * N);
	{
		cudaDeviceSynchronize();
		const auto start_clock = std::chrono::system_clock::now();
		cudaMemcpy(device_array, host_device_array, sizeof(DEVICE_T) * N, cudaMemcpyDefault);
		cudaDeviceSynchronize();
		const auto end_clock = std::chrono::system_clock::now();

		const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1.e-6;

		std::printf("[cudaMemcpy] Host => Device : %e [GiB/s]\n", N * sizeof(DEVICE_T) / elapsed_time / (1lu << 30));
	}
	{
		cudaDeviceSynchronize();
		const auto start_clock = std::chrono::system_clock::now();
		cudaMemcpy(host_device_array, device_array, sizeof(DEVICE_T) * N, cudaMemcpyDefault);
		cudaDeviceSynchronize();
		const auto end_clock = std::chrono::system_clock::now();

		const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1.e-6;

		std::printf("[cudaMemcpy] Device => Host : %e [GiB/s]\n", N * sizeof(DEVICE_T) / elapsed_time / (1lu << 30));
	}

	cudaFreeHost(host_device_array);
	cudaFree(device_array);
}
