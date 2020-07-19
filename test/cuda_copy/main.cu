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
	S_EXP_T* s_exp_array;
	MANTISSA_T* mantissa_array;

	cudaMalloc(&device_array, sizeof(DEVICE_T) * N);
	cudaMallocHost(&s_exp_array, sizeof(S_EXP_T) * N);
	cudaMallocHost(&mantissa_array, sizeof(MANTISSA_T) * N);

	for (std::size_t i = 0; i < N; i++) {
		aonfp::decompose(s_exp_array[i], mantissa_array[i], dist(mt));
	}

	{
		cudaDeviceSynchronize();
		const auto start_clock = std::chrono::system_clock::now();
		aonfp::cuda::copy_to_device(device_array, s_exp_array, mantissa_array, N);
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
		aonfp::cuda::copy_to_host(s_exp_array, mantissa_array, device_array, N);
		cudaDeviceSynchronize();
		const auto end_clock = std::chrono::system_clock::now();

		const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1.e-6;

		std::printf("Device => Host : %e [GiB/s]\n", N * (sizeof(S_EXP_T) + sizeof(MANTISSA_T)) / elapsed_time / (1lu << 30));
	}

	cudaFreeHost(s_exp_array);
	cudaFreeHost(mantissa_array);
}
