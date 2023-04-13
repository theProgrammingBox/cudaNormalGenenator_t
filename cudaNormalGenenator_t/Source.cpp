#include "cudaNormalGenerator_t.cuh"

int main()
{
	printf("cuDNN version: %d.%d.%d\n", CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL);
	printf("cuBLAS version: %d.%d.%d\n", CUBLAS_VER_MAJOR, CUBLAS_VER_MINOR, CUBLAS_VER_PATCH);
	printf("cuRAND version: %d.%d.%d\n", CURAND_VERSION / 1000, (CURAND_VERSION % 1000) / 100, CURAND_VERSION % 100);
	printf("\n");
	
	cudaNormalGenerator_t normalGenerator;
	cudaNormalGeneratorCreate(&normalGenerator);
	cudaNormalGeneratorSetSeed(&normalGenerator, 0);

	printf("Normal generator created.\n");

	const uint32_t size = 1024;
	__half* arr;
	cudaMalloc(&arr, size * sizeof(__half));
	CurandGenerateNormalF16(&normalGenerator, arr, size, 0.0f, 1.0f);
	
	__half* arrHost = (__half*)malloc(size * sizeof(__half));
	cudaMemcpy(arrHost, arr, size * sizeof(__half), cudaMemcpyDeviceToHost);
	
	
	return 0;
}