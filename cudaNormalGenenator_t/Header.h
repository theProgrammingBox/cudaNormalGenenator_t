#pragma once
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <iostream>

void CurandGenerateUniformI16(curandGenerator_t generator, __half* output, uint32_t size, float min = -1.0f, float max = 1.0f)
{
	curandGenerate(generator, (uint32_t*)output, (size >> 1) + (size & 1));
}

__global__ void CurandNormalizeF16(__half* output, uint32_t size, float min, float range)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
		output[index] = __float2half(*(uint16_t*)(output + index) * 0.0000152590218967f * range + min);
}