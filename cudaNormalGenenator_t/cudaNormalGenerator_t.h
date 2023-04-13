#pragma once
#include "Header.cuh"

struct cudaNormalGenerator_t
{
	curandGenerator_t curandGenerator;
	int32_t kn[128];
	float fn[128];
	float wn[128];
};

void cudaNormalGeneratorCreate(cudaNormalGenerator_t* generator)
{
	curandCreateGenerator(&generator->curandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
    
    double dn = 3.442619855899;
    const double m1 = 2147483648.0;
    const double vn = 9.91256303526217E-03;
    double q = vn / exp(-0.5 * dn * dn);

    kn[0] = dn / q * m1;
    kn[1] = 0;
    wn[0] = q / m1;
    wn[127] = dn / m1;
    fn[0] = 1.0;
    fn[127] = exp(-0.5 * dn * dn);

    double tn;
    for (uint8_t i = 126; 1 <= i; i--)
    {
        tn = dn;
        dn = sqrt(-2.0 * log(vn / dn + exp(-0.5 * dn * dn)));
        kn[i + 1] = dn / tn * m1;
        fn[i] = exp(-0.5 * dn * dn);
        wn[i] = dn / m1;
    }
}

void cudaNormalGeneratorSetSeed(cudaNormalGenerator_t* generator, uint64_t seed)
{
	curandSetPseudoRandomGeneratorSeed(generator->curandGenerator, seed);
}

void CurandGenerateNormalF16(cudaNormalGenerator_t* generator, __half* output, uint32_t size, float mean = 0.0f, float standardDeviation = 1.0f)
{
	curandGenerate(generator, (uint32_t*)output, (size >> 1) + (size & 1));
	CudaZigguratF16 << <std::ceil(0.0009765625f * size), 1024 >> > (generator, output, size, mean, standardDeviation);
}

__global__ void CudaZigguratF16(cudaNormalGenerator_t* generator, __half* output, uint32_t size, float mean, float standardDeviation)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
    {
        uint32_t uint32Temp;
        int32_t int32Seed;
        uint8_t int8Seed;
        float x, y;

        seed = *(uint16_t*)(output + index);
        seed ^= seed << 16;
        
        uint32Temp = seed;
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        int32Seed = uint32Temp + seed;
        int8Seed = int32Seed & 127;
        if ((int32Seed ^ (int32Seed >> 31)) - (int32Seed >> 31) < generator->kn[int8Seed])
        {
            output[index] = generator->wn[int8Seed] * int32Seed;
            return;
        }

        for (;;)
        {
            if (int8Seed == 0)
            {
                for (;;)
                {
                    uint32Temp = seed;
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    x = (uint32Temp + seed) * 2.3283064365386963e-10f;
                    x = (1065353216 - *(int32_t*)&x) * 2.30830217163e-08f;

                    uint32Temp = seed;
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    y = (uint32Temp + seed) * 2.3283064365386963e-10f;
                    y = (1065353216 - *(int32_t*)&y) * 7.94660834913e-08f;

                    if (x * x <= y + y)
                    {
                        x += 3.442620f;
                        uint32Temp = int32Seed & 0x80000000 ^ *(uint32_t*)&x;
                        output[index] = *(float*)&uint32Temp;
                        return;
                    }
                }
            }

            uint32Temp = seed;
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            y = (uint32Temp + seed) * 2.3283064365386963e-10f;
            x = generator->wn[int8Seed] * int32Seed;
            uint32Temp = -6169045.423972f * x * x + 1065101626.864132f;
            if (y * (generator->fn[int8Seed - 1] - generator->fn[int8Seed]) + generator->fn[int8Seed] < *(float*)&uint32Temp)
                return x;

            uint32Temp = seed;
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            int32Seed = uint32Temp + seed;
            int8Seed = int32Seed & 127;
            if ((int32Seed ^ (int32Seed >> 31)) - (int32Seed >> 31) < generator->wn[int8Seed])
                return generator->wn[int8Seed] * int32Seed;
        }
    }
}