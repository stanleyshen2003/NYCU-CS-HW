#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define N 16


#include <cuda_runtime.h>

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int* d_img, int resX, int maxIterations) {
    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;


    
    float2 cXY = make_float2(lowerX + threadX * stepX, lowerY + threadY * stepY);
    float2 z = cXY;
    float2 temp;
    float2 zXY2;

    int i;
    for (i = 0; i < maxIterations; ++i) {
        zXY2.x = z.x * z.x;
        zXY2.y = z.y * z.y;
        if (zXY2.x + zXY2.y > 4.0f) break;
        temp.x = zXY2.x - zXY2.y + cXY.x;
        temp.y = 2.0f * z.x * z.y + cXY.y;
        z = temp;
    }

    d_img[threadY * resX + threadX] = i;
}


// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    int size = resX * resY * sizeof(int);

    int* ans;

    cudaHostRegister(img, size, cudaHostRegisterDefault);
    cudaHostGetDevicePointer(&ans, img, 0);

    // thread per block and block num
    dim3 threadsPerBlock(N, N);
    dim3 numBlocks(resX / threadsPerBlock.x, resY / threadsPerBlock.y);

    // launch kernel
    mandelKernel<<<numBlocks, threadsPerBlock>>>(lowerX, lowerY, stepX, stepY, ans, resX, maxIterations);

    cudaDeviceSynchronize();
    // cudaMemcpy(img, ans, size, cudaMemcpyDeviceToHost);
    cudaFree(ans);
    cudaHostUnregister(img);
}