#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define N 16

__global__ void mandelKernel (float lowerX, float lowerY, float stepX, float stepY, int* d_img, int resX, size_t pitch, int maxIterations){
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;

    float x = lowerX + threadX * stepX;
    float y = lowerY + threadY * stepY;

    float z_re = x, z_im = y;
    int i;
    for (i = 0; i < maxIterations; ++i) {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
    }
    *((int*)((char*)d_img + threadY * pitch) + threadX) = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int* host_image, *pinnedImg;
    int size = resX * resY * sizeof(int);
    size_t pitch;
    cudaHostAlloc((void**)&host_image, size, cudaHostAllocDefault);
    cudaMallocPitch((void**)&pinnedImg, &pitch, resX * sizeof(int), resY);


    // thread per block and block num
    dim3 threadsPerBlock(N, N);
    dim3 numBlocks(resX / threadsPerBlock.x, resY / threadsPerBlock.y);

    // launch kernel
    mandelKernel<<<numBlocks, threadsPerBlock>>>(lowerX, lowerY, stepX, stepY, pinnedImg, resX, pitch, maxIterations);

    cudaMemcpy2D(img, resX * sizeof(int), pinnedImg, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
    // memcpy(img, host_image, size);
    cudaFree(pinnedImg);
    cudaFreeHost(host_image);
}