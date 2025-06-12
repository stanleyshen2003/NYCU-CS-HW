#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define N 16

__global__ void mandelKernel (float lowerX, float lowerY, float stepX, float stepY, int* d_img, int resX, int resY, int maxIterations){
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
    d_img[threadY * resX + threadX] = i;
    
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    int size = resX * resY * sizeof(int);

    int *host_image = (int*)malloc(size);
    int* ans;
    cudaMalloc((void**)&ans, size);

    // thread per block and block num
    dim3 threadsPerBlock(N, N);
    dim3 numBlocks(resX / threadsPerBlock.x, resY / threadsPerBlock.y);

    // launch kernel
    mandelKernel<<<numBlocks, threadsPerBlock>>>(lowerX, lowerY, stepX, stepY, ans, resX, resY, maxIterations);

    cudaMemcpy(host_image, ans, size, cudaMemcpyDeviceToHost);
    memcpy(img, host_image, size);
    free(host_image);
    cudaFree(ans);
}