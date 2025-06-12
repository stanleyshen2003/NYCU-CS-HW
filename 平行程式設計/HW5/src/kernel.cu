#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define N 16

// mandel function in serial.cpp
__device__ int mandel(float c_re, float c_im, int maxIteration) {
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < maxIteration; ++i) {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    return i;
}



__global__ void mandelKernel (float lowerX, float lowerY, float stepX, float stepY, int* d_img, int resX, int resY, int maxIterations){
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    if (thisX < resX && thisY < resY) {
        float x = lowerX + thisX * stepX;
        float y = lowerY + thisY * stepY;
        int index = thisY * resX + thisX;
        d_img[index] = mandel(x, y, maxIterations);
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // Allocate device memory
    int* d_img;
    size_t imgSize = resX * resY * sizeof(int);
    cudaMalloc((void**)&d_img, imgSize);

    // Define thread block size and grid size
    dim3 threadsPerBlock(N, N);
    dim3 numBlocks((resX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (resY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    mandelKernel<<<numBlocks, threadsPerBlock>>>(lowerX, lowerY, stepX, stepY, d_img, resX, resY, maxIterations);

    // Copy the result back to host memory
    cudaMemcpy(img, d_img, imgSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_img);
}