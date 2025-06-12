#include "kernel.h"

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by using CUDA
void mandelbrotThread(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    hostFE(x1, y1, x0, y0, output, width, height, maxIterations);
}
