#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"
#define GROUP_SIZE 8

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;

    // command queue
    cl_command_queue queue = clCreateCommandQueueWithProperties(*context, *device, 0, &status);

    // create buffers on device
    cl_mem d_inputImage = clCreateBuffer(*context, CL_MEM_READ_ONLY, imageHeight * imageWidth * sizeof(float), NULL, &status);
    cl_mem d_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize * sizeof(float), NULL, &status);
    cl_mem d_outputImage = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageHeight * imageWidth * sizeof(float), NULL, &status);

    // transfer input data to device
    clEnqueueWriteBuffer(queue, d_inputImage, CL_TRUE, 0, imageHeight * imageWidth * sizeof(float), inputImage, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_filter, CL_TRUE, 0, filterSize * sizeof(float), filter, 0, NULL, NULL);

    // compile kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);

    // set arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_inputImage);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_filter);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_outputImage);
    clSetKernelArg(kernel, 3, sizeof(int), &filterWidth);

    size_t globalWorkSize[2] = {imageWidth, imageHeight};
    size_t localWorkSize[2] = {GROUP_SIZE, GROUP_SIZE};

    // execute kernel
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    
    // create buffer
    clEnqueueReadBuffer(queue, d_outputImage, CL_TRUE, 0, imageHeight * imageWidth * sizeof(float), outputImage, 0, NULL, NULL);

    // clean up
    clReleaseKernel(kernel);
    clReleaseMemObject(d_inputImage);
    clReleaseMemObject(d_filter);
    clReleaseMemObject(d_outputImage);
    clReleaseCommandQueue(queue);
}