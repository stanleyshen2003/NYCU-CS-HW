#ifndef __HOSTFE__
#define __HOSTFE__
#include <CL/cl.h>

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program);

#endif