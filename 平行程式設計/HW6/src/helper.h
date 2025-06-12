#ifndef __HELPER__
#define __HELPER__

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define CHECK(status, cmd)                           \
    {                                                \
        if (status != CL_SUCCESS)                    \
        {                                            \
            printf("%s failed (%d)\n", cmd, status); \
            exit(-1);                                \
        }                                            \
    }

// This function reads in a text file and stores it as a char pointer
char *readSource(char *kernelPath);

void initCL(cl_device_id *device, cl_context *context, cl_program *program);

float *readFilter(const char *filename, int *filterWidth);
#endif