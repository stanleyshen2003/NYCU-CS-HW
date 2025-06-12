#include <stdio.h>
#include <algorithm>
#include <getopt.h>

#include "CycleTimer.h"

extern void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int numRows,
    int maxIterations,
    int output[]);

extern void mandelbrotThread(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations,
    int output[]);

extern void mandelbrotThreadRef(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations,
    int output[]);

extern void writePPMImage(
    int *data,
    int width, int height,
    const char *filename,
    int maxIterations);

void scaleAndShift(float &x0, float &x1, float &y0, float &y1,
                   float scale,
                   float shiftX, float shiftY)
{

    x0 *= scale;
    x1 *= scale;
    y0 *= scale;
    y1 *= scale;
    x0 += shiftX;
    x1 += shiftX;
    y0 += shiftY;
    y1 += shiftY;
}

void usage(const char *progname)
{
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -i  --iter <INT>       Use specified interation (>=256) (Default = 256)\n");
    printf("  -v  --view <INT>       Use specified view settings (1 or 2) (Default = 1)\n");
    printf("  -g  --gpu-only <INT>   Only run GPU or not (1 or 0) (Default = 0)\n");
    printf("  -?  --help             This message\n");
}

bool verifyResult(int *gold, int *result, int width, int height)
{

    int i, j;

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            if (abs(gold[i * width + j] - result[i * width + j]) > 0)
            {
                printf("Mismatch : [%d][%d], Expected : %d, Actual : %d\n",
                       i, j, gold[i * width + j], result[i * width + j]);
                return 0;
            }
        }
    }

    return 1;
}

int main(int argc, char **argv)
{

    const unsigned int width = 1600;
    const unsigned int height = 1200;
    int maxIterations = 256;
    bool isGPUOnly = false;

    float x0 = -2;
    float x1 = 1;
    float y0 = -1;
    float y1 = 1;

    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"iter", 1, 0, 'i'},
        {"view", 1, 0, 'v'},
        {"gpu-only", 1, 0, 'g'},
        {"help", 0, 0, '?'},
        {0, 0, 0, 0}};

    while ((opt = getopt_long(argc, argv, "i:v:g:?", long_options, NULL)) != EOF)
    {

        switch (opt)
        {
        case 'i':
        {
            int iter = atoi(optarg);
            if (iter < 256)
            {
                fprintf(stderr, "Iteration should >= 256\n");
                return 1;
            }
            maxIterations = iter;
            break;
        }
        case 'v':
        {
            int viewIndex = atoi(optarg);
            // change view settings
            if (viewIndex == 2)
            {
                float scaleValue = .015f;
                float shiftX = -.986f;
                float shiftY = .30f;
                scaleAndShift(x0, x1, y0, y1, scaleValue, shiftX, shiftY);
            }
            else if (viewIndex > 1)
            {
                fprintf(stderr, "Invalid view index\n");
                return 1;
            }
            break;
        }
        case 'g':
        {
            int flag = atoi(optarg);
            // change GPU settings
            if (flag == 1 || flag == 0)
            {
                isGPUOnly = flag;
            }
            else
            {
                fprintf(stderr, "Invalid setting. Only allow 0 or 1.\n");
                return 1;
            }
            break;
        }
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options

    int *output_test = new int[width * height];
    int *output_thread = new int[width * height];

    //
    // Run the serial implementation.  Run the code three times and
    // take the minimum to get a good estimate.
    //
    double minSerial = 1e30;
    if (!isGPUOnly)
    {
        for (int i = 0; i < 5; ++i)
        {
            memset(output_test, 0, width * height * sizeof(int));
            double startTime = CycleTimer::currentSeconds();
            mandelbrotSerial(x0, y0, x1, y1, width, height, 0, height, maxIterations, output_test);
            double endTime = CycleTimer::currentSeconds();
            minSerial = std::min(minSerial, endTime - startTime);
        }

        printf("[mandelbrot serial]:\t\t[%.3f] ms\n", minSerial * 1000);
    }

    //
    // Run the reference threaded version
    //

    double minRef = 0;
    double recordRef[10] = {0};
    for (int i = 0; i < 10; ++i)
    {
        memset(output_thread, 0, width * height * sizeof(int));
        double startTime = CycleTimer::currentSeconds();
        mandelbrotThreadRef(x0, y0, x1, y1, width, height, maxIterations, output_test);
        double endTime = CycleTimer::currentSeconds();
        recordRef[i] = endTime - startTime;
    }
    std::sort(recordRef, recordRef + 10);
    for (int i = 3; i < 7; ++i)
    {
        minRef += recordRef[i];
    }
    minRef /= 4;

    printf("[mandelbrot reference]:\t\t[%.3f] ms\n", minRef * 1000);
    writePPMImage(output_test, width, height, "mandelbrot-ref.ppm", maxIterations);

    //
    // Run the threaded version
    //

    double minThread = 0;
    double recordThread[10] = {0};
    for (int i = 0; i < 10; ++i)
    {
        memset(output_thread, 0, width * height * sizeof(int));
        double startTime = CycleTimer::currentSeconds();
        mandelbrotThread(x0, y0, x1, y1, width, height, maxIterations, output_thread);
        double endTime = CycleTimer::currentSeconds();
        recordThread[i] = endTime - startTime;
    }
    std::sort(recordThread, recordThread + 10);
    for (int i = 3; i < 7; ++i)
    {
        minThread += recordThread[i];
    }
    minThread /= 4;

    printf("[mandelbrot thread]:\t\t[%.3f] ms\n", minThread * 1000);
    writePPMImage(output_thread, width, height, "mandelbrot-thread.ppm", maxIterations);

    if (!verifyResult(output_test, output_thread, width, height))
    {
        printf("Error : Output from threads does not match test output\n");

        delete[] output_test;
        delete[] output_thread;

        return 1;
    }

    // compute speedup
    if (!isGPUOnly)
        printf("\t\t\t\t(%.2fx speedup over the CPU serial version)\n", minSerial / minThread);
    printf("\t\t\t\t(%.2fx speedup over the reference)\n", minRef / minThread);

    delete[] output_test;
    delete[] output_thread;

    return 0;
}
