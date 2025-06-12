#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>
#include "CycleTimer.h"
#include "helper.h"
#include "hostFE.h"
#include "bmpfuncs.h"
#include "serialConv.h"

void usage(const char *progname)
{
   printf("Usage: %s [options]\n", progname);
   printf("Program Options:\n");
   printf("  -i  --input   <String> Input image\n");
   printf("  -f  --filter  <INT>    Use which filter (1, 2, 3)\n");
   printf("  -?  --help             This message\n");
}

int compare(const void *a, const void *b)
{
   double *x = (double *)a;
   double *y = (double *)b;
   if (*x < *y)
      return -1;
   else if (*x > *y)
      return 1;
   return 0;
}

int main(int argc, char **argv)
{
   int i, j;

   // Rows and columns in the input image
   int imageHeight;
   int imageWidth;

   double start_time, end_time;

   char *inputFile = "input.bmp";
   const char *outputFile = "output.bmp";
   const char *refFile = "ref.bmp";
   char *filterFile = "filter1.csv";

   // parse commandline options
   int opt;
   static struct option long_options[] = {
       {"filter", 1, 0, 'f'},
       {"input", 1, 0, 'i'},
       {"help", 0, 0, '?'},
       {0, 0, 0, 0}};

   while ((opt = getopt_long(argc, argv, "i:f:?", long_options, NULL)) != EOF)
   {

      switch (opt)
      {
      case 'i':
      {
         inputFile = optarg;

         break;
      }
      case 'f':
      {
         int idx = atoi(optarg);
         if (idx == 2)
            filterFile = "filter2.csv";
         else if (idx == 3)
            filterFile = "filter3.csv";

         break;
      }
      case '?':
      default:
         usage(argv[0]);
         return 1;
      }
   }
   // end parsing of commandline options

   // read filter data
   int filterWidth;
   float *filter = readFilter(filterFile, &filterWidth);

   // Homegrown function to read a BMP from file
   float *inputImage = readImage(inputFile, &imageWidth, &imageHeight);
   // Size of the input and output images on the host
   int dataSize = imageHeight * imageWidth * sizeof(float);
   // Output image on the host
   float *outputImage = (float *)malloc(dataSize);

   // helper init CL
   cl_program program;
   cl_device_id device;
   cl_context context;
   initCL(&device, &context, &program);

   double minThread = 0;
   double recordThread[10] = {0};
   for (int i = 0; i < 10; ++i)
   {
      memset(outputImage, 0, dataSize);
      start_time = currentSeconds();
      // Run the host to execute the kernel
      hostFE(filterWidth, filter, imageHeight, imageWidth, inputImage, outputImage,
             &device, &context, &program);
      end_time = currentSeconds();
      recordThread[i] = end_time - start_time;
   }
   qsort(recordThread, 10, sizeof(double), compare);
   for (int i = 3; i < 7; ++i)
   {
      minThread += recordThread[i];
   }
   minThread /= 4;

   printf("\n[conv opencl]:\t\t[%.3f] ms\n\n", minThread * 1000);

   // Write the output image to file
   storeImage(outputImage, outputFile, imageHeight, imageWidth, inputFile);

   // Output image of reference on the host
   float *refImage = NULL;
   refImage = (float *)malloc(dataSize);
   memset(refImage, 0, dataSize);

   double minSerial = 0;
   double recordSerial[10] = {0};
   for (int i = 0; i < 10; ++i)
   {
      memset(refImage, 0, dataSize);
      start_time = currentSeconds();
      serialConv(filterWidth, filter, imageHeight, imageWidth, inputImage, refImage);
      end_time = currentSeconds();
      recordSerial[i] = end_time - start_time;
   }
   qsort(recordSerial, 10, sizeof(double), compare);
   for (int i = 3; i < 7; ++i)
   {
      minSerial += recordSerial[i];
   }
   minSerial /= 4;

   printf("\n[conv serial]:\t\t[%.3f] ms\n\n", minSerial * 1000);

   storeImage(refImage, refFile, imageHeight, imageWidth, inputFile);

   int diff_counter = 0;
   for (i = 0; i < imageHeight; i++)
   {
      for (j = 0; j < imageWidth; j++)
      {
         if (abs(outputImage[i * imageWidth + j] - refImage[i * imageWidth + j]) > 10)
         {
            diff_counter += 1;
         }
      }
   }

   float diff_ratio = (float)diff_counter / (imageHeight * imageWidth);
   printf("Diff ratio: %f\n", diff_ratio);

   if (diff_ratio > 0.1)
   {
      printf("\n\033[31mFAILED:\tResults are incorrect!\033[0m\n");
      return -1;
   }
   else
   {
      printf("\n\033[32mPASS:\t(%.2fx speedup over the serial version)\033[0m\n", minSerial / minThread);
   }

   return 0;
}
