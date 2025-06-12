// =============================================================================
//  Program : file_read.c
//  Author  : Chun-Jen Tsai
//  Date    : Dec/06/2023
// -----------------------------------------------------------------------------
//  Description:
//      This is a library of file reading functions for MNIST test
//  images & labels. It also contains a function for reading the model
//  weights file of a neural network.
//
//  This program is designed as one of the homework projects for the course:
//  Microprocessor Systems: Principles and Implementation
//  Dept. of CS, NYCU (aka NCTU), Hsinchu, Taiwan.
// -----------------------------------------------------------------------------
//  Revision information:
//
//  None.
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "fat32.h"
#include "file_read.h"
#include "config.h"

// Our FAT32 file I/O routine need a large buffer area to read in
// the entire file before processing. the Arty board has 256MB DRAM.
uint8_t *fbuf  = (uint8_t *) 0x81000000L;

float_t **read_images(char *filename, int *n_images, int *n_rows, int *n_cols, int padding)
{
    uint8_t *iptr;
    float_t **images;
    int idx, jdx, size, row, col;

    read_file(filename, fbuf);
    iptr = fbuf;
    iptr += sizeof(int); // skip the ID of the file.

    *n_images = big2little32(iptr);
    iptr += sizeof(int);
    // printf("#images = %d\n", *n_images);

    *n_rows = big2little32(iptr) + padding*2;
    iptr += sizeof(int);
    // printf("#rows = %d\n", *n_rows);

    *n_cols = big2little32(iptr) + padding*2;
    iptr += sizeof(int);
    // printf("#cols = %d\n", *n_cols);
    size = (*n_rows) * (*n_cols);

    images = (float_t **) malloc(sizeof(float_t *) * *n_images);
    for (idx = 0; idx < *n_images; idx++)
    {
        images[idx] = (float_t *) calloc(size, sizeof(float_t));

        /* Convert the image pixels to PyTorch's input tensor format */
        for (row = padding; row < *n_rows-padding; row++)
        {
		    for (col = padding; col < *n_cols-padding; col++)
		    {
	            images[idx][row * (*n_cols) + col] = (float_t) *(iptr++)/255.0;
	        }
        }

        /* Normalize the pixels by PyTorch's transforms.Normalize(mean, std) rule */
        for (jdx = 0; jdx < size; jdx++)
        {
            images[idx][jdx] = (images[idx][jdx] - 0.1307) / 0.3081;
        }
    }

    return images;
}

uint8_t *read_labels(char *filename)
{
    uint8_t *labels;
    int n_labels;

    n_labels = read_file(filename, fbuf) - 8;
    if ((labels = (uint8_t *) malloc(n_labels)) == NULL)
    {
        printf("read_labels: out of memory.\n");
        exit (-1);        
    }
    memcpy((void *) labels, (void *) (fbuf+8), n_labels);
    
    return labels;
}

float_t *read_weights(char *filename)
{
    int size;
    float_t *weights;

    size = read_file(filename, fbuf);
    if ((weights = (float_t *) malloc(size)) == NULL)
    {
        printf("read_weights(): Out of memory.\n");
        exit (1);
    }
    memcpy((void *) weights, (void *) fbuf, size);
    // for (int i = 0; i < size/sizeof(float_t); i++)
    // {
    //     printf("%3d: %f\n", i, weights[i]);
    // }
    return (float_t *) weights;
}

