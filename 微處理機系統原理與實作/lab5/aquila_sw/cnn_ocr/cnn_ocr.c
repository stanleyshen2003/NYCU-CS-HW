// =============================================================================
//  Program : cnn_ocr.c
//  Author  : Chun-Jen Tsai
//  Date    : Dec/03/2024
// -----------------------------------------------------------------------------
//  Description:
//      This program implements a 5-layer CNN that has 99% accuracy
//  on the MNIST OCR dataset. The CNN library used in this program is
//  developped by Chang-Chun Liao.
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
#include <time.h>
#include "list.h"
#include "config.h"
#include "network.h"
#include "layer.h"
#include "activation_function.h"
#include "dummy_head_layer.h"
#include "convolutional_layer.h"
#include "average_pooling_layer.h"
#include "fully_connected_layer.h"
#include "file_read.h"

int cnn_evaluate(network *nn, float_t *image, int image_size)
{
    layer_base *output_layer;
    float_t* cnn_outputs;
	float_t max;
	int idx, max_idx;

    // Perform a single-image OOCR inference.
    predict(nn, image, image_size);
    output_layer = list_last_entry(&nn->layers, layer_base, list);
    cnn_outputs = output_layer->out_ptr_;

	// Compute the index to the maximal neuron value of the output layer.
	max = -1.0, max_idx = 0;
	for (idx = 0; idx < 10; idx++)
	{
	    if (max < cnn_outputs[idx])
	    {
	        max_idx = idx;
	        max = cnn_outputs[idx];
	    }
	}

	return max_idx;
}

int main()
{
    network *nn;
    cnn_controller ctrl;
    int n_images, n_rows, n_cols;
    int image_size, class_id;
    unsigned correct_count = 0;
    clock_t  tick, ticks_per_msec = CLOCKS_PER_SEC/1000;

    // Read the test images & ground-truth labels.
    printf("\n(1) Reading the test images, labels, and neural weights.\n");
    tick = clock();
    float_t *weights = read_weights("weights.dat");
    float_t **images = read_images("test-images.dat", &n_images, &n_rows, &n_cols, 0);
    uint8_t *labels = read_labels("test-labels.dat");
    image_size = n_rows*n_cols;
    tick = (clock() - tick)/ticks_per_msec;
    printf("It took %ld msec to read files from the SD card.\n\n", tick);

    // Initialize the neural network model.
    nn = (network *) malloc(sizeof(network));
    memset((void *) &ctrl, 0, sizeof(cnn_controller));
    ctrl.wgt_cur_ptr = weights;

    init_network(nn);
    push_back(&new_dummy_head_layer(&ctrl, identity, image_size)->list, &nn->layers);
    push_back(&new_convolutional_layer(&ctrl, relu, 28, 28, 5, 5, 1, 3, same, 1, 1, 1, 0, 0)->list, &nn->layers);
    push_back(&new_average_pooling_layer(&ctrl, identity, 24, 24, 3, 2, 2)->list, &nn->layers);
    push_back(&new_convolutional_layer(&ctrl, relu, 12, 12, 5, 5, 3, 32, same, 1, 1, 1, 0, 0)->list, &nn->layers);
    push_back(&new_average_pooling_layer(&ctrl, identity, 8, 8, 32, 2, 2)->list, &nn->layers);
    push_back(&new_fully_connected_layer(&ctrl, relu, 4*4*32, 10, 0)->list, &nn->layers);

    // Perform hand-written digits recognition tests.
    printf("(2) Perform the hand-written digits recognition test.\n");
    printf("Here, we use a 5-layer CNN neural network model.\n");
    printf("Begin computing ... ");
    tick = clock();
    for (int idx = 0; idx < n_images; idx++)
    {
        class_id = cnn_evaluate(nn, images[idx], image_size);
        if ((int) labels[idx] == class_id) ++correct_count;
    }
    tick = (clock() - tick)/ticks_per_msec;
    printf("tested %d images. The accuracy is %.2f%%\n\n",
        n_images, 100.0f * (float) correct_count / n_images);
    printf("It took %ld msec to perform the test.\n\n", tick);

    // Free all allocated memory blocks.
    free_network(nn);
    for (int idx = 0; idx < n_images; idx++) free(images[idx]);
    free(images);
    free(labels);
    free(weights);

    return 0;
}

