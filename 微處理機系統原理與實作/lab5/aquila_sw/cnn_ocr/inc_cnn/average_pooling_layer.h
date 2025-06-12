#pragma once
// =============================================================================
//  Program : activation_function.h
//  Author  : Chang-Jyun Liao
//  Date    : July/14/2024
// -----------------------------------------------------------------------------
//  Description:
//      This file defines the average pooling layer for the CNN models.
// -----------------------------------------------------------------------------
//  Revision information:
//
//  None.
// -----------------------------------------------------------------------------
//  License information:
//
//  This software is released under the BSD-3-Clause Licence,
//  see https://opensource.org/licenses/BSD-3-Clause for details.
//  In the following license statements, "software" refers to the
//  "source code" of the complete hardware/software system.
//
//  Copyright 2024,
//                    Embedded Intelligent Systems Lab (EISL)
//                    Deparment of Computer Science
//                    National Yang Ming Chiao Tung University
//                    Hsinchu, Taiwan.
//
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//  3. Neither the name of the copyright holder nor the names of its contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
// =============================================================================

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "config.h"
#include "layer.h"
#include "list.h"
#include "util.h"
#include "activation_function.h"

typedef struct _average_pooling_layer
{
    layer_base base;
    uint64_t stride_;
    float_t scale_factor_;
    uint64_t pooling_size_;
    index3d in_;
    index3d out_;
} average_pooling_layer;

average_pooling_layer * get_average_pooling_layer_entry(struct list_node *ptr)
{
    return list_entry(ptr, average_pooling_layer, base.list);
}

void average_pooling_layer_forward_propagation(struct list_node *ptr, input_struct *input)
{
    average_pooling_layer *entry = get_average_pooling_layer_entry(ptr);
    if (input->in_size_ != entry->base.in_size_)
    {
        printf("Error input size not match %lu/%lu\n", input->in_size_, entry->base.in_size_);
        exit(-1);
    }
    float_t *in = input->in_ptr_;
    float_t *a = entry->base.a_ptr_;
    float_t *out = entry->base.out_ptr_;
    input->in_ptr_ = out;
    input->in_size_ = entry->base.out_size_;
    uint64_t stride_ = entry->stride_;

    index3d in_ = entry->in_;
    index3d out_ = entry->out_;

    uint64_t total_size = entry->base.out_size_;
    
    uint64_t dim = out_.height_*out_.width_;
    for (uint64_t o = 0; o < total_size; o++)
    {
        uint64_t c = o / dim;
        a[o] = (float_t)0;
        uint64_t xy = o % dim;
        uint64_t dsty = xy / out_.width_;
        uint64_t dstx = xy % out_.width_;
        uint64_t y = dsty*stride_;
        uint64_t x = dstx*stride_;
        uint64_t dymax = min(entry->pooling_size_, in_.height_ - y);
        uint64_t dxmax = min(entry->pooling_size_, in_.width_ - x);

        for (uint64_t dy = 0; dy < dymax; dy++)
            for (uint64_t dx = 0; dx < dxmax; dx++)
            {
                a[o] += in[get_index(&in_, x + dx, y + dy, c)];
            }
        a[o] *= entry->scale_factor_;
    }

    for (uint64_t o = 0; o < total_size; o++)
        out[o] = entry->base.activate(a, o, entry->base.out_size_);
    
#ifdef PRINT_LAYER
    printf("[%s] done [%f, %f, ... , %f, %f]\n", entry->base.layer_name_, out[0], out[1], out[entry->base.out_size_-2], out[entry->base.out_size_-1]);
#endif
}

static uint64_t pool_out_dim(uint64_t in_size, uint64_t pooling_size, uint64_t stride)
{
    return (int)(((float_t)in_size - pooling_size) / stride) + 1;
}

layer_base * new_average_pooling_layer(
                                       cnn_controller *ctrl,
                                       float_t(*activate) (float_t *, uint64_t, uint64_t),
                                       uint64_t in_width,
                                       uint64_t in_height,
                                       uint64_t in_channels,
                                       uint64_t pooling_size,
                                       uint64_t stride
                                       )
{

    average_pooling_layer *ret = (average_pooling_layer *)malloc(sizeof(average_pooling_layer));
    ctrl->padding_size = 0;
    init_layer(&ret->base,
               ctrl,
               in_width*in_height*in_channels,
               pool_out_dim(in_width, pooling_size, stride) * pool_out_dim(in_height, pooling_size, stride) * in_channels, 
               0,
               0,
               activate==relu);
#ifdef PRINT_LAYER
    static uint64_t call_time = 0;
    sprintf(ret->base.layer_name_, "avg_pool%lu", call_time++);
#endif
    ret->scale_factor_ = (float_t)1 / (pooling_size*pooling_size);
    ret->stride_ = stride;
    ret->pooling_size_ = pooling_size;
    ret->in_ = new_index3d(in_width, in_height, in_channels);
    ret->out_ = new_index3d(pool_out_dim(in_width, pooling_size, stride), pool_out_dim(in_height, pooling_size, stride), in_channels);

    ret->base.activate = activate;
    ret->base.forward_propagation = average_pooling_layer_forward_propagation;
    // printf("insize of average pooling layer %d\n", ret->base.in_size_);
    // printf("avg pool: in [%f, %f, ... , %f, %f]\n", ret->base.in_ptr_[0], ret->base.in_ptr_[1], ret->base.in_ptr_[ret->base.in_size_-2], ret->base.in_ptr_[ret->base.in_size_-1]);
    // printf("avg pool: b  [%f, %f, ... , %f, %f]\n", ret->base._b[0], ret->base._b[1], ret->base._b[in_channels-2], ret->base._b[in_channels-1]);
    return &ret->base;
}

