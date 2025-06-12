#pragma once
// =============================================================================
//  Program : activation_function.h
//  Author  : Chang-Jyun Liao
//  Date    : July/14/2024
// -----------------------------------------------------------------------------
//  Description:
//      This file defines pseudo layers used when layer creation for the CNN models.
// -----------------------------------------------------------------------------
//  Revision information:
//
//  Dec/5/2024, by Chang-Jyun Liao:
//      Add a member variable in _layer_base for supporting network freeing.
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
#include "config.h"
#include "list.h"
#include "activation_function.h"

typedef struct _input_struct
{
    float_t *in_ptr_;
    uint64_t in_size_;
} input_struct;

typedef struct _layer_base
{
    uint64_t a_done_flag;
    uint64_t done_flag;
    float_t (*activate) (float_t *, uint64_t, uint64_t);
    void (*forward_propagation) (struct list_node *, input_struct*);
    struct list_node* (*get_entry) ();
    uint64_t in_size_;
    uint64_t out_size_;
    float_t *a_ptr_;
    float_t *padded_ptr;
    float_t *out_ptr_;
    uint8_t padded;

    float_t *_W;
    float_t *_b;
    struct list_node list;
#ifdef PRINT_LAYER
    char layer_name_[20];
#endif
} layer_base;


void init_layer(layer_base *layer, cnn_controller *ctrl, uint64_t in_dim, uint64_t out_dim, uint64_t weight_dim, uint64_t bias_dim, uint8_t need_space_for_a)
{
    layer->in_size_ = in_dim;
    layer->out_size_ = out_dim;
    layer->done_flag = 0;
    layer->a_done_flag = 0;

    if (ctrl->padding_size) {
        layer->padded_ptr = (float_t *)malloc(ctrl->padding_size * sizeof(float_t));
        layer->padded = 1;
    } else {
        layer->padded_ptr = (float_t *)ctrl->lyr_cur_ptr;
        layer->padded = 0;
    }
    layer->a_ptr_ = (float_t *)malloc(out_dim * sizeof(float_t));
    if (need_space_for_a) 
        layer->out_ptr_ = (float_t *)malloc(out_dim * sizeof(float_t));
    else 
        layer->out_ptr_ = layer->a_ptr_;
    
    ctrl->lyr_cur_ptr = (void *)layer->out_ptr_;

    layer->_W = (float_t *)ctrl->wgt_cur_ptr;
    ctrl->wgt_cur_ptr += weight_dim * sizeof(float_t);

    layer->_b = (float_t *)ctrl->wgt_cur_ptr;
    ctrl->wgt_cur_ptr += bias_dim * sizeof(float_t);
}

