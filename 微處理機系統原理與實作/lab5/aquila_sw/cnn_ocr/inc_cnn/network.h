#pragma once
// =============================================================================
//  Program : activation_function.h
//  Author  : Chang-Jyun Liao
//  Date    : July/12/2024
// -----------------------------------------------------------------------------
//  Description:
//      This file defines the abstract network storing layers.
// -----------------------------------------------------------------------------
//  Revision information:
//
//  Dec/5/2024, by Chang-Jyun Liao:
//      Add a function for network freeing.
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

#include "list.h"
#include "layer.h"
#include "config.h"
#include "util.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef PRINT_LAYER
#define PRINT_LOG(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define PRINT_LOG(fmt, ...)
#endif

typedef struct _network
{
    struct list_node layers;
} network;

static inline void init_network(network *net)
{
    net->layers.next = &net->layers;
    net->layers.prev = &net->layers;
}

static void free_network(network *net)
{
    struct list_node * cur;
    struct list_node * head = &(net->layers);
    struct list_node * nxt = head->next;
    for (cur = nxt; cur != head; cur = nxt)
    {
        nxt = cur->next;
        layer_base *ptr = list_entry(cur, layer_base, list);
        PRINT_LOG("-------------------------------------\n");
        PRINT_LOG("[%s]\n", ptr->layer_name_);
        if (ptr->padded)
        {
            PRINT_LOG("Freeing padded_ptr\n");
//            free(ptr->padded_ptr);
        }
        PRINT_LOG("Freeing a_ptr_\n");
//        free(ptr->a_ptr_);
#if USE_MATH_LIB // Comment these functions out to avoid using <math.h>
        if (ptr->activate == softmax)
        {
            PRINT_LOG("Freeing out_ptr_\n");
            free(ptr->out_ptr_);
        }
#endif
            PRINT_LOG("Freeing base\n");
        free(ptr);
    }
    PRINT_LOG("-------------------------------------\n");
    PRINT_LOG("freeing network\n");
    free(net);
}

void predict(network *net, void *input_base, uint64_t size)
{
    struct list_node * pos;
    input_struct prev_output = { (float_t *)input_base, size };

    list_for_each(pos, &net->layers)
    {
        layer_base *tmp = list_entry(pos, layer_base, list);
        tmp->forward_propagation(pos, &prev_output);
        // prev_output.in_ptr_ = tmp->out_ptr_;
        // prev_output.in_size_ = tmp->out_size_;
    }
}

