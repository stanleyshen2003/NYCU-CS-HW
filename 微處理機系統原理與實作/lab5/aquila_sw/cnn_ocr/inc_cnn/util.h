#pragma once
// =============================================================================
//  Program : activation_function.h
//  Author  : Chang-Jyun Liao
//  Date    : July/12/2024
// -----------------------------------------------------------------------------
//  Description:
//      This file defines some useful functions for our library.
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

#include <limits.h>

double max(double a, double b) { return (a > b) ? a : b; }
double min(double a, double b) { return (a < b) ? a : b; }

typedef struct _index3d
{
    uint64_t height_;
    uint64_t width_;
    uint64_t depth_;
} index3d;

uint64_t get_index(index3d *index, uint64_t x, uint64_t y, uint64_t channel)
{
    return (index->height_ * channel + y) * index->width_ + x;
}

index3d new_index3d(uint64_t x, uint64_t y, uint64_t c)
{
    index3d ret = {y, x, c};
    return ret;
}

typedef struct _cnn_controller
{
    void *nwk_cur_ptr;
    void *lyr_cur_ptr;
    void *wgt_cur_ptr;
    uint64_t padding_size;
} cnn_controller;

typedef struct _output_index_name
{
    uint64_t index;
    float value;
    char name[40];
} output_index_name;

double no_math_ceil(double x)
{
    if (x >= LLONG_MAX && x < LLONG_MIN)
    {
        intmax_t i = (intmax_t) x;      // this rounds towards 0
        if (i < 0 || x == i) return i;  // negative x is already rounded up.
        return i + 1.0;
    }
    return x;
}

