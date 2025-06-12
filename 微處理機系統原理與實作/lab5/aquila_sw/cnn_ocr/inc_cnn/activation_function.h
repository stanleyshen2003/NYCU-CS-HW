#pragma once
// =============================================================================
//  Program : activation_function.h
//  Author  : Chang-Jyun Liao
//  Date    : July/14/2024
// -----------------------------------------------------------------------------
//  Description:
//      This file defines the available activation functions for the CNN models.
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

#include "util.h"
#include "config.h"
#include <stdint.h>
#include <stdlib.h>

static inline float_t identity(float_t *arr, uint64_t index, uint64_t out_dim)
{
    return arr[index];
}

static inline float_t relu(float_t *arr, uint64_t index, uint64_t out_dim)
{
    return max((float_t)0, arr[index]);
}

static inline float_t bounded_relu(float_t *arr, uint64_t index, uint64_t out_dim)
{
    return max((float_t)0, min((float_t)1, arr[index]));
}

#if USE_MATH_LIB // Comment these functions out to avoid using <math.h>
#include <math.h>

static inline float_t softmax(float_t *arr, uint64_t index, uint64_t out_dim)
{
    float_t max = 0;
    for (uint64_t i = 0; i < out_dim; i++)
        if (arr[i] > max)
            max = arr[i];
    float_t numer = exp(arr[index] - max);
    float_t denom = 0;
    for (uint64_t i = 0; i < out_dim; i++)
        denom += exp(arr[i] - max);
    return numer / denom;
}
#endif

