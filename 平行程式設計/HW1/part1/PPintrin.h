#ifndef PPINTRIN_H_
#define PPINTRIN_H_

#include <cstdlib>
#include <cmath>
#include "logger.h"
#include "def.h"
//*******************
//* Type Definition *
//*******************

extern Logger PPLogger;

template <typename T>
struct __pp_vec {
  T value[VECTOR_WIDTH];
};

// Declare a mask with __pp_mask
struct __pp_mask : __pp_vec<bool> {};

// Declare a floating point vector register with __pp_vec_float
#define __pp_vec_float __pp_vec<float>

// Declare an integer vector register with __pp_vec_int
#define __pp_vec_int   __pp_vec<int>

//***********************
//* Function Definition *
//***********************

// Return a mask initialized to 1 in the first N lanes and 0 in the others
__pp_mask _pp_init_ones(int first = VECTOR_WIDTH);

// Return the inverse of maska
__pp_mask _pp_mask_not(__pp_mask &maska);

// Return (maska | maskb)
__pp_mask _pp_mask_or(__pp_mask &maska, __pp_mask &maskb);

// Return (maska & maskb)
__pp_mask _pp_mask_and(__pp_mask &maska, __pp_mask &maskb);

// Count the number of 1s in maska
int _pp_cntbits(__pp_mask &maska);

// Set register to value if vector lane is active
//  otherwise keep the old value
void _pp_vset_float(__pp_vec_float &vecResult, float value, __pp_mask &mask);
void _pp_vset_int(__pp_vec_int &vecResult, int value, __pp_mask &mask);
// For user's convenience, returns a vector register with all lanes initialized to value
__pp_vec_float _pp_vset_float(float value);
__pp_vec_int _pp_vset_int(int value);

// Copy values from vector register src to vector register dest if vector lane active
// otherwise keep the old value
void _pp_vmove_float(__pp_vec_float &dest, __pp_vec_float &src, __pp_mask &mask);
void _pp_vmove_int(__pp_vec_int &dest, __pp_vec_int &src, __pp_mask &mask);

// Load values from array src to vector register dest if vector lane active
//  otherwise keep the old value
void _pp_vload_float(__pp_vec_float &dest, float* src, __pp_mask &mask);
void _pp_vload_int(__pp_vec_int &dest, int* src, __pp_mask &mask);

// Store values from vector register src to array dest if vector lane active
//  otherwise keep the old value
void _pp_vstore_float(float* dest, __pp_vec_float &src, __pp_mask &mask);
void _pp_vstore_int(int* dest, __pp_vec_int &src, __pp_mask &mask);

// Return calculation of (veca + vecb) if vector lane active
//  otherwise keep the old value
void _pp_vadd_float(__pp_vec_float &vecResult, __pp_vec_float &veca, __pp_vec_float &vecb, __pp_mask &mask);
void _pp_vadd_int(__pp_vec_int &vecResult, __pp_vec_int &veca, __pp_vec_int &vecb, __pp_mask &mask);

// Return calculation of (veca - vecb) if vector lane active
//  otherwise keep the old value
void _pp_vsub_float(__pp_vec_float &vecResult, __pp_vec_float &veca, __pp_vec_float &vecb, __pp_mask &mask);
void _pp_vsub_int(__pp_vec_int &vecResult, __pp_vec_int &veca, __pp_vec_int &vecb, __pp_mask &mask);

// Return calculation of (veca * vecb) if vector lane active
//  otherwise keep the old value
void _pp_vmult_float(__pp_vec_float &vecResult, __pp_vec_float &veca, __pp_vec_float &vecb, __pp_mask &mask);
void _pp_vmult_int(__pp_vec_int &vecResult, __pp_vec_int &veca, __pp_vec_int &vecb, __pp_mask &mask);

// Return calculation of (veca / vecb) if vector lane active
//  otherwise keep the old value
void _pp_vdiv_float(__pp_vec_float &vecResult, __pp_vec_float &veca, __pp_vec_float &vecb, __pp_mask &mask);
void _pp_vdiv_int(__pp_vec_int &vecResult, __pp_vec_int &veca, __pp_vec_int &vecb, __pp_mask &mask);


// Return calculation of absolute value abs(veca) if vector lane active
//  otherwise keep the old value
void _pp_vabs_float(__pp_vec_float &vecResult, __pp_vec_float &veca, __pp_mask &mask);
void _pp_vabs_int(__pp_vec_int &vecResult, __pp_vec_int &veca, __pp_mask &mask);

// Return a mask of (veca > vecb) if vector lane active
//  otherwise keep the old value
void _pp_vgt_float(__pp_mask &vecResult, __pp_vec_float &veca, __pp_vec_float &vecb, __pp_mask &mask);
void _pp_vgt_int(__pp_mask &vecResult, __pp_vec_int &veca, __pp_vec_int &vecb, __pp_mask &mask);

// Return a mask of (veca < vecb) if vector lane active
//  otherwise keep the old value
void _pp_vlt_float(__pp_mask &vecResult, __pp_vec_float &veca, __pp_vec_float &vecb, __pp_mask &mask);
void _pp_vlt_int(__pp_mask &vecResult, __pp_vec_int &veca, __pp_vec_int &vecb, __pp_mask &mask);

// Return a mask of (veca == vecb) if vector lane active
//  otherwise keep the old value
void _pp_veq_float(__pp_mask &vecResult, __pp_vec_float &veca, __pp_vec_float &vecb, __pp_mask &mask);
void _pp_veq_int(__pp_mask &vecResult, __pp_vec_int &veca, __pp_vec_int &vecb, __pp_mask &mask);

// Adds up adjacent pairs of elements, so
//  [0 1 2 3] -> [0+1 0+1 2+3 2+3]
void _pp_hadd_float(__pp_vec_float &vecResult, __pp_vec_float &vec);

// Performs an even-odd interleaving where all even-indexed elements move to front half
//  of the array and odd-indexed to the back half, so
//  [0 1 2 3 4 5 6 7] -> [0 2 4 6 1 3 5 7]
void _pp_interleave_float(__pp_vec_float &vecResult, __pp_vec_float vec);

// Add a customized log to help debugging
void addUserLog(const char * logStr);

#endif
