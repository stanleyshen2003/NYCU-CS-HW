#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float value_float_vec, result, ones_float_vec, max_float_vec;
  __pp_vec_int exp_int_vec, zeros_int_vec, ones_int_vec;
  __pp_mask ones_mask, zeros_mask, not_done_mask, exp_zero_mask, over_maximum_mask, greater_than_n_mask, smaller_than_n_mask;

  ones_mask = _pp_init_ones();
  zeros_mask = _pp_init_ones(0);
  ones_float_vec = _pp_vset_float(1.f);
  max_float_vec = _pp_vset_float(9.999999f);
  ones_int_vec = _pp_vset_int(1);
  zeros_int_vec = _pp_vset_int(0);

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    _pp_vload_float(value_float_vec, values + i, ones_mask);
    _pp_vload_int(exp_int_vec, exponents + i, ones_mask);
    result = _pp_vset_float(1.f);

    if (i + VECTOR_WIDTH >= N){
      if (N % VECTOR_WIDTH == 0)
        smaller_than_n_mask = _pp_init_ones();
      else
        smaller_than_n_mask = _pp_init_ones(N % VECTOR_WIDTH);
      greater_than_n_mask = _pp_mask_not(smaller_than_n_mask);
      not_done_mask = smaller_than_n_mask;
      _pp_vmove_int(exp_int_vec, zeros_int_vec, greater_than_n_mask);
      _pp_vmove_float(result, value_float_vec, greater_than_n_mask);
    }

    // if exp is zero
    _pp_veq_int(exp_zero_mask, exp_int_vec, zeros_int_vec, ones_mask);

    // set the value to done
    not_done_mask = _pp_mask_not(exp_zero_mask);

    // if exp not 0
    // while not all done
    while (_pp_cntbits(not_done_mask) != 0){
      // multiply not done values
      _pp_vmult_float(result, result, value_float_vec, not_done_mask);

      // substract exp by one
      _pp_vsub_int(exp_int_vec, exp_int_vec, ones_int_vec, not_done_mask);

      // if over 9.999999f
      _pp_vgt_float(over_maximum_mask, result, max_float_vec, not_done_mask);
      over_maximum_mask = _pp_mask_and(over_maximum_mask, not_done_mask);

      // assign to result
      _pp_vmove_float(result, max_float_vec, over_maximum_mask);

      // set the corresponding exp to 0
      _pp_vmove_int(exp_int_vec, zeros_int_vec, over_maximum_mask);

      // if exp is 0, set them to done
      _pp_veq_int(exp_zero_mask, exp_int_vec, zeros_int_vec, ones_mask);
      not_done_mask = _pp_mask_not(exp_zero_mask);
    }
    // write the rest of the values
    if (i + VECTOR_WIDTH >= N){
      _pp_vstore_float(output + i, result, smaller_than_n_mask);
    }
    else
      _pp_vstore_float(output + i, result, ones_mask);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  float sum = 0.0;
  int add_times = 0;
  int temp = VECTOR_WIDTH;
  __pp_vec_float row;
  __pp_mask ones_mask = _pp_init_ones();
  while (temp > 1)
  {
    temp >>= 1;
    add_times++;
  }

  __pp_vec_float sums = _pp_vset_float(0.f);

  for (int i = 0; i < N; i += VECTOR_WIDTH){
    _pp_vload_float(row, values + i, ones_mask);
    _pp_vadd_float(sums, sums, row, ones_mask);
  }


  for (int j = 0; j < add_times; j++){
    _pp_hadd_float(sums, sums);
    _pp_interleave_float(sums, sums);
  }
  

  return sums.value[0];
}