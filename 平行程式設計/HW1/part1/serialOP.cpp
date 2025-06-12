// computes the absolute value of all elements in the input array
// values, stores result in output
void absSerial(float *values, float *output, int N)
{
  for (int i = 0; i < N; i++)
  {
    float x = values[i];
    if (x < 0)
    {
      output[i] = -x;
    }
    else
    {
      output[i] = x;
    }
  }
}

// accepts an array of values and an array of exponents
//
// For each element, compute values[i]^exponents[i] and clamp value to
// 9.999.  Store result in output.
void clampedExpSerial(float *values, int *exponents, float *output, int N)
{
  for (int i = 0; i < N; i++)
  {
    float x = values[i];
    int y = exponents[i];
    if (y == 0)
    {
      output[i] = 1.f;
    }
    else
    {
      float result = x;
      int count = y - 1;
      while (count > 0)
      {
        result *= x;
        count--;
      }
      if (result > 9.999999f)
      {
        result = 9.999999f;
      }
      output[i] = result;
    }
  }
}

// returns the sum of all elements in values
float arraySumSerial(float *values, int N)
{
  float sum = 0;
  for (int i = 0; i < N; i++)
  {
    sum += values[i];
  }

  return sum;
}