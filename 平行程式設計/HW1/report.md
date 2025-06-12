# PP lab1

## Q1

### Q1-1

Table for VECTOR_WIDTH & vector utilization

| 2 | 4 | 8 | 16 |
| -------- | -------- | -------- | -------- |
| 73.6%     | 68.3%     | 65.5%     | 63.7%     |

Vector utilization decreases as VECTOR_WIDTH increases. Vector utilization checks the percentage of the lane used in the vector. Difference between the iterations that need to be executed to finish the calculation of an element in a single vector will have high impact on vector utilization. 
Suppose we have VECTOR_WIDTH = 4, first index exp = 10, last 3 index exp = 0, the vector will be low utilized (about 25%) because the last 3 index don't need much computation.
Now suppose we have VECTOR_WIDTH = 2, and with the same data as the last example. The first vector will have approximately 50% utilization, and the second one will have 100% (but it only create a few vector operations, so won't affect the overall utilization a lot). Compared to VECTOR_WIDTH = 4, having a smaller width reduce useless computation on the last two elements, creating a higher vector utilization.


<!-- ![image](https://hackmd.io/_uploads/BJcQm-m0C.png)
![image](https://hackmd.io/_uploads/S1yyIWX0A.png)
![image](https://hackmd.io/_uploads/H1-PQbXRA.png)
![image](https://hackmd.io/_uploads/S1k5XW7RR.png) -->

## Q2
### Q2-1

Add line 4-6 to tell clang that the arrays are aligned. The number is 32 because AVX2 uses 32 Byte registers.

```cpp=
#include "test.h"
void test1(float *__restrict a, float *__restrict b, float *__restrict c, int N) {
  __builtin_assume(N == 1024);
  a = (float *)__builtin_assume_aligned(a, 32);
  b = (float *)__builtin_assume_aligned(b, 32);
  c = (float *)__builtin_assume_aligned(c, 32);

  for (int i=0; i<I; i++) {
    for (int j=0; j<N; j++) {
      c[j] = a[j] + b[j];
    }
  }
}
```

### Q2-2
Execution time (seconds).
- Speed up after vectorize - 4x
- Speed up after vectorize with AVX2 - 8x

| No vector | With vector | AVX2 |
| -------- | -------- | -------- |
| 6.940095     | 1.72826   | 0.861764  |

After vectorization, the speed up is 4, meaning that it will process 4 floats in 1 vector. That means the bit width of the vector register 4\*4 bytes (float in c use 4 bytes). As for AVX2 registers, the bit width is 8\*4 bytes.

### Q2-3
The compiler gives the following message.
```
test2.c:11:5: remark: loop not vectorized: unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop [-Rpass-analysis=loop-vectorize]
    for (int j = 0; j < N; j++)
    ^
test2.c:11:5: remark: loop not vectorized [-Rpass-missed=loop-vectorize]
```
In the code, the compiler is not sure whether the pointer a, b, and c overlaps. 
```cpp
for (int j = 0; j < N; j++) {
    c[j] = a[j];
    if (b[j] > a[j])
        c[j] = b[j];
}
```

Consider the following example.
c = a + 1, using a vector width 1, 2 will generate different results because there are read/write operations in a loop that depend on each other.

Using vector width 1, c[1] equals to max(a0,b0,b1) after the second iteration.
Execution:
1. (first iteration) c[0] = max(a0, b0)
2. (second iteration) c[1] = max(a1, b1) = max(c[0], b1) = max(a0, b0, b1)
<center>
    <img src="https://hackmd.io/_uploads/H1l5tl4C0.png" style="width:50%;">
</center>
![image](g)


Using vector width 2, c[1] equals to min(a0, b+1).
Execution:
1. (copy value) c[0:1] = a[0:1], which means c[0] = a[1] = a0, c[1] = a1
2. (if condition) c[0] = max(a0, b0), c[1] = a1 if b1>a0 else b1.

<center>
    <img src="https://hackmd.io/_uploads/rkOW5eEC0.png" style="width:50%;">
</center>

This shows that the result differs between different vector width, causing unexpected output, so the compiler says it is unsafe, thus not vectorizing the code.
