#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "globals.h"
#include "randdp.h"
#include "timers.h"
#include "cg_impl.h"

void init(double *zeta);
void iterate(double *zeta, int *it);

int main(int argc, char *argv[])
{
  int i, j, k, it;

  double zeta;

  double t, t_total;

  //char Class;
  logical verified;
  double zeta_verify_value, epsilon, err;

  char *t_names[T_last];

  for (i = 0; i < T_last; i++)
  {
    timer_clear(i);
  }

  timer_start(T_init);

  zeta_verify_value = VALID_RESULT;

  printf("\nCG start...\n\n");
  printf(" Size: %11d\n", NA);
  printf(" Iterations: %5d\n", NITER);
  printf("\n");

  init(&zeta);

  zeta = 0.0;

  //---------------------------------------------------------------------
  //---->
  // Do one iteration untimed to init all code and data page tables
  //---->                    (then reinit, start timing, to niter its)
  //---------------------------------------------------------------------
  for (it = 1; it <= 1; it++)
  {
    iterate(&zeta, &it);
  } // end of do one iteration untimed

  //---------------------------------------------------------------------
  // set starting vector to (1, 1, .... 1)
  //---------------------------------------------------------------------
  for (i = 0; i < NA + 1; i++)
  {
    x[i] = 1.0;
  }

  zeta = 0.0;

  timer_stop(T_init);

  printf(" Initialization time = %15.3f seconds\n", timer_read(T_init));
  t_total += timer_read(T_init);

  timer_start(T_bench);

  //---------------------------------------------------------------------
  //---->
  // Main Iteration for inverse power method
  //---->
  //---------------------------------------------------------------------
  for (it = 1; it <= NITER; it++)
  {
    iterate(&zeta, &it);
  } // end of main iter inv pow meth

  timer_stop(T_bench);

  //---------------------------------------------------------------------
  // End of timed section
  //---------------------------------------------------------------------

  t = timer_read(T_bench);
  t_total += t;

  printf("\nComplete...\n");

  epsilon = 1.0e-10;
  err = fabs(zeta - zeta_verify_value) / zeta_verify_value;
  if (err <= epsilon)
  {
    verified = true;
    printf(" VERIFICATION SUCCESSFUL\n");
    printf(" Zeta is    %20.13E\n", zeta);
    printf(" Error is   %20.13E\n", err);
  }
  else
  {
    verified = false;
    printf(" VERIFICATION FAILED\n");
    printf(" Zeta                %20.13E\n", zeta);
    printf(" The correct zeta is %20.13E\n", zeta_verify_value);
  }

  printf("\n\nExecution time : %lf seconds\n\n", t);

  printf("Total Time: %lf seconds\n\n", t_total);

  return 0;
}
