#ifndef _SYRAH_CYCLE_TIMER_H_
#define _SYRAH_CYCLE_TIMER_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

long long currentTicks();
static double secondsPerTick();
static double currentSeconds();

long long currentTicks()
{
  unsigned int a, d;
  asm volatile("rdtsc"
               : "=a"(a), "=d"(d));
  return (unsigned long long)a | (long long)d << 32;
}

// Return the conversion from ticks to seconds.
static double secondsPerTick()
{
  static int initialized = 0;
  static double secondsPerTick_val;
  if (initialized)
    return secondsPerTick_val;
  FILE *fp = fopen("/proc/cpuinfo", "r");
  char input[1024];
  if (!fp)
  {
    fprintf(stderr, "CycleTimer::resetScale failed: couldn't find /proc/cpuinfo.");
    exit(-1);
  }
  // In case we don't find it, e.g. on the N900
  secondsPerTick_val = 1e-9;
  while (!feof(fp) && fgets(input, 1024, fp))
  {
    // NOTE(boulos): Because reading cpuinfo depends on dynamic
    // frequency scaling it's better to read the @ sign first
    float GHz, MHz;
    if (strstr(input, "model name"))
    {
      char *at_sign = strstr(input, "@");
      if (at_sign)
      {
        char *after_at = at_sign + 1;
        char *GHz_str = strstr(after_at, "GHz");
        char *MHz_str = strstr(after_at, "MHz");
        if (GHz_str)
        {
          *GHz_str = '\0';
          if (1 == sscanf(after_at, "%f", &GHz))
          {
            //printf("GHz = %f\n", GHz);
            secondsPerTick_val = 1e-9f / GHz;
            break;
          }
        }
        else if (MHz_str)
        {
          *MHz_str = '\0';
          if (1 == sscanf(after_at, "%f", &MHz))
          {
            //printf("MHz = %f\n", MHz);
            secondsPerTick_val = 1e-6f / GHz;
            break;
          }
        }
      }
    }
    else if (1 == sscanf(input, "cpu MHz : %f", &MHz))
    {
      //printf("MHz = %f\n", MHz);
      secondsPerTick_val = 1e-6f / MHz;
      break;
    }
  }
  fclose(fp);

  initialized = 1;
  return secondsPerTick_val;
}

static double currentSeconds()
{
  return currentTicks() * secondsPerTick();
}

#endif // #ifndef _SYRAH_CYCLE_TIMER_H_
