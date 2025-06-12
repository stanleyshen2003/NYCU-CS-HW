#ifndef LOGGER_H_
#define LOGGER_H_

#include <stdio.h>
#include <vector>
#include <string.h>
using namespace std;

#define MAX_INST_LEN 32

struct __pp_mask;

struct Log {
  char instruction[MAX_INST_LEN];
  unsigned long long mask; // support vector width up to 64
};

struct Statistics {
  unsigned long long utilized_lane;
  unsigned long long total_lane;
  unsigned long long total_instructions;
};

class Logger {
  private:
    vector<Log> log;
    Statistics stats;

  public:
    void addLog(const char * instruction, __pp_mask mask, int N = 0);
    void printStats();
    void printLog();
    void refresh() {
      stats.total_instructions = 0;
      stats.total_lane = 0;
      stats.utilized_lane = 0;
      fflush(stdout);
    };
};

#endif
