#include <iostream>
#include <thread>
#include <mutex>
#include <semaphore.h>


using namespace std;

static mutex io_mutex;
sem_t sem[100];
void count(int index) {
  
  int num = 1000000;
  while (num--) {}
  {
    // lock_guard<mutex> lock(io_mutex);
    sem_wait(&sem[index]);
    cout << "I'm thread " << index << ", local count: 1000000\n";
    sem_post(&sem[(index+1)%100]);
  }
}

int main(void) {
  thread t[100];
  sem_init(&sem[0], 0, 1);
  for(int i = 1; i < 100; i++)
    sem_init(&sem[i], 0, 0);
  for (int i = 0; i < 100; i++)
    t[i] = thread(count, i);

  for (int i = 0; i < 100; i++)
    t[i].join();
}
