#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>

volatile int child_done = 0;

void show(int fork_id) {
    printf("Fork %d. I'm the child %d, my parent is %d.\n", fork_id, getpid(), getppid());
    fflush(stdout);
}

void signal_handler(int signo) {
    if (signo == SIGUSR1) {
        child_done = 1;
    }
}

int main() {
    signal(SIGUSR1, signal_handler);

    pid_t main_pid = getpid();
    printf("Main Process ID: %d\n\n", main_pid);
    fflush(stdout);

    pid_t pid;
    pid = fork(); // Fork 1
    if (pid == 0) {
        show(1); 
    } else if (pid > 0) {
        while(!child_done);
        child_done = 0;

        pid = fork(); // Fork 2
        if (pid == 0) {
            show(2);
        } else {
            while(!child_done);
            child_done = 0;
        }
    }
    
    pid = fork(); // Fork 3
    if (pid == 0) {
        show(3);
    } else {
        while(!child_done);
        child_done = 0;
    }

    if (getpid() != main_pid) { 
        kill(getppid(), SIGUSR1);
    }
    sleep(10);

    return 0;
}
