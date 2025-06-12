#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/wait.h>
#include <cstring>
#include <cstdlib>

void copy(char* to,char* from){
    for(int i=0;i<100;i++){
        to[i] = from[i];
    }
}

void getlast(char* tmp, char* input, int num){
    for(int i=0;i<100-num;i++){
        tmp[i] = input[num+i];
    }
    printf("%s in func\n", tmp);
}

int main() {
    // Use execl to execute the pwd command
    char *input = (char*)malloc(sizeof(char) * 100);
    char * temp = (char*)malloc(sizeof(char) * 100);
    pid_t pid;

    while(1){
        if(input) {
            if (!strncmp(input, "cat ", 4)){
            printf("\n");
            }
            free(input);
        }
        input = (char*)malloc(sizeof(char) * 100);
        
        printf("osh> ");
        fgets(input, sizeof(char)*100, stdin);
        input[strlen(input) - 1] = '\0';
        if(!strcmp(input, "exit")){
            printf("Process end");
            exit(0);
        }
        pid = fork();

        if(pid == 0){
            copy(temp, input);
            if (!strncmp(input, "cat ", 4)){
                temp = temp + 4;
                execl("/bin/cat", "cat", temp, NULL);
            }
            if(!strcmp(input,"pwd")) execl("/bin/pwd", "pwd", (char *)0);
            else if(!strcmp(input,"date")) execl("/bin/date", "date", (char *)0);
            else if(!strcmp(input,"ps")) execl("/bin/ps", "ps", (char *)0);
            else if(!strcmp(input,"ps -f")) execl("/bin/ps", "ps", "-f", NULL);
            else if(!strcmp(input,"ls")) execl("/bin/ls", "ls", (char *)0);
            else if(!strcmp(input,"ls -a")) execl("/bin/ls", "ls", "-a", NULL);
            else if(!strcmp(input,"ls -al")) execl("/bin/ls", "ls", "-al", NULL);
            else if(!strcmp(input,"ls -l")) execl("/bin/ls", "ls", "-l", NULL);
            else if(!strcmp(input,"ls -a -l") || !strcmp(input, "ls -l -a")) execl("/bin/ls", "ls", "-a", "-l", NULL);
            exit(0);
        }
        else{
            wait(NULL);
        }
    }
    return 0;
}
