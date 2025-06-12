#include <iostream>
#include <sstream>
#include <map>
#include <vector>
#include <string>
#include <set>
#include <stdio.h>
using namespace std;

int main() {
    string program;
    
    int char_num;
    char stack[205];
    int top=-1;
    cin >> char_num;
    
    getline(cin, program);
    getline(cin, program);
    
    for (int i = 0; i < char_num; i++){
        if (program[i] == '(' || program[i] == '{' || program[i] == '[')
            stack[++top] = program[i];
        else if (program[i] == ')'){
            if (top == -1 || stack[top] != '('){
                cout << ") " << i << "\n";
                return 0;
            }
            top--;
        }
        else if (program[i] == ']'){
            if (top == -1 || stack[top] != '['){
                cout << "] " << i << "\n";
                return 0;
            }
            top--;
        }
        else if (program[i] == '}'){
            if (top == -1 || stack[top] != '{'){
                cout << "} " << i << "\n";
                return 0;
            }
            top--;
        }
    }
    
    cout << "ok so far\n";
    
    

    

    return 0;
}
