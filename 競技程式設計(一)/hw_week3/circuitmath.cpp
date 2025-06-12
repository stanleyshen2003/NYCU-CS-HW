#include <iostream>
#include <sstream>
#include <map>
#include <vector>
#include <string>
#include <set>
using namespace std;

int main() {
    int num_wire;
    
    cin >> num_wire;
    
    vector<bool> wires;
    bool stack[250];
    int top = -1;
    char TF;
    for(int i = 0; i < num_wire; i++){
        cin >> TF;
        if (TF == 'T')
            wires.push_back(true);
        else
            wires.push_back(false);
    }
    
    char circuit;
    
    while (cin >> circuit){
        if (circuit == '+'){
            stack[top-1] = stack[top] | stack[top-1];
            top--;
        }
        else if (circuit == '-')
            stack[top] = !stack[top];
        else if (circuit == '*'){
            stack[top-1] = stack[top] & stack[top-1];
            top--;
        }
        else
            stack[++top] = wires[circuit - 'A'];
    }
    if (stack[0])
        cout << "T\n";
    else
        cout << "F\n";

    return 0;
}
