#include <iostream>
#include <list>
using namespace std;

int main(){
    int lines;
    cin >> lines;
    
    string line;
    getline(cin, line);
    
    for (int i = 0; i < lines; i++){
        getline(cin, line);
        list<char> front;
        list<char> end;
        char cursor_end = true;
        
        for (int i = 0; i < line.length(); i++){
            if (cursor_end) {
                if (line[i] == '<') {
                    if (!end.empty())
                    end.pop_back();
                }
                else if (line[i] == ']') {
                    continue;
                }
                else if (line[i] == '[') {
                    cursor_end = false;
                }
                else {
                    end.push_back(line[i]);
                }
            }
            else {
                if (line[i] == '<') {
                    if (!front.empty())
                        front.pop_back();
                }
                else if (line[i] == ']') {
                    end.splice(end.begin(), front);
                    front.clear();
                    cursor_end = true;
                }
                else if (line[i] == '[') {
                    end.splice(end.begin(), front);
                    front.clear();
                }
                else {
                    front.push_back(line[i]);
                }
            }
        }
        
        if (!cursor_end){
            end.splice(end.begin(), front);
            front.clear();
        }
        
        for (auto i : end){
                cout << i;
            }
            cout << "\n";
        
    }
    return 0;
}