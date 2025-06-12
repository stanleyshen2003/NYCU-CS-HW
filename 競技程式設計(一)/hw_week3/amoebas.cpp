#include <iostream>
#include <sstream>
#include <map>
#include <vector>
#include <string>
#include <set>
using namespace std;

int row, col;


void clear(bool** table, int r, int c){
    if (r < 0 || r>=row || c < 0 || c >= col || !table[r][c])
        return;
    table[r][c] = false;
    
    for (int i = -1; i < 2; i++)
        for (int j = -1; j < 2; j++)
            clear(table, r+i, c+j);
}


int main(){
    string pixel;
    
    cin >> row >> col;
    
    bool** table = new bool* [row];
    for (int i = 0; i < row; i++)
        table[i] = new bool [col];
        

    for (int i = 0; i < row; i++){
        cin >> pixel;

        for (int j = 0; j < col; j++) {
            if (pixel[j] == '#')
                table[i][j] = true;
            else
                table[i][j] = false;
        }
    }
    
    int ans = 0;
    
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            if (table[i][j]){
                ans++;
                clear(table, i, j);
            }
    
    cout << ans << '\n';
    
    for (int i = 0; i < row; i++)
        delete(table[i]);
    delete(table);

    return 0;
}
