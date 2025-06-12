# include <bits/stdc++.h>
using namespace std;

int main(){
    
    int C, n;
    
    while (cin >> C){
        cin >> n;
        
        int costs[n], weights[n], table[C+1];
        bool uses[n][C+1];
        
        memset(table, 0, sizeof(table));
        memset(uses, false, sizeof(uses));
            
        for (int i = 0; i < n; i++){
            cin >> costs[i] >> weights[i];
        }
        
        for (int i = 0; i < n; i++){
            for (int j = C; j >= weights[i]; j--){
                if (table[j] < table[j-weights[i]] + costs[i]){
                    table[j] = table[j-weights[i]] + costs[i];
                    uses[i][j] = true;
                }
            }
        }
        
        vector<int> items;
        int weight = C;
        for (int i = n-1; i >= 0; i--){
            if (uses[i][weight]){
                items.push_back(i);
                weight -= weights[i];
            }
        }
        
        cout << items.size() << "\n";
        if (items.size() != 0)
            cout << items[0];
        for (int i = 1; i < items.size(); i++)
            cout << " " << items[i];
        cout << "\n";
    }

    return 0;
}