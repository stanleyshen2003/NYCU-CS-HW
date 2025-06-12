#include <bits/stdc++.h>
using namespace std;

int main()
{
    int n, m;
    cin >> n;
    vector<int> prices(n);
    for (int i = 0; i < n; i++)
        cin >> prices[i];
    cin >> m;
    vector<int> total(m);
    for (int i = 0; i < m; i++)
        cin >> total[i];
        
    vector<vector<map<int, int>>> table(30001, vector<map<int, int>>());
    
    for (int i = 0; i < n; i++){
        if (table[prices[i]].size() >= 2)
            continue;
        map<int, int> tmp;
        tmp[i]++;
        table[prices[i]].push_back(tmp);
        for (int j = prices[i]+1; j < 30001; j++){
            if (table[j].size() >= 2)
                continue;
            vector<map<int, int>> last = table[j-prices[i]];
            if (last.size()==1){
                last[0][i]++;
                table[j].push_back(last[0]);
            }
            else if(last.size()>=2){
                table[j].push_back({});
                table[j].push_back({});
            }
        }
    }
    for (int i = 0; i < m; i++){
        vector<map<int, int>> current = table[total[i]];
        if (current.size()==0)
            cout << "Impossible\n";
        else if (current.size()>=2)
            cout << "Ambiguous\n";
        else{
            vector<int> tmp;
            for (pair<int, int> item: current[0]){
                for (int j = 0; j < item.second; j++)
                    tmp.push_back(item.first);
            }
            for (int item: tmp)
                cout << item+1 << " ";
            cout << "\n";
        }
        
    }

    return 0;
}