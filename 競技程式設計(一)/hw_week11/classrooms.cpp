# include <bits/stdc++.h>
using namespace std;
#define pii pair<int, int>

int main(){
    int k, n;
    
    vector<pii> tasks;
    cin >> k >> n;
    int start, end;
    for (int i = 0; i < k; i++){
        cin >> start >> end;
        tasks.push_back({end, start});
    }
    
    sort(tasks.begin(), tasks.end());
    
    int ans = 0;
    set<pii> slot_using;

    for (int i = 0; i < n; i++)
        slot_using.insert({0, i});
    
    for (int i = 0; i < tasks.size(); i++){
        int end = tasks[i].first;
        int start = tasks[i].second;
        
        auto it = slot_using.lower_bound({start, -1});
        
        if (it == slot_using.begin()){
            continue;
        }
        
        it--;
        int machine = (*it).second;
        ans++;
        slot_using.erase(it);
        slot_using.insert({end, machine});
    }
    
    cout << ans << "\n";
    
    return 0;
}