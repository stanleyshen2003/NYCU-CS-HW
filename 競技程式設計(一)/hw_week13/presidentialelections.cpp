#include <bits/stdc++.h>
using namespace std;

int main()
{
    int s, d, c, f, u, required_vote, total, total_delegates = 0;
    cin >> s;
    vector<int> delegates, required_votes;
    
    for (int i = 0; i < s; i++){
        cin >> d >> c >> f >> u;
        delegates.push_back(d);
        total_delegates += d;
        if (u+c <= f)
            required_vote = INT_MAX;
        
        else{
            required_vote = max(0, ((c+f+u)/2+1)-c);
        }
        required_votes.push_back(required_vote);
    }
    
    vector<int> table(2017, INT_MAX);
    table[0] = 0;
    
    for (int i = 0; i < delegates.size(); i++){
        if (required_votes[i] == INT_MAX)
            continue;
        for (int j = 2016; j >= delegates[i]; j--){
            if (table[j-delegates[i]] != INT_MAX){
                table[j] = min(table[j-delegates[i]] + required_votes[i], table[j]);
            }
        }
    }
    
    total_delegates = total_delegates / 2 + 1;
    int min_votes = INT_MAX;
    for (int i = total_delegates; i < 2017; i++){
        min_votes = min(min_votes, table[i]);
    }
    
    if (min_votes == INT_MAX)
        cout << "impossible\n";
    else
        cout << min_votes << "\n";

    return 0;
}