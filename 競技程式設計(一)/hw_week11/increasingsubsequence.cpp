# include <bits/stdc++.h>
using namespace std;

int main(){
    string line;
    while (getline(cin, line)) {
        if (line == "0") break;
        
        vector<int> nums;

        stringstream ss(line);
        string token;
        
        ss >> token;

        while (ss >> token)
            nums.push_back(stoi(token));
            
        vector<int> store, store_i, direction(nums.size());
        
        for (int i = 0; i < nums.size(); i++)
            direction[i] = i;
            
        for (int i = 0; i < nums.size(); i++){
            // cout << "i: " << i << "\n";
            int num = nums[i], idx;
            if (store.empty() || store[store.size()-1] < num){
                store.push_back(num);
                store_i.push_back(i);
                idx = store.size()-1;
            }
            else{
                auto it = lower_bound(store.begin(), store.end(), num);
                *it = num;
                idx = it - store.begin();
                store_i[idx] = i;
            }
            
            if (idx != 0){
                direction[i] = store_i[idx-1];
            }
            
        }

        cout << store.size() << " ";
        
        vector<int> print;
        
        int iter_id = store_i[store_i.size()-1];
        while (1){
            print.push_back(nums[iter_id]);
            if (direction[iter_id] == iter_id)
                break;
            iter_id = direction[iter_id];
        }
        reverse(print.begin(), print.end());
        for (int p:print)
            cout << p << " ";
        cout << "\n";
    }

    return 0;
}