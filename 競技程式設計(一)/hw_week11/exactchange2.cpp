#include <bits/stdc++.h>
using namespace std;
int main(){
    int cases, targetP, diff, num_coin;
    cin >> cases;
    
    while (cases--){
        cin >> targetP >> num_coin;
        int coins[num_coin];
        for (int i = 0; i < num_coin; i++)
            cin >> coins[i];
            
        int total = 0;
        for (int i = 0; i < num_coin; i++)
            total += coins[i];
        diff = total - targetP;
        
        int coin_nums[diff+1];
        int totals[diff+1];
        memset(coin_nums, 0, (diff+1) * sizeof(int));
        memset(totals, 0, (diff+1) * sizeof(int));
        
        for (int i = 0; i < num_coin; i++){
            for (int j = diff; j >= coins[i]; j--){
                if (coin_nums[j] < coin_nums[j-coins[i]] + coins[i]){
                    coin_nums[j] = coin_nums[j-coins[i]] + coins[i];
                    totals[j] = totals[j-coins[i]] + 1;
                }
                else if (coin_nums[j] == coin_nums[j-coins[i]] + coins[i]){
                    totals[j] = max(totals[j], totals[j-coins[i]] + 1);
                }
            }
        }
        
        cout << total - coin_nums[diff] << " " << num_coin - totals[diff] << "\n";
    }

    return 0;
}