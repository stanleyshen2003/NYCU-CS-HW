#include <bits/stdc++.h>
using namespace std;
#define ll long long

struct node{
    int number;
    ll sum;
    int ancestor;
    int total_element;
};


int find_root(int index, vector<node>& table){
    if (index == table[index].ancestor)
        return index;
    int root = find_root(table[index].ancestor, table);
    table[index].ancestor = root;
    return root;
}

int main(){

    int N, M;
    int operation, a, b, root1, root2, r3;
    
    while (cin >> N >> M){
        
        vector<node> table;
        int real_index[N+1];

        for (int i = 0; i < N+1; i++){
            node temp;
            temp.number = i;
            temp.sum = i;
            temp.ancestor = i;
            temp.total_element = 1;
            table.push_back(temp);
            real_index[i] = i;
        }
        
        while (M--){
            cin >> operation;
            
            if (operation == 1){
                cin >> a >> b;
                root1 = find_root(real_index[a], table);
                root2 = find_root(real_index[b], table);
                
                if (root1 == root2)
                    continue;
                
                table[root2].ancestor = root1;
                table[root1].sum += table[root2].sum;
                table[root1].total_element += table[root2].total_element;
            }
            
            
            else if (operation == 2){
                cin >> a >> b;
                root1 = find_root(real_index[a], table);
                root2 = find_root(real_index[b], table);
                
                if (root1 == root2)
                    continue;
                
                table[root1].sum -= a;
                table[root1].total_element -= 1;
                table[root2].sum += a;
                table[root2].total_element += 1;
                node temp;
                temp.number = a;
                temp.sum = a;
                temp.ancestor = root2;
                temp.total_element = 1;
                table.push_back(temp);
                real_index[a] = table.size() -1;
            }
            
            else {
                cin >> a;
                
                root1 = find_root(real_index[a], table);
                cout << table[root1].total_element << " " << table[root1].sum << "\n";
            }
        }
    }

    return 0;
}