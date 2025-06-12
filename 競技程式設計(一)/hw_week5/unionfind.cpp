
#include <bits/stdc++.h>
using namespace std;


int find(int a, int table[]){
    if (table[a] == a)
        return a;
    table[a] = find(table[a], table);
    return table[a];
}

void unionab(int a, int b, int table[]){
    int root1 = find(a, table);
    int root2 = find(b, table);
    table[root1] = root2;
}


int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    int N, Q;
    char operation;
    int a, b;
    
    cin >> N >> Q;
    
    int table[N];
    for(int i = 0; i < N; i++)
        table[i] = i;
    
    while (Q--){
        cin >> operation >> a >> b;
        if (operation == '=')
            unionab(a, b, table);
        else{
            if (find(a, table) == find(b, table))
                cout << "yes\n";
            else
                cout << "no\n";
        }

    }

    return 0;
}