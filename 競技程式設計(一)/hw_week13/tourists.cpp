#include <bits/stdc++.h>
using namespace std;

const int MAXN = 200005;
const int LOG = 20;

vector<int> tree[MAXN];
int up[MAXN][LOG];
int depth[MAXN];
int n;

void dfs(int v, int p) {
    up[v][0] = p;
    for (int i = 1; i < LOG; ++i) {
        if (up[v][i - 1] != -1)
            up[v][i] = up[up[v][i - 1]][i - 1];
        else
            up[v][i] = -1;
    }
    for (int u : tree[v]) {
        if (u != p) {
            depth[u] = depth[v] + 1;
            dfs(u, v);
        }
    }
}

int lca(int u, int v) {
    if (depth[u] < depth[v]) swap(u, v);

	// binary lifting
    for (int i = LOG - 1; i >= 0; --i) {
        if (up[u][i] != -1 && depth[up[u][i]] >= depth[v])
            u = up[u][i];
    }
    if (u == v) return u;

    for (int i = LOG - 1; i >= 0; --i) {
        if (up[u][i] != -1 && up[u][i] != up[v][i]) {
            u = up[u][i];
            v = up[v][i];
        }
    }

    return up[u][0];
}

int get_distance(int u, int v) {
    int ancestor = lca(u, v);
    return depth[u] + depth[v] - 2 * depth[ancestor] + 1;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> n;
    for (int i = 1; i < n; ++i) {
        int u, v;
        cin >> u >> v;
        tree[u].push_back(v);
        tree[v].push_back(u);
    }

    memset(up, -1, sizeof(up));
    depth[1] = 0;
    dfs(1, -1);  // root at node 1

    long long total = 0;
    for (int i = 1; i <= n; ++i) {
        for (int j = 2 * i; j <= n; j += i) {
            total += get_distance(i, j);
        }
    }

    cout << total << "\n";
    return 0;
}
