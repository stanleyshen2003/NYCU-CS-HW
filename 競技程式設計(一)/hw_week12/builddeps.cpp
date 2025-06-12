#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <sstream>

using namespace std;

int main() {
    int n;
    cin >> n;
    cin.ignore();

    unordered_map<string, vector<string>> dependencies;
    unordered_map<string, vector<string>> reverse_graph;
    unordered_set<string> all_files;

    for (int i = 0; i < n; ++i) {
        string line;
        getline(cin, line);
        stringstream ss(line);
        string file, dep;
        ss >> file;
        file.pop_back();
        all_files.insert(file);
        while (ss >> dep) {
            dependencies[file].push_back(dep);
            reverse_graph[dep].push_back(file);
        }
    }

    string changed_file;
    getline(cin, changed_file);

    // BFS to find all affected files
    unordered_set<string> affected;
    queue<string> q;
    q.push(changed_file);

    while (!q.empty()) {
        string curr = q.front(); q.pop();
        if (affected.count(curr)) continue;
        affected.insert(curr);
        for (const string& dependent : reverse_graph[curr]) {
            q.push(dependent);
        }
    }

    // Topological sort on affected subgraph
    unordered_map<string, int> in_degree;
    for (const string& f : affected) {
        in_degree[f] = 0;
    }

    for (const string& f : affected) {
        for (const string& dep : dependencies[f]) {
            if (affected.count(dep)) {
                in_degree[f]++;
            }
        }
    }

    queue<string> topo_queue;
    for (const auto& [file, deg] : in_degree) {
        if (deg == 0) {
            topo_queue.push(file);
        }
    }

    vector<string> result;
    while (!topo_queue.empty()) {
        string curr = topo_queue.front(); topo_queue.pop();
        result.push_back(curr);
        for (const string& dependent : reverse_graph[curr]) {
            if (in_degree.count(dependent)) {
                in_degree[dependent]--;
                if (in_degree[dependent] == 0) {
                    topo_queue.push(dependent);
                }
            }
        }
    }

    for (const string& f : result) {
        cout << f << endl;
    }

    return 0;
}
