#include <iostream>
#include <sstream>
#include <map>
#include <vector>
#include <string>
#include <set>
using namespace std;

int main() {
    int num_orders;
    
    while (cin >> num_orders) {
        cin.ignore();
        
        map<string, set<string>> orders;

        for (int i = 0; i < num_orders; i++) {
            string line;
            getline(cin, line);

            istringstream iss(line);
            string customer, food;
            iss >> customer;

            while (iss >> food) {
                orders[food].insert(customer);
            }
        }

        for (const auto& order : orders) {
            cout << order.first;
            for (const auto& c : order.second)
                cout << ' ' << c;
            cout << '\n';
        }
        cout << '\n';

    }

    return 0;
}
