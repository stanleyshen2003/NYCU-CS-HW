#include <iostream>

#define ll long long
using namespace std;


int main()
{
    ll a, b;

    cin >> a >> b;

    if (a % 2 != 0) {
        cout << 0 << endl;
    }
    else {
        ll k = a / 2;

        if (k % 2 != 0) {
             cout << k << endl;
        }
        else {
            if (b == 1) {
                cout << k << endl;
            } else {
                cout << 0 << endl;
            }
        }
    }

    return 0;
}