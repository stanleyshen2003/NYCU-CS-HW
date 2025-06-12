#include <iostream>
using namespace std;

int n, global_count = 0;

bool is_prime(int num) {
    if (num == 1) return false;

    for (int i = 2; i * i <= num; i++) {
        if (num % i == 0) {
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    cin >> n;
    
    for (int i = 1; i <= n; i++) {
        if (is_prime(i)) global_count++;
    }
    
    cout << global_count << endl;

    return 0;
}
