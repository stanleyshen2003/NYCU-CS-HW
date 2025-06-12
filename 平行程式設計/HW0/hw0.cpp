#include <iostream>
#include <cstdlib>
#include <time.h> 

using namespace std;

int main() {
    long long total = 1e8;
    long long sum = 0;
    double x, y;
    srand(time(NULL));
    for (long long i = 1; i <= total; i++) {
        x = (double)rand() / RAND_MAX;
        y = (double)rand() / RAND_MAX;
        if (x * x + y * y <= 1) {
            sum++;
        }
    }
    cout << "PI = " << 4 * (double)sum / (double)total << endl;
    return 0;
}