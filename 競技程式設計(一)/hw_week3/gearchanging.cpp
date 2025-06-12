#include <iostream>
#include <set>
using namespace std;

int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    int M, N;
    float P;
    cin >> M >> N >> P;
    set<float> numbers;
    
    float crank[M], temp;
    for(int i = 0; i < M; i++){
        cin >> crank[i];
    }
    for(int i = 0; i < N; i++) {
        cin >> temp;
        for(int j = 0; j < M; j++)
            numbers.insert(crank[j]/temp);
    }
    
    for (auto i = numbers.begin(); next(i)!=numbers.end(); i++){
        if ((*(next(i)) / *i) > 1 + 0.01 * P){
           cout << "Time to change gears!";
           return 0;
        }
    }
    
    cout << "Ride on!";

    return 0;
}