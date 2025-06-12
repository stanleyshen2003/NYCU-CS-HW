#include<iostream>
using namespace std;
struct test{
    int *a;
    int *b;
    int *c;
};

int main(){
    test x[20];
    for(int i=0;i<20;i++)
    cout << x[i].a << x[i].b << x[i].c ;

    return 0;
}