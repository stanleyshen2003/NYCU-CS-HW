
#include <iostream>

using namespace std;

int max(int a,int b){
    if(a>b) return a;
    return b;
}

int main()
{
    int cases,n,t,m;
    cin>>cases;
    while(cases--){
        cin>>n>>t>>m;
        int cars[m],k=m,i=0,lasttime=0;
        while(k--){
            cin>>cars[i++];
        }
        if(m<=n){
            cout<<cars[m-1]+t<<" "<<1<<endl;
        }
        if(m%n==0){
            lasttime=0;
            i=n-1;
            while(i<m){
                lasttime=max(lasttime,cars[i])+2*t;
                i+=n;
            }
            lasttime-=t;
            cout<<lasttime<<" "<<m/n<<endl;
        }
        else{
            lasttime=cars[m%n-1]+2*t;
            i=n+m%n-1;
            while(i<m){
                lasttime = max(cars[i],lasttime)+2*t;
                i+=n;
            }
            lasttime-=t;
            cout<<lasttime<<" "<<m/n+1<<endl;
        }
    }

    return 0;
}

