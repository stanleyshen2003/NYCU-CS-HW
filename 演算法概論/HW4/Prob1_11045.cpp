#include <iostream>
#include <string>
using namespace std;
string table[6]={"XXL","XL","L","M","S","XS"};
int ans[6];
int n,m,yes;

struct suitchoose{
	int x;
	int y;
};
suitchoose suits[31];

int trans(string x){
    for(int i=0;i<6;i++){
        if(table[i]==x){
            return i;
        }
    }
    return 0;
}

void solve(int i){
	if(i==m){
		yes=1;
		return;
	}
	if(ans[suits[i].x]>0){
		ans[suits[i].x]-=1;
		solve(i+1);
		ans[suits[i].x]+=1;
	}
	if(ans[suits[i].y]>0){
		ans[suits[i].y]-=1;
		solve(i+1);
		ans[suits[i].y]+=1;
	}
}

int main()
{
    int cases,sets;
    string a,b;
    cin>>cases;
    while(cases--){
        yes=0;
        cin>>n>>m;
        sets = n/6;
        for(int i=0;i<6;i++){
        	ans[i] = sets;
		}
        for(int i=0;i<m;i++){
            cin>>a>>b;
            suits[i].x=trans(a);
            suits[i].y=trans(b);
        }
        solve(0);
        if(yes) cout<<"YES"<<endl;
        else cout<<"NO"<<endl;
    }

    return 0;
}

