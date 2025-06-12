#include <iostream>
#include <algorithm>
using namespace std;
int main(){
	int years[2000000];
	int n;
	while(cin>>n){
		if(n==0) break;
		for(int i=0;i<n;i++){
			cin>>years[i];
		}
		sort(years,years+n);
		for(int i=0;i<n-1;i++){
			cout<<years[i]<<" ";
		}
		cout<<years[n-1]<<endl;
	}

	return 0;
}

