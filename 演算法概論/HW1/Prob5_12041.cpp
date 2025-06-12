#include <iostream>
#include <string>
typedef long long LL;
using namespace std;
LL fib[50];
void init(){
	fib[0]=fib[1]=1;
	for(int i=2;i<50;i++){
		fib[i]=fib[i-1]+fib[i-2];
	}
}
void find(unsigned long long N,LL left,LL right){
	if(left>right) return;
	if(N==0) cout<<0;
	else if(N==1) cout<<1;
	else{
		if(left<fib[N-2]){
			if(right>fib[N-2]-1){
				find(N-2,left,fib[N-2]-1);		// use both n-1 & n-2 
				find(N-1,0,right-fib[N-2]);	
			}
			else find(N-2,left,right);			// use only n-2
		}
		else find(N-1,left-fib[N-2],right-fib[N-2]);	// use only n-1
	}
}
int main(){
	int cases;
	unsigned long long N;
	LL i,j;
	cin>>cases;
	init();
	for(int k=0;k<cases;k++){
		cin>>N>>i>>j;
		if(N>50) N=50-(N%2);
		find(N,i,j);
		cout<<endl;
	}

	return 0;
}

