#include <iostream>
#include <algorithm>
using namespace std;
int table[1000];
int length;
int arr[10];
void resetT(){
	for(int i=0;i<1000;i++){
		table[i]=-1;
	}
	length=0;
}

bool found(int a){
	for(int i=0;i<length;i++){
		if(table[i]==a) return true;
	}
	return false;
}

int big2small(int a){
	int power=0;
	int temp=a;
	int ans=0,min;
	while(temp){
		power++;
		temp/=10;
	}
	temp=a;
	for(int i=0;i<power;i++){
		arr[i]=temp%10;
		temp/=10;
	} 
	sort(arr,arr+power);
	for(int i=power-1;i>=0;i--){
		ans*=10;
		ans+=arr[i];
	}
	return ans;
}
int reverse(int a){
	int ans=0;
	while(a){
		ans*=10;
		ans+=a%10;
		a/=10;
	}
	return ans;
}

int main(){
	int origin;
	int first,second;
	while(cin>>origin && origin!=0){
		resetT();
		cout<<"Original number was "<<origin<<endl;
		table[0]=origin;
		while(1){
			first=big2small(table[length]);
			second=reverse(first);
			cout<<first<<" - "<<second<<" = "<<first-second<<endl;
			table[++length]=first-second;
			if(found(first-second)) break;
		}
		cout<<"Chain length "<<length<<endl;
		cout<<endl;
	}

	return 0;
}

