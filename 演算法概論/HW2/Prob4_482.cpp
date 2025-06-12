#include <iostream>
#include <string>
using namespace std;

int arr[3000];
string newarr[3000];
int main(){
	int cases,indexstring,indexarr;
	string x;
	getline(cin,x);
	cases=stoi(x);
	while(cases--){
		indexstring=indexarr=0;
		getline(cin,x);
		getline(cin,x);
		for(int i=0;i<x.length();i++){
			if(x[i]==' '){
				arr[indexarr++]=stoi(x.substr(indexstring,i-indexstring));
				indexstring=i+1;
			}
		}
		arr[indexarr++]=stoi(x.substr(indexstring,x.length()-indexstring));
		//next
		getline(cin,x);
		indexstring=indexarr=0;
		for(int i=0;i<x.length();i++){
			if(x[i]==' '){
				newarr[indexarr++]=x.substr(indexstring,i-indexstring);
				indexstring=i+1;
			}
		}
		newarr[indexarr++]=x.substr(indexstring,x.length()-indexstring);
		
		for(int i=1;i<=indexarr;i++){
			for(int j=0;j<indexarr;j++){
				if(arr[j]==i){
					cout<<newarr[j]<<endl;
				}
			}
		}
		if(cases!=0)cout<<endl;
	}

	return 0;
}

