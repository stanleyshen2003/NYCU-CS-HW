#include <iostream>
using namespace std;
int main(){
	int testcases,students;
	int nowbiggest,ans,second;
	cin>>testcases;
	for(int i=0;i<testcases;i++){
		cin>>students;
		cin>>nowbiggest;
		cin>>second;
		ans=nowbiggest-second;
		if(second>nowbiggest) nowbiggest=second;
		for(int j=2;j<students;j++){
			cin>>second;
			if(nowbiggest-second>ans) ans=nowbiggest-second;
			if(second>nowbiggest) nowbiggest = second;
		}
		cout<<ans<<endl;
	}

	return 0;
}

