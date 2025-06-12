#include <iostream>
#include <string>
using namespace std;
char table[14] = {'c','d','e','f','g','a','b','C','D','E','F','G','A','B'};
bool bottoms[14][10]={{0,1,1,1,0,0,1,1,1,1},{0,1,1,1,0,0,1,1,1,0},{0,1,1,1,0,0,1,1,0,0},
	{0,1,1,1,0,0,1,0,0,0},{0,1,1,1,0,0,0,0,0,0},{0,1,1,0,0,0,0,0,0,0},{0,1,0,0,0,0,0,0,0,0},
	{0,0,1,0,0,0,0,0,0,0},{1,1,1,1,0,0,1,1,1,0},{1,1,1,1,0,0,1,1,0,0},{1,1,1,1,0,0,1,0,0,0},
	{1,1,1,1,0,0,0,0,0,0},{1,1,1,0,0,0,0,0,0,0},{1,1,0,0,0,0,0,0,0,0}};
int returnIndex(char key){
	for(int i=0;i<14;i++){
		if(table[i]==key) return i;
	}
}
int main(){
	bool prekey[10];
	int ans[10];
	int cases,keyindex;
	bool nowkey[10];
	string notes;
	cin>>cases;
	for(int i=-1;i<cases;i++){
		for(int j=0;j<10;j++){
			prekey[j]=nowkey[j]=ans[j]=0;
		}
		getline(cin,notes);
		if(i==-1) continue;
		for(int j=0;j<notes.length();j++){
			keyindex = returnIndex(notes[j]);
			for(int k=0;k<10;k++){
				nowkey[k]=bottoms[keyindex][k];
			}
			for(int k=0;k<10;k++){
				if(prekey[k]==0 && nowkey[k]==1){
					ans[k]++;
				}
				prekey[k]=nowkey[k];
			}
		} 
		for(int j=0;j<10;j++){
			if(j!=9) cout<<ans[j]<<" ";
			else cout<<ans[j]<<endl;
		}
	}
	return 0;
}

