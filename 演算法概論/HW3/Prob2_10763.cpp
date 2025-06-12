#include <iostream>

using namespace std;

struct table{
    int from;
    int to;
};



int main()
{   
    
    int went[1000][1000];
    int students;
    int from,to;
    int flag,correct;
    while(cin>>students){
        if(students==0) break;
        for(int i=0;i<1000;i++)
			for(int j=0;j<1000;j++) 
				went[i][j]=0;
        correct=1;
        for(int i=0;i<students;i++){
            cin>>from>>to;
            went[from][to]++;
        }
        for(int i=0;i<1000;i++){
        	for(int j=0;j<1000;j++){
        		if(went[i][j]!=went[j][i]){
        			correct=0;
				}
			}
		}
        if(correct==1) cout<<"YES"<<endl;
        else cout<<"NO"<<endl;
    }

    return 0;
}
