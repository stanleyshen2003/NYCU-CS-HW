#include <iostream>
using namespace std;
#include <string>
bool iscases(string h){
    if(h.length()<3){
        return true;
    }
    return false;
}
int* transform(string h){
    int count=1;
    int now=0;
    int *ans=new int[22];
    int *newans=new int [22];
    for(int i=0;i<22;i++){
        ans[i]=0;
    }
    for(int i=0;i<h.length();i++){
        if(h[i]!=' '){
            now*=10;
            now+=h[i]-'0';
        }
        else{
            ans[count++]=now;
            now=0;
        }
    }
    ans[count]=now;
    for(int i=0;i<22;i++){
        newans[ans[i]]=i;
    }
    return newans;
}
int main(){
	int cases,inputcor=1;
	string x;
	int* ans=new int[22];
	int* stdans=new int[22];
	int table[22][22];
	while(getline(cin,x)){
	    if(iscases(x)){
	        cases=stoi(x);
	        inputcor=1;
	    }
	    else{
	        if(inputcor){
	            ans=transform(x);
	            inputcor=0;
	        }
	        else{
	            stdans=transform(x);
	            for(int i=0;i<22;i++){
	                for(int j=0;j<22;j++){
	                    table[i][j]=0;
            	    }
            	}
            	for(int i=1;i<=cases;i++){
            	    for(int j=1;j<=cases;j++){
            	        if(ans[i]==stdans[j]){
            	            table[i][j]=table[i-1][j-1]+1;
            	        }
                        else{
                            if(table[i-1][j]>=table[i][j-1]){
                                table[i][j]=table[i-1][j];
                            }
                            else table[i][j]=table[i][j-1];
                        }    	   
            	    }
            	}
            	cout<<table[cases][cases]<<endl;
	            
	        }
	        
	    }
	} 

	return 0;
}

