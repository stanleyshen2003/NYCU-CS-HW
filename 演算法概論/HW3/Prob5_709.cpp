#include <iostream>
#include <string>
using namespace std;

int notfinish(string x){
    for(int i=0;i<x.length();i++){
        if(x[i]==' ')
            return 1;
    }
    return 0;
}
string table[10000];
int total=0;
string slice(string x){
    int i=0;
    while(x[i]!=' ') i++;
    string a="";
    for(int j=0;j<i;j++){
        a+=x[j];
    }
    if(a!=""){
		table[total++]=a;
    }
    a="";
    for(int j=i+1;j<x.length();j++){
        a+=x[j];
    }
    return a;
}

int main()
{
    string width;
    int intwid;
    string x;
    while(getline(cin,width)){
        intwid=stoi(width);
        int i=0;
        while(1){
            getline(cin,x);
            
            if(x.length()==0){
                break;
            }
            while(notfinish(x)){
                x=slice(x);
            }
        }
        for(int q=0;q<50;q++){
        	cout<<table[q]<<endl;
        }
        i=0;
        int letter;
        int noword=0;
        while(letter<intwid && i<total){
            if(letter+table[i].length()>intwid){
                int nowlen=0;
                for(int q=noword;q<i;q++){
                    nowlen+=table[i].length();
                }
                int words=i-noword;
                int spaces=intwid-nowlen;
                for(int q=noword;q<i-1;q++){
                    cout<<table[q];
                    for(int f=0;f<spaces/(words-1);f++){
                        cout<<" ";
                    }
                }
                if(spaces%(words-1)==0){
                    for(int f=0;f<spaces/(words-1);f++)   cout<<" ";
                    cout<<table[i-1];
                }
                else{
                    for(int f=0;f<spaces/(words-1)-1;f++)   cout<<" ";
                    cout<<table[i-1];
                }
                cout<<endl;
                noword=i;
            }
            letter+=table[i++].length()+1;
        }
    }
  
    return 0;
}


