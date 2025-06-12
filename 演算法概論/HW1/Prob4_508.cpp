#include <iostream>
#include <string>
using namespace std;
struct dicttype{
	char dictchar;
	string mor;
};
struct contexttype{
	string word;
	string morse_of_word; 
};
dicttype dict[50];
contexttype context[101];
contexttype newcontext[101];
bool check[100];
int inputdict=-1;
int inputcont=-1;
string inmorse;
string find(char a){
	for(int i=0;i<inputdict;i++){
		if(dict[i].dictchar==a)
			return dict[i].mor;
	}
}
contexttype makemorse(contexttype x){
	string a=x.word;
	string morse="";
	for(int i=0;i<a.length();i++){
		morse+=find(a[i]);
	}
	x.morse_of_word=morse;
	return x;
}

void makeex(contexttype x,int a){
	for(int i=0;i<a;i++){
		if(context[i].morse_of_word==x.morse_of_word)
			newcontext[i].word=newcontext[i].word+"!";
	}
}
string mvlast(string a){
    string ans="";
    for(int i=0;i<a.length()-1;i++){
        ans+=a[i];
    }
    return ans;
}
int abso(int a){
	if(a<0) return a*(-1);
	else return a;
}
void findword(string in){
	for(int i=0;i<inputcont;i++){
		if(in==context[i].morse_of_word){
			cout<<newcontext[i].word<<endl;
			return;
		}
	}
	for(int i=0;i<inputcont;i++){
		for(int j=0;j<in.length();j++){
			if(check[i] && context[i].morse_of_word.length()>j){
				if(in[j]!=context[i].morse_of_word[j]){
					check[i]=0;
					break;
				}
			}
		}		
	}
	int min=99;
	for(int i=0;i<inputcont;i++){
		if(check[i]==1 && min>abso(in.length()-context[i].morse_of_word.length())){
			min=abso(in.length()-context[i].morse_of_word.length());
		}
	}
	for(int i=0;i<inputcont;i++){
		if(check[i]==1 && min==abso(in.length()-context[i].morse_of_word.length())){
			cout<<context[i].word<<"?"<<endl;
			return;
		}
	}
	
}
int main(){
	while(cin>>dict[++inputdict].dictchar){
		if(dict[inputdict].dictchar=='*') break;
		cin>>dict[inputdict].mor;
	}
	while(cin>>context[++inputcont].word){
		if(context[inputcont].word=="*") break;
		context[inputcont]=makemorse(context[inputcont]);
		newcontext[inputcont].word=context[inputcont].word;
		newcontext[inputcont]=makemorse(newcontext[inputcont]);
		makeex(newcontext[inputcont],inputcont);
	}
	while(cin>>inmorse){
		if(inmorse=="*") break;
		for(int i=0;i<100;i++){
			check[i]=1;
		}
		findword(inmorse);
	}

	return 0;
}

