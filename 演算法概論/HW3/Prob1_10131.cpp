#include <iostream>

using namespace std;

struct elephant{
    int height;
    int IQ;
    int index;
};

elephant elephants[5000];
int ans[5000];
int last[5000];

void selsort(elephant a[5000],int n){
    elephant temp;
    for(int i=0;i<n;i++){
        int min=i;
        for(int j=i;j<n;j++){
            if(a[j].height<a[min].height){
                min=j;
            }
        }
        temp=a[min];
        a[min]=a[i];
        a[i]=temp;
    }
}

int main()
{   int n=0;
    while(cin>>elephants[n].height>>elephants[n++].IQ){}
    n--;
    for(int i=0;i<n;i++){
        elephants[i].index=i;
    }
    selsort(elephants,n);
    for(int i=n-1;i>0;i--){
        int maxstep=1;
        last[i]=i;
        for(int j=i;j<n;j++){
        	if(elephants[j].IQ<elephants[i].IQ && ans[j]+1>maxstep){
        		maxstep=ans[j]+1;
        		last[i]=j;
			}
		}
		ans[i]=maxstep;
    }
    int max=1,maxindex=0;
    for(int i=0;i<n;i++){
        if(ans[i]>max){
            max=ans[i];
            maxindex=i;
        } 
    }
    cout<<max<<endl;
    cout<<elephants[maxindex].index+1<<endl;
    while(last[maxindex]!=maxindex){
        cout<<elephants[last[maxindex]].index+1<<endl;
        maxindex=last[maxindex];
    }
    return 0;
}
