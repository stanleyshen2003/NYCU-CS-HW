#include <iostream>
using namespace std;
long long ans;
int temp[500000];
void merge(int arr[],int left,int mid,int right){
	int indexleft=left,indexright=mid+1;
	for(int i=left;i<=right;i++){
	    if(indexleft>mid){
	        temp[i]=arr[indexright++];
	        continue;
	    }
	    if(indexright>right){
	        temp[i]=arr[indexleft++];
	        continue;
	    }
	    if(arr[indexleft]<=arr[indexright]){
	        temp[i]=arr[indexleft++];
	        continue;
	    }
	    else{
	        ans+=indexright-i;
	        temp[i]=arr[indexright++];
	    }
	}
	for(int i=left;i<=right;i++){
	    arr[i]=temp[i];
	}
}

void mergesort(int arr[],int left,int right){
    int mid=(right+left)/2;
	if(left<right){
		mergesort(arr,left,mid);
		mergesort(arr,mid+1,right);
		merge(arr,left,mid,right);
	}
}

int main(){
	int n;
	int list[500000];
	while(cin>>n && n!=0){
		ans=0;
		for(int i=0;i<n;i++){
			cin>>list[i];		
		}
		mergesort(list,0,n-1);
		cout<<ans<<endl;
	}


	return 0;
}

