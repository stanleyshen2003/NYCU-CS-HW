#include<iostream>
using namespace std;

void recurse( int i, int n, int arr[] ){
    if( i > n ) return;
    if( i == n ){
        for( int j = 0 ; j < i ; j++ )
            cout<<"  ";
        cout<<"writeln(";
        for( int j = 0 ; j < n ; j++ ){
            cout<<(char)('a'+arr[j]);
            if(j!=n-1)  cout<<",";
        }
        cout<<")"<<endl;
        return;
    }
    for( int j = i ; j >= 0 ; j-- ){
        for( int k = 0 ; k < i ; k++ )
            cout<<"  ";
        if( j == i ) cout<<"if "<<(char)('a'+arr[j-1])<<" < "<<(char)('a'+i)<<" then"<<endl;
        else if(j==0) cout<<"else"<<endl;
        else cout<<"else if "<<(char)('a'+arr[j-1])<<" < "<<(char)('a'+i)<<" then"<<endl;
        int newarr[10] = {0};
        for( int k = 0 ; k<j ; k++ )
            newarr[k] = arr[k];
        newarr[j] = i;
        for( int k = j+1 ; k<=i ; k++ )
            newarr[k] = arr[k-1];
        recurse(i+1, n, newarr);
    }
} 

int main(){
    int n,cases;
    cin>>cases;
    while(cases--){
        cin>>n;
        cout<<"program sort(input,output);"<<endl;
        cout<<"var"<<endl;
        for( int j = 0 ; j < n ; j++ ){
            cout<<(char)('a'+j);
            if(j!=n-1)  cout<<",";
        }
        cout<<" : integer;"<<endl;
        cout<<"begin"<<endl;
        cout<<"  readln(";
        for( int j = 0 ; j < n ; j++ ){
            cout<<(char)('a'+j);
            if(j!=n-1)  cout<<",";
        }
        cout<<");"<<endl;
        int arr[10] = {0};
        recurse( 1, n, arr );
        cout<<"end."<<endl;
        if(cases) cout<<endl;
    }
    return 0;
}



