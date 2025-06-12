#include<iostream>
#include<string>
using namespace std;

int main(){
    int P, R;
    int cases = 0;
    string namea, nameb;
    while(cin>>P>>R){
        if(P == 0 && R == 0){
            break;
        }
        string nameindex[100];
        for(int i=0;i<100;i++){
            nameindex[i]="";
        }
        int idnum = 0;
        int table[51][51];
        for(int i=0;i<51;i++){
            for(int j=0;j<51;j++){
                table[i][j] = 9999999;
            }
        }
        for(int i=0;i<R;i++){
            cin>>namea>>nameb;
            int id1 = 100, id2 = 100;
            for(int i=0;i<idnum;i++){
                if(nameindex[i]==namea){
                    id1=i;
                }
                if(nameindex[i]==nameb){
                    id2=i;
                }
            }
            if(id1>90){
                id1=idnum;
                nameindex[idnum++] = namea;
            }
            if(id2>90){
                id2=idnum;
                nameindex[idnum++] = nameb;
            }
            table[id1][id2] = table[id2][id1] = 1;
        }
        for(int k=0;k<P;k++){
            for(int i=0;i<P;i++){
                for(int j=0;j<P;j++){
                    int tmp = table[i][k] + table[k][j];
                    if(tmp<table[i][j] && i!=j && i!=k && j!=k){
                        table[i][j] = tmp;
                        //table[j][i] = tmp;
                    }
                }
            }
        }
        int ans = 0;
        /* Get the maximum value */
        int flag=0;
        for(int i=0;i<P;i++){
            if(flag){
                break;
            }
            for(int j=0;j<P;j++){
                if(i != j && table[i][j] > 100){
                    ans = -1;
                    flag = 1;
                    break;
                }
                if(table[i][j] < 100 && ans != -1 && ans < table[i][j]){
                    ans = table[i][j];
                }
            }
        }
        /*
        for(int i=0;i<P;i++){
            for(int j=0;j<P ;j++){
                cout<<table[i][j]<<" ";
            }
            cout<<endl;
        }*/
        cases+=1;
        if(ans == -1){
            cout<<"Network "<<cases<<": DISCONNECTED"<<endl<<endl;
        }
        else{
            cout<<"Network "<<cases<<": "<<ans<<endl<<endl;
        }
    }
    return 0;
}
