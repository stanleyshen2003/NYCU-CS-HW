#include <iostream>
#include <string>
using namespace std;

int square[105][105];
int dir[4][4]={{1,0},{-1,0},{0,1},{0,-1}};
char table[20] = {'a','b','c','d','e','f','g','h','i','j','A','B','C','D','E','F','G','H','I','J'};

int toint(char a){
    for(int i=0;i<20;i++){
        if(table[i]==a) return i;
    }
    return -1;
}



string tobinary(int a){
    string x="";
    string temp=" ";
    int tmp;
    while(a){
        tmp = a%2;
        temp[0]=(char)(tmp+'0');
        x=temp+x;
        a/=2;
    }
    return x;
}

struct point{
    int x,y;
};
point que[10000];
int front,endq;
void push(point x){
    que[front]=x;
    front=(front+1)%10000;
}
point pop(){
    int x = endq;
    endq = (endq+1)%10000;
    return que[x];
}

void bfs(int show[105][105],int edge,int step){
    point first; first.x=0;first.y=0;
    push(first);
    while(front!=endq){
        point newe = pop();
        int x = newe.x;int y=newe.y;
        for(int i=0;i<4;i++){
            if(x+dir[i][0]<edge && x+dir[i][0]>=0 && y+dir[i][1]>=0 && y+dir[i][1]<edge && show[x+dir[i][0]][y+dir[i][1]]==-1){
                show[x+dir[i][0]][y+dir[i][1]]=show[x][y]+1;
                point tmp; tmp.x=x+dir[i][0];tmp.y=y+dir[i][1];
                push(tmp);
            }
        }
        if(show[edge-1][edge-1]!=-1){
            while(front!=endq) pop();
            break;
        }
    }
}


int main(){
    int edge;
    string temp;
    while(cin>>edge){
        for(int i=0;i<edge;i++){
            cin>>temp;
            for(int j=0;j<edge;j++){
                square[i][j] = toint(temp[j]);   
            }
        }
        int ans=10000;
        for(int i=0;i<1024;i++){
            string num=tobinary(i);
            while(num.length()<10)    num="0"+num;
            int newt[105][105];
            for(int j=0;j<edge;j++){
                for(int k=0;k<edge;k++){
                    newt[j][k]=10000;
                    if(num[square[j][k]%10]=='1' && square[j][k]>=10){
                        newt[j][k]=-1;
                    }
                    else if(num[square[j][k]%10]=='0' && square[j][k]<10){
                        newt[j][k]=-1;
                    }
                }
            }
            if(newt[0][0]==10000 || newt[edge-1][edge-1]==10000) continue;
            bfs(newt,edge,0);
            if(newt[edge-1][edge-1]<ans && newt[edge-1][edge-1]>0)    
                ans = newt[edge-1][edge-1];
            if(i==192){
                //cout<<i<<"!"<<endl;
                /*for(int q=0;q<edge;q++){
                    for(int j=0;j<edge;j++){
                        cout<<newt[q][j]<<" ";
                    }
                    cout<<endl;
                }*/
            }    
        }
        
        //cout<<ans;
        if(ans==10000) cout<<-1<<endl;
        else cout<<ans+2<<endl;
    }

    return 0;
}

