#include <iostream>
#include <iomanip>
using namespace std;
int N,M,Q,cityA,cityB,costAB;
int roads[3002][3002],visited[3002],dp[3002][3002];

bool need[3002][3002];
int edgefound;
struct edge{
    int x;
    int y;
    int cost;
};
edge edges[3002*3002];
edge tmp[3002*3002];
struct vertice{
    int index;
    int ancestor;
};
vertice vertices[3002];

int find(int a){
    while(vertices[a].ancestor!=a){
        a=vertices[a].ancestor;
    }
    return a;
}

void merge(int front,int mid,int end){
    int index=front,count=front;
    int rs=mid+1;
    while(count<=mid && rs<=end){
        if(edges[count].cost>edges[rs].cost){
            tmp[index++]=edges[rs++]; 
            
        }
        else{
            tmp[index++] = edges[count++];
            
        }
    }
    while(count<=mid){
        tmp[index++] = edges[count++];
    }
    while(rs<=end){
        tmp[index++] = edges[rs++];
    }
    for(int i=front;i<=end;i++){
        edges[i]=tmp[i];
    }
}


void sort(int front,int end){
    if(front < end){
        int mid = (front+end)/2;
        sort(front,mid);
        sort(mid+1,end);
        merge(front,mid,end);
    }
}

int kruskal(){
    for(int i=0;i<N;i++){
        vertices[i].index = vertices[i].ancestor = i;
    }
    int ct=0;
    int ans=0;
    
    while(edgefound<N-1){
        edge newedge = edges[ct++];
        if(find(newedge.x)!=find(newedge.y)){
            ans+=newedge.cost;
            need[newedge.x][newedge.y] = need[newedge.y][newedge.x] = true;
            edgefound++;
            if(find(newedge.x)<find(newedge.y)){
                vertices[find(newedge.y)].ancestor = find(newedge.x);
            }
            else{
                vertices[find(newedge.x)].ancestor = find(newedge.y);
            }
        }
    }
    return ans;
}

int min(int a,int b){
    if (a<b) return a;
    return b;
}

int dfs(int rt,int u,int v)
{
    int s=999999999;
    for(int i=0;i<N;i++){
        if(need[u][i]==true){
            if(i==v)continue;
            int tmp=dfs(rt,i,u);
            if(s>tmp){
                s=tmp;
            }
            dp[u][i]=dp[i][u]=min(dp[u][i],tmp);
        }
    }
    if(rt!=v){
        s=min(s,roads[rt][u]);
    }
    return s;
}



int main()
{   std::ios::sync_with_stdio(false);
    edge temp;
    while (cin>>N>>M && N!=0){
        edgefound = 0;
        for(int i=0;i<N;i++){
            for(int j=0;j<N;j++){
                roads[i][j] = 999999999;
                need[i][j] = false;
                dp[i][j] = 999999999;
            }
        }
        for(int i=0;i<M;i++){
            cin>>cityA>>cityB>>costAB;
            roads[cityA][cityB] = costAB;
            roads[cityB][cityA] = costAB;
            temp.x=cityA;
            temp.y=cityB;
            temp.cost=costAB;
            edges[i] = temp;
        }
        sort(0,M-1);
        cin>>Q;
        double ans = (double)kruskal();
        
        for(int i=0;i<N;i++){
            dfs(i,i,-1);
        }
        double finalans=0;
        
        for(int i=0;i<Q;i++){
            cin>>cityA>>cityB>>costAB;
            if(!need[cityA][cityB]){
                finalans+=ans;
            }
            else{
                finalans+=(float)(ans-roads[cityA][cityB]+min(dp[cityB][cityA],costAB));
            }
        }
        
        cout<<fixed<<setprecision(4)<<finalans/Q<<endl;
    }

    return 0;
}

