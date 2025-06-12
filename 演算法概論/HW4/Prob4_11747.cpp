#include <iostream>
#include <iomanip>
#include <algorithm> 
using namespace std;
int N,M,Q,cityA,cityB,costAB;
int roads[1002][1002],visited[1002];
#define maxint 2147483647
bool need[1002][1002];
int noneed[25005];
int edgefound;
struct edge{
    int x;
    int y;
    int cost;
};
edge edges[25005];
edge tmp[25005];
struct vertice{
    int index;
    int ancestor;
};
vertice vertices[1002];

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
    //cout<<vertices[1].ancestor;
    int ct=0;
    int ans=0;
    int edgecount =0;
    while(edgefound < N-1 && edgecount<M){
        edge newedge = edges[ct++];
        edgecount++;
        //cout<<find(newedge.x)<<"#"<<endl<<newedge.y;
        //cout<<find(newedge.y)<<endl;
        //cout<<newedge.cost<<" "<<newedge.x<< " "<<newedge.y<<endl;
        if(find(vertices[newedge.x].index)!=find(vertices[newedge.y].index)){
            ans+=newedge.cost;
            need[newedge.x][newedge.y] = need[newedge.y][newedge.x] = true;
            //cout<<newedge.x<<" "<<newedge.y<<"@"<<endl;
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

int main()
{   
    std::ios::sync_with_stdio(false);
    edge temp;
    while (cin>>N>>M && N!=0){
        edgefound = 0;
        for(int i=0;i<1002;i++){
            for(int j=0;j<1002;j++){
                roads[i][j] = maxint;
                need[i][j] = false;
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
        //cout<<N<<M<<endl<<endl;
        kruskal();
        int countnoneed=0;
        for(int i=0;i<N;i++){
            for(int j=i+1;j<N;j++){
                if(!need[i][j] && roads[i][j]!=maxint){
                    noneed[countnoneed++]=roads[i][j];
                    //cout<<i<<' '<<j<<endl;
                    //cout<<need[i][j]<<" "<<roads[i][j];
                }
            }
        }/*
        for(int i=0;i<3;i++){
            for(int q=0;q<3;q++){
                cout<<need[i][q]<<" ";
            }
            cout<<endl;
        }*/
        sort(noneed,noneed+countnoneed);
        if(countnoneed==0){
            cout<<"forest"<<endl;
        }
        else{
            for(int i=0;i<countnoneed;i++){
                cout<<noneed[i];
                if(i!=countnoneed-1) cout<<" ";
            }
            cout<<endl;
        }
    }

    return 0;
}

