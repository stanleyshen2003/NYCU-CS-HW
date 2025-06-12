#include <iostream>
using namespace std;

/*

I created two struct - edge & vertice

find() : find the ancestor of a vertice, the vertices chained
together have the same ancestor.

merge() & sort() : sort the edges (from min cost to max cost)

kruskal() : find the minimum spanning using kruskal's algorithm

*/
struct edge{        
    int x;          // one vertice
    int y;          // the other vertice
    int cost;       // the cost(supply power) between vertices
};

edge edges[1002*1002];
edge tmp[1002*1002];

struct vertice{
    int index;		// index of this vertice
    int ancestor;	// ancestor of this vertice
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


int kruskal(int Num){
    
    for(int i=0;i<Num;i++){             // reset the vertices
        vertices[i].index = vertices[i].ancestor = i;
    }
    int ct = 0;
    int ans = 0;
    int edgefound = 0;
    
    while(edgefound<Num-1){             // while not finish finding
        edge newedge = edges[ct++];     // use unused edge with minimum cost
        if(find(newedge.x)!=find(newedge.y)){   // if ancestor is different
            ans+=newedge.cost;                  // add the cost of the edge (needed)
            
            if(find(newedge.x)<find(newedge.y)){	// update the ancestors
                vertices[find(newedge.y)].ancestor = find(newedge.x);
            }
            else{
                vertices[find(newedge.x)].ancestor = find(newedge.y);
            }
            edgefound++;
        }
    }
    return ans;
}

int main()
{
    int Num;
    cin>>Num;
    int firstnode,secondnode,distance;
    int count = 0;
    while((cin >> firstnode) && !cin.eof()){
        cin>>secondnode>>distance;
        edges[count].x = firstnode;
        edges[count].y = secondnode;
        edges[count++].cost = distance;
    }
    sort(0,count-1);
    
    int ans;
    ans = kruskal( Num);
    cout<<ans<<endl;
    return 0;
}

