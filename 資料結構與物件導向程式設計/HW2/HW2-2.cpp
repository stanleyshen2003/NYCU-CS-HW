#include <iostream>
using namespace std;

// define a structure for storing
struct Path{
    int roadWidth;
    int Ts[3];
};

int main()
{
    int N, M;
    cin >> N >> M;
    
    // create a graph
    Path ** graph;
    graph = new Path *[N + 1];
    for(int i = 0; i < N+1; i++){
        graph[i] = new Path [N + 1];
    }
    // reset the graph
    for(int i = 0; i < N+1; i++){
        for(int j = 0; j < N+1; j++){
            graph[i][j].roadWidth = 0;
            for(int k = 0; k < 3; k++){
                graph[i][j].Ts[k] = 9999999;
            }
        }
    }
    
    // input into the graph
    int from, to, width, truckTime, bikeTime, carTime;
    for(int i = 0;i<M;i++){
        cin >> from >> to >> width >> truckTime >> bikeTime >> carTime;
        graph[from][to].roadWidth = graph[to][from].roadWidth = width;
        graph[from][to].Ts[0] = graph[to][from].Ts[0] = truckTime;
        graph[from][to].Ts[1] = graph[to][from].Ts[1] = bikeTime;
        graph[from][to].Ts[2] = graph[to][from].Ts[2] = carTime;
    }

    // For the three kind of vehicle, if the vehicleW is bigger than road width, 
    // set the time for the road using the vehicle to 9999999, which is the same
    // as their is no path.
    int vehicleW;
    for(int i = 0; i < 3; i++){
        cin >> vehicleW;
        for(int j = 0; j < N + 1; j++){
            for(int k = 0; k < N + 1; k++){
                if(graph[j][k].roadWidth < vehicleW){
                    graph[j][k].Ts[i] = 9999999;
                }
            }
        }
    }
    int p;
    
    // the minimum possible path for each pair of vertex
    int ** mingraph;
    mingraph = new int *[N + 1];
    for(int i = 0; i < N+1; i++){
        mingraph[i] = new int [N + 1];
    }
    int smallest;
    for(int i=0;i<N+1;i++){
        for(int j=0;j<N+1;j++){
            smallest = graph[i][j].Ts[0];				// here is to find the minimum
            if(smallest > graph[i][j].Ts[1]) smallest = graph[i][j].Ts[1];
            if(smallest > graph[i][j].Ts[2]) smallest = graph[i][j].Ts[2];
            mingraph[i][j] = smallest;
        }
    }
    
    // since we may need to ask for the shortest path for more than source, 
    // so I implement the Floyd_Warshall algorithm to find the shortest path from each node. 
    for(int i = 0; i < N + 1; i++){
        for(int j = 0; j < N + 1; j++){
            for(int k = 0; k < N + 1; k++){
                if(mingraph[j][k] >= mingraph[j][i] + mingraph[i][k]){
                    mingraph[j][k] = mingraph[j][i] + mingraph[i][k];
                }
            }
        }
    }
    
    // input query and output the minimum path of three vehicle
    cin >> p;
    while(p--){
        cin >> from >> to;
        cout << mingraph[from][to] << endl;
    }
    
    return 0;
}

