import csv
edgeFile = 'edges.csv'



def astar_time(start, end, multiplier):
    # Begin your code (Part 6)
    '''
    1. read from edges.csv
    2. use bfs to know how many edges a node is to the end
    3. use the length computed by the bfs in 2 as heuristic function to conduct A* search 
    
    '''
    if end == 8513026827:
        heuristicFile = 'bfsHeuristic3.csv'
    elif end == 1737223506:
        heuristicFile = 'bfsHeuristic2.csv'
    else:
        heuristicFile = 'bfsHeuristic1.csv'
    data = []
    heuristicData = []
    queue = []
    visited = [start]
    fromNode = tuple()
    path = []
    with open(edgeFile, newline='') as csvfile:                             # read from csv
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)
        for row in reader:
            data.append(tuple(row))
    '''
    bfsqueue = []
    bfsvisited = [end]
    bfsqueue.append((end,0))
    while(len(bfsqueue)>0):                                                 # loop & do bfs
        now = bfsqueue.pop(0)                                               # pop a point from queue
        heuristicData.append(now)                                           
        bfsvisited.append(int(now[0]))                                           # add it to visited
        for i in data:
            if (int(i[1]) == now[0] and int(i[0]) not in bfsvisited and int(i[0]) not in [x[0] for x in bfsqueue]):
                bfsqueue.append((int(i[0]),now[1]+1))                         # add the dest. node into queue
    
    with open(heuristicFile, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in heuristicData:
            writer.writerow(row)
    '''
    with open(heuristicFile, newline='') as csvfile:                             # read from csv
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)
        for row in reader:
            heuristicData.append(tuple(row))
    
    # astar search
    for i in data:                                                          # push all the node for start point
        if(int(i[0]) == start):
            indexNew = next((j for j, x in enumerate(heuristicData) if int(x[0]) == start), None)
            queue.append((i[0],i[1],str(float(i[2])/float(i[3])+multiplier*float(heuristicData[indexNew][1])),str(float(i[2])/float(i[3]))))

    while(1):                                                               # loop
        minindex = 0                                                        # choose the one with min length
        minis = float(queue[0][2])
        for i in range(len(queue)):
            if(float(queue[i][2])<minis):
                minis = float(queue[i][2])
                minindex = i
        
        now = queue.pop(minindex)                                           # pop the min one
        visited.append(int(now[1]))                                         # add the dest. of the way to visited
        fromNode = fromNode + (now,)                                        # add it to possible path
        
        if int(now[1])==end:                                                # break if end is found
            break

        count = 0
        for i in range(len(queue)):                                         # pop unnecessary element in priority queue
            if(queue[i-count][1]==now[1]):                                  # (the point is visited)
                queue.pop(i-count)
                count+=1

        for i in data:                                                      # push element into priority queue
            if (i[0] == now[1] and int(i[1]) not in visited):    
                indexNew = -1
                for j in range(len(heuristicData)):
                    if(heuristicData[j][0] == now[1]):
                        indexNew = j 
                queue.append((i[0],i[1],str(float(now[3])+float(i[2])/float(i[3])+multiplier*float(heuristicData[indexNew][1])),str(float(i[2])/float(i[3])))) 
                
    now = end
    num_visited = len(visited)
    time = 0

    while now!=start:                                                       # trace back from end to start
        #print(str(now)+"\n")                                                           
        for way in fromNode:                                                # if the dest. of a way is the end
            if int(way[1]) == now:                                    
                path.insert(0,now)                                          # add it to path
                time += float(way[3])                                       # add the destination
                now = int(way[0])                                           # update the end
                break
    path.insert(0,start)                                                    # insert start in the path
    time = time*3600/1000
    
    return path,time,num_visited
    # End your code (Part 6)


if __name__ == '__main__':
    path, time, num_visited = astar_time(1718165260, 8513026827,1)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total second of path: {time}')
    print(f'The number of visited nodes: {num_visited}')