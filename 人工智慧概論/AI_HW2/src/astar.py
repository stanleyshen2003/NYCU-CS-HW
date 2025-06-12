import csv
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'


def astar(start, end):
    '''
    1. read data from the 2 csv
    2. push nodes for starting point
    3. loop:
            pop the min g(n) + h(n) -> break if dest. is end -> add to visited and fromNode -> push adjacent nodes into queue
    4. trace back
    '''
    # Begin your code (Part 4)
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
    with open(heuristicFile, newline='') as csvfile:                        # read from csv
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)
        for row in reader:
            heuristicData.append(tuple(row))

    for i in data:                                                          # push all the node for start point
        if(int(i[0]) == start):
            indexNew = next((j for j, x in enumerate(heuristicData) if x[0] == i[1]), None)
            if(end == 1079387396):
                    queue.append((i[0],i[1],str(float(i[2])+float(heuristicData[indexNew][1])),i[2],i[2])) 
            elif(end == 1737223506):
                    queue.append((i[0],i[1],str(float(i[2])+float(heuristicData[indexNew][2])),i[2],i[2])) 
            elif(end == 8513026827):
                    queue.append((i[0],i[1],str(float(i[2])+float(heuristicData[indexNew][3])),i[2],i[2])) 

    minis = 0

    while(1):                                                               # loop
        minindex = 0                                                        # choose the one with min length
        minis = float(queue[0][2])
        for i in range(len(queue)):
            if(float(queue[i][2])<minis):
                minis = float(queue[i][2])
                minindex = i
        
        now = queue.pop(minindex)                                           # pop the min one
        visited.append(int(now[1]))                                         # add the dest. of the way to visited
        fromNode = fromNode + (now,)                                        # add it to path
        
        if int(now[1])==end:                                                # break if end is found
            break
        count = 0
        for i in range(len(queue)):                                         # pop unnecessary element in priority queue
            if(queue[i-count][1]==now[1]):                                  # (the point is visited)
                queue.pop(i-count)
                count+=1

        for i in data:                                                      # push element into priority queue
            if (i[0] == now[1] and int(i[1]) not in visited):     
                indexNew = next((j for j, x in enumerate(heuristicData) if x[0] == i[1]), None)              
                if(end == 1079387396):
                    queue.append((i[0],i[1],str(float(now[4])+float(i[2])+float(heuristicData[indexNew][1])),i[2],str(float(i[2])+float(now[4])))) 
                elif(end == 1737223506):
                    queue.append((i[0],i[1],str(float(now[4])+float(i[2])+float(heuristicData[indexNew][2])),i[2],str(float(i[2])+float(now[4]))))
                elif(end == 8513026827):
                    queue.append((i[0],i[1],str(float(now[4])+float(i[2])+float(heuristicData[indexNew][3])),i[2],str(float(i[2])+float(now[4]))))
                
    now = end
    num_visited = len(visited)
    dist = 0

    while now!=start:                                                       # trace back from end to start
        #print(str(now)+"\n")                                                           
        for way in fromNode:                                                # if the dest. of a way is the end
            if int(way[1]) == now:                                          
                path.insert(0,now)                                          # add it to path
                dist += float(way[3])                                       # add the destination
                now = int(way[0])                                           # update the end
                break
    path.insert(0,start)                                                    # insert start in the path
    return path,dist,num_visited
    # End your code (Part 4)


if __name__ == '__main__':
    path, dist, num_visited = astar(1718165260, 8513026827)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
