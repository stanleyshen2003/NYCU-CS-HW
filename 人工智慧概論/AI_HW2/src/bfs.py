import csv
edgeFile = 'edges.csv'


def bfs(start, end):
    '''
    1. read all the data from .csv
    2. do bfs with a queue iteratively until end is found:
            pop first element -> add the path adjacent to the point into queue and fromNode
    3. trace back the path and compute the total distant
    '''
    # Begin your code (Part 1)
    data = []
    queue = []
    queue.append(start)
    fromNode = tuple()
    visited = [start]
    path = []
    done = False
    count = 0
    with open(edgeFile, newline='') as csvfile:                             # read from csv
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)
        for row in reader:
            data.append(tuple(row))
    
    while(done == False):                                                   # loop & do bfs
        now = queue.pop(0)                                                  # pop a point from queue
        visited.append(now)                                                 # add it to visited
        for i in data:
            if (int(i[0]) == now and int(i[1]) not in visited and int(i[1]) not in queue):
                fromNode = fromNode + ((int(i[0]),int(i[1]),float(i[2])),)  # add possible path
                queue.append(int(i[1]))                                     # add the dest. node into queue
                if int(i[1])==end:                                          # found end -> break
                    done = True
    
    now = end                                                                       
    num_visited = len(visited)
    dist = 0
    while now!=start:                                                       # trace back
        for way in fromNode:
            if way[1] == now:                                               # if it is the path
                path.insert(0,now)                                          # add to path
                dist += way[2]                                              # add the distance
                now = way[0]                                                # renew dest. point
                break
    path.insert(0,start)                                                    # insert the start point
    return path,dist,num_visited
    # End your code (Part 1)


if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
