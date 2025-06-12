import csv
edgeFile = 'edges.csv'


def dfs(start, end):
    '''
    1. read all the data from .csv
    2. do bfs with a stack iteratively until end is found:
            pop last element -> add the path adjacent to the point into stack and fromNode
    3. trace back the path and compute the total distant
    '''
    # Begin your code (Part 2)
    data = []
    stack = []
    stack.append(start)
    fromNode = tuple()
    visited = [start]
    path = []
    done = False
    with open(edgeFile, newline='') as csvfile:                                 # read .csv
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)
        for row in reader:
            data.append(tuple(row))
    
    while(done == False):
        now = stack.pop()                                                       # pop the last element
        visited.append(now)
        for i in data:
            if (int(i[0]) == now and int(i[1]) not in visited and int(i[1]) not in stack):
                fromNode = fromNode + ((int(i[0]),int(i[1]),float(i[2])),)      # add to possible path
                stack.append(int(i[1]))                                         # add to stack
                if int(i[1])==end:                                              # break if end is found
                    done = True

    now = end                                                                   # trace back
    num_visited = len(visited)
    dist = 0
    while now!=start:
        for way in fromNode:
            if way[1] == now:
                path.insert(0,now)
                dist += way[2]
                now = way[0]
                break
    path.insert(0,start)
    return path,dist,num_visited
    # End your code (Part 2)


if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
