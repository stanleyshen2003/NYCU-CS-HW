cases = int(input())

table = {}

def f(x, y, ab, c, d):
    if (x, y) in table:
        return table[(x, y)]
    if x <= 0 or y <= 0:
        return d
    
    ans = 0
    for i in range(0, len(ab), 2):
        ans += f(x-ab[i], y-ab[i+1], ab, c, d)
    ans += c
    table[(x, y)] = ans
    return ans

while (cases):
    param = input().split()
    
    ab = param[:-2]
    ab = [int(x) for x in ab]
    c = int(param[-2])
    d = int(param[-1])
    
    query = input().split()
    query = [int(x) for x in query]
    
    for i in range(0, len(query), 2):
        print(f(query[i], query[i+1], ab, c, d))
    
    table = {}
    
    cases -= 1