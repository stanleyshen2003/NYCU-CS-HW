import math

# function to get the 3 significant digits with rounding
def get3(double):
    formatted_x = format(double, '.3g')
    return float(formatted_x)

# function to get the 3 significant digits with chopping
def chop(num, n=3):
    if num == 0:
        return 0
    sign = -1 if num < 0 else 1
    num = abs(num)
    exponent = int(math.floor(math.log10(num)))
    if exponent < n - 1:
        return sign * math.floor(num * 10**(n - exponent - 1)) / 10**(n - exponent - 1)
    else:
        return sign * math.floor(num / 10**(exponent - n + 1)) * 10**(exponent - n + 1)

# initial array, row 1 is put to the last row in (2) and (3)
array = [ [2.51, 1.48, 4.53, 0.05],[1.48, 0.93, -1.3, 1.03] ,[2.68, 3.04, -1.48, -0.53]]
for i in range(2):
    for j in range(i+1, 3):
        mul = array[j][i]
        for k in range(4):
            array[j][k] = get3(array[j][k] - get3(get3(mul * array[i][k]) / array[i][i]))
        print (array[0])            # print process
        print(array[1])
        print(array[2])
        print("|")
        print("V")
array[1][0] = 0
array[2][0] = 0
array[2][1] = 0
print (array[0])
print(array[1])
print(array[2])
print("|")
print("V")

for i in range(2, -1, -1):
    for j in range(i-1, -1, -1):
        mul = array[j][i]
        for k in range(4):
            array[j][k] = get3(array[j][k] - get3(get3(mul * array[i][k]) / array[i][i]))
        print (array[0])
        print(array[1])
        print(array[2])
        print("|")
        print("V")
# now : row-echlon form
x = [0, 0, 0]
for i in range(3):
    x[i] = array[i][3] / array[i][i]
print("x = ",end="")
print(x)
y = [0,0,0]
array = [[2.51, 1.48, 4.53, 0.05], [1.48, 0.93, -1.3, 1.03],[2.68, 3.04, -1.48, -0.53] ]
for i in range(3):
    for j in range(3):
        y[i] += array[i][j]*x[j]
    # y now = Ax', so we minus y with the initial result   
    y[i] = y[i] - array[i][3]
print("y = ",end="")
print(y)

acr = (y[0]**2+y[1]**2+y[2]**2)**(1/2)

print("accuracy : ",end="")
print (acr)
