import matplotlib.pyplot as plt
with open('record.txt') as f:
    lines = f.readlines()
    print(lines)
    line = lines[0]
    a = line.split(',')
    a.pop()
    a = [float(i) for i in a]
    plt.plot(a)

    # Add labels and a title
    plt.xlabel('Index')
    plt.ylabel('Integer Value')
    plt.title('Sequential List of Integers')


    # Show the plot
    plt.show()
    f.close()