NK = input()
N, K = NK.split()
N, K = int(N), int(K)

if N > pow(2, K):
    print("You will become a flying monkey!")
else:
    print("Your wish is granted!")