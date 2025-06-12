A = [4.62 -1.21 3.22; -3.07 5.48 2.11;1.26 3.11 4.57];
b = [2.22;-3.17;5.11];
x=[0 0 0]';
maxerror = 0.00001;
n = size(x,1);
for w = 1:0.1:2                                     % choose w = 1,1.1,1.2,1.3...2
    error = inf;
    itr = 0;
    x=[0 0 0]';
    while error>maxerror
        x_old = x;
        for i=1:n
            sum = 0;
            for j=1:n
                sum = sum+A(i,j)*x(j);              % the sum
            end
            x(i) = x(i)+(w/A(i,i))*(b(i)-sum);      % update x(i)
        end
        itr = itr+1;
        error = norm(x_old-x);
    end
    fprintf("w : %-2f\nIteration : %d\n",w,itr)
end