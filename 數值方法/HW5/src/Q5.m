t = -0.25:0.25:1.25;
A = zeros(7,7);

A(1,1:7) = [-2 1 2 0 -2 -1 2];
A(2,2) = 1;
b = zeros(6,1);
b(1) = 3;
b(2) = 5/2;
for i = 3:7
    A(i,i-2) = 1+t(i-1)/8;
    A(i,i-1) = -2+t(i-1)/16;
    A(i,i) = 1-t(i-1)/8;
    b(i) = t(i-1)^3 /16;
end
x = A\b;
fprintf("    subinterval h = 0.25\n")
fprintf("    x_values for t = 0 to t = 1 having h = 0.25 :\n")
disp(x(2:6))