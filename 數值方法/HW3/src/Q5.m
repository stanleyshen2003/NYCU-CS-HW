format long;
A = [1 0.4 0.7;1 1.2 2.1; 1 3.4 4 ;1 4.1 4.9; 1 5.7 6.3 ;1 7.2 8.1 ;1 9.3 8.9];
At = A';
AtA = At*A;
b = [0.031 0.933 3.058 3.349 4.87 5.757 8.921]';
Atb = At*b;
a = AtA\Atb;
disp("(a)")
disp("   A'Aa = A'b, where")
disp("   A = ")
disp(A)
disp("   B = ")
disp(b)
fprintf("(b)\n")
fprintf("   z = %d * x + %d * y + %d\n", a(2),a(3), a(1))

fprintf("(c)\n")

fprintf("   sum of square = %d\n",sum((b-A*a).*(b-A*a)))