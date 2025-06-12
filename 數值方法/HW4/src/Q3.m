syms h;
syms c1 c2 c3 c4 c5;
syms f1 f2 f3 f4 f5;
A = [1 1 1 1 1; -2*h -h 0 h 2*h; 4*h^2 h^2 0 h^2 4*h^2; -8*h^3 -h^3 0 h^3 8*h^3;16*h^4 h^4 0 h^4 16*h^4];

b = [0 0 2 0 0]';
x = [c1; c2; c3; c4; c5];
x = solve(A*x == b, x);
disp("for f''(x) : ")
disp(x)

b = [0 0 0 6 0]';
x = [c1;c2;c3;c4;c5];
x = solve(A*x == b, x);
disp("for f'''(x) : ")
disp(x)


