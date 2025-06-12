syms x;
p = pade(cos(x)^2, x, 0,'Order',[3 3]);
disp("for cos(x)^2 : ")
disp(p)

disp("for sin(x^4-x) :")
p = pade(sin(x^4-x), x, 0,'Order',[3 3]);
disp(p)

disp("for x*exp(x) :")
p = pade(x*exp(x), x, 0,'Order',[3 3]);
disp(p)
