x0 = [-1.250595 - 0.0489946i, 0.665052 + 0.013409i, 0.08858- 0.50358i]; % initial guess
x = fsolve(@myfun, x0);
disp(x)
function F = myfun(x)
    F = [x(1)-3*x(2)-x(3)^2+3;
         2*x(1)^3 + x(2) - 5*x(3)^2+2;
         4*x(1)^2 + x(2) + x(3)-7];
end
