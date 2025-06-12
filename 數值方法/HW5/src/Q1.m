format long
% for h = 0.1
h1 = 0.1;
y0 = 0;
for i = 1:h1:2-h1
    y0 = y0 + h1 * f(y0,i);
end
disp("using h = 0.1, ans: ")
disp(y0)

% for h = 0.05
h2 = 0.05;
y1 = 0;
for i = 1:h2:2-h2
    y1 = y1 + h2 * f(y1,i);
end
disp("using h = 0.05, ans: ")
disp(y1)

% find the real value 
myf = @(t, y) y^2 + t^2;
y0 = 0;
tspan = [1, 2];
[t, y] = ode45(myf, tspan, y0);
y_val = interp1(t, y, 2);
fprintf("real value: %.8f\n", y_val)
fprintf("error for h = 0.05 is %.8f\n", y_val - y1)

function y = f(y1, t)
    y = y1^2 + t^2;
end