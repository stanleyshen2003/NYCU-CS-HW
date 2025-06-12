format long

y0 = [0 1 0];
tspan = [0 0.2 0.4 0.6 0.8 1];

% solve the equation by sending the correct input into ode45()
[t, y] = ode45(@f, tspan, y0);                      

fprintf("(a)\n")
for i=1:5
    y_val = interp1(t, y, i*0.2);
    fprintf("    t = %.1f : y(%.1f) = %f, y'(%.1f) = %f, y''(%.1f) = %f\n",i*0.2,i*0.2,y_val(1), i*0.2, y_val(2), i*0.2, y_val(3) )
end

adams = zeros(6,3);
adams(1:4, 1:3) = y(1:4,1:3);

y_three = zeros(6,1);
for i = 1:4
    y_three(i) = t(i) + 2 * y(i,1) - t(i) * y(i, 2);
end

% for t = 0.8
% predictor
temp1 = zeros(1,3);
temp1 = adams(4,1:3) + 0.2/24 * (55*[adams(4,2) adams(4,3), y_three(4)] - 59*[adams(3,2) adams(3,3) y_three(3)] + 37*[adams(2,2) adams(2,3) y_three(2)] ...
    - 9*[adams(1,2) adams(1,3) y_three(1)]);
y_three(5) = t(5) + 2 * temp1(1,1) - t(5) * temp1(1,2);

% corrector
adams(5,1:3) = adams(4,1:3) + 0.2/24 * (9*[temp1(1,2) temp1(1,3) y_three(5)] + 19*[adams(4,2) adams(4,3), y_three(4)] - 5*[adams(3,2) adams(3,3) y_three(3)] + ...
    [adams(2,2) adams(2,3) y_three(2)]);
y_three(5) = t(5) + 2 * adams(5,1) - t(5) * adams(5,2);

% for t = 1
% predictor
temp1 = adams(5,1:3) + 0.2/24 * (55*[adams(5,2) adams(5,3), y_three(5)] - 59*[adams(4,2) adams(4,3) y_three(4)] + 37*[adams(3,2) adams(3,3) y_three(3)] ...
    - 9*[adams(2,2) adams(2,3) y_three(2)]);
y_three(6) = t(6) + 2 * temp1(1,1) - t(6) * temp1(1,2);
% corrector
adams(6,1:3) = adams(5,1:3) + 0.2/24 * (9*[temp1(1,2) temp1(1,3) y_three(6)]+ 19*[adams(5,2) adams(5,3), y_three(5)] - 5*[adams(4,2) adams(4,3) y_three(4)] + ...
    [adams(3,2) adams(3,3) y_three(3)]);
y_three(6) = t(6) + 2 * adams(6,1) - t(5) * adams(6,2);

fprintf("(b)\n")
fprintf("    when t = 1 and using Adams-Moulton method, y = %.10f\n", adams(6,1))

fprintf("(c)\n")
fprintf("    from the ode45() in (a), we know that when t = 1, y = %.10f, therefore the error is %.10f\n", y_val(1), abs(y_val(1)-adams(6,1)))
function dx = f(t,x)
    dx = [x(2) x(3) t+2*x(1)-t*x(2)]';
end