A = [1 0 0 0 0;1 -2+(pi/4)^2/4 1 0 0; 0 1 -2+(pi/4)^2/4 1 0 ;0 0 1 -2+(pi/4)^2/4 1; 0 0 0 0 1];
b = [0 0 0 0 2]';
x = A\b;
fprintf("(a)\n   y(theta) = \n")
disp(x)

syms theta;
y = 2*sin(theta/2);
theta_values = [pi/4, pi/2, 3*pi/4];
ans1 = double(subs(y, theta, theta_values));
fprintf("                        real value      estimated value       error\n")
for i = 1:3
    fprintf("   y(%.1d * pi / 4) :      %.8f        %.8f       %.8f%%\n", i, ans1(i), x(i+1), abs(ans1(i)-x(i+1))/ans1(i))
end


fprintf("\n(b)\n")
% choose h = pi/30
h = pi/30;                                         
A_new = zeros(31,31);
for i = 2:30
    A_new(i,i-1) = 1;
    A_new(i,i) = -2+h^2/4;
    A_new(i,i+1) = 1;
end
A_new(31,31) = 1;
A_new(1,1) = 1;
b_new = zeros(31,1);
b_new(31) = 2;

% calculate Ax = b
x_new = A_new\b_new;                                

% substitute each theta into y'
theta_values_new = zeros(29,1);                     
for i=1:29
    theta_values_new(i) = i*pi/30;                  
end
ans_new = double(subs(y, theta, theta_values_new));

fprintf("                        real value      estimated value       error\n")
for i = 1:29
    fprintf("   y(pi * %2d / 30) :    %.8f        %.8f       %.8f%%\n", i, ans_new(i), x_new(i+1), abs(ans_new(i)-x_new(i+1))/ans_new(i))
end
% find all the errors
errors = abs(ans_new - x_new(2:30,1))./ans_new;    

% display the maximum 
fprintf("   maximum error = %.8f\n", max(errors))





fprintf("(c)\n")
fprintf("   use secant method with runge-kutta method with specific h\n")
fprintf("   using h = pi/2 can reduce the error small enough\n\n")
h = pi/2;                   % the h (can be changed directly)
U3 = secant(@shooting, [0, 0.7], [0, 1.2], 1e-5, h);
[dummy, x, Ux1] = shooting([0, 0.7], h);
[dummy, x, Ux2] = shooting([0, 1.2],h);
[dummy, x, Ux3] = shooting([0, U3(2)], h);
plot(x, Ux1(:,1), x, Ux2(:,1), x, Ux3(:,1));

theta_values_new = zeros(pi/h-1,1);
for i=1:pi/h-1
    theta_values_new(i) = i*h;
end
ans_new = double(subs(y, theta, theta_values_new));

errors = abs(ans_new - Ux3(2:pi/h, 1))./ans_new; 

fprintf("                        real value      estimated value\n")
for i = 1:pi/h - 1
    fprintf("   y(pi * %d / 2) :      %.8f        %.8f\n", i, ans_new(i), Ux3(i+1))
end
fprintf("   maximum error = %.8f\n", max(errors))

function [P, x, U] = shooting(U0, h)
    [x, U] = rk(h, U0(2));
    P = U(length(x),1) - 2;
end

function x2 = secant(f, x0, x1, tol, h)
    if abs(f(x0, h)) < abs(f(x1, h))
        tmp = x0;
        x0 = x1;
        x1 = tmp;
    end
    x2 = x1 - f(x1, h)*(x0-x1)/(f(x0, h)-f(x1, h));
    while abs(f(x2, h)) > tol
        x0 = x1;
        x1 = x2;
        x2 = x1 - f(x1, h)*(x0-x1)/(f(x0, h)-f(x1, h));
    end
end

function [x, ans1] = rk(h, first)
    dy_dx = @(x, y, z) z;
    dz_dx = @(x, y, z) -y/4;
    
    % Define the step size and the number of steps
    num_steps = pi/h;
    
    % Initialize arrays to store the x, y, and z values
    x = zeros(num_steps+1, 1);
    y = zeros(num_steps+1, 1);
    z = zeros(num_steps+1, 1);
    
    % Set the initial values
    x(1) = 0;
    y(1) = 0;
    z(1) = first;
    
    % Runge-Kutta method
    for i = 1:num_steps
        k1y = h * dy_dx(x(i), y(i), z(i));
        k1z = h * dz_dx(x(i), y(i), z(i));
        
        k2y = h * dy_dx(x(i) + h/2, y(i) + k1y/2, z(i) + k1z/2);
        k2z = h * dz_dx(x(i) + h/2, y(i) + k1y/2, z(i) + k1z/2);
        
        k3y = h * dy_dx(x(i) + h/2, y(i) + k2y/2, z(i) + k2z/2);
        k3z = h * dz_dx(x(i) + h/2, y(i) + k2y/2, z(i) + k2z/2);
        
        k4y = h * dy_dx(x(i) + h, y(i) + k3y, z(i) + k3z);
        k4z = h * dz_dx(x(i) + h, y(i) + k3y, z(i) + k3z);
        
        x(i+1) = x(i) + h;
        y(i+1) = y(i) + (k1y + 2*k2y + 2*k3y + k4y)/6;
        z(i+1) = z(i) + (k1z + 2*k2z + 2*k3z + k4z)/6;
    end

    ans1 = [y z];

end

