table = zeros(7,7);
x = 0.3;
h = 0.2;
x_all = [x x+h x+2*h x+3*h x+4*h x+5*h x+6*h];
table(1:7,1) = f(x_all);
sign = 1;
for i = 2:7
    for j = 1:7-i+1
        sign = 1;
        for k = j+i-1:-1:j
            table(j,i) = table(j,i) + table(k,1) * nchoosek(i-1,k-j) * sign;
            sign = sign * -1;
        end
    end

end

disp(table)
disp(" (a)")
fprintf("   estimate using x1, x2, x3, x4\n")
fprintf("   s = (x - x0)/h = 2.1\n")
s = (0.72-0.5)/0.2;
fprintf("   f(x) = %d + s*%d + s*(s-1)*%d / 2 + s*(s-1)*(s-2)*%d / 6\n", table(2,1), table(2,2), table(2,3), table(2,4))
fprintf("   f'(x) = (%d + %d * (2*s - 1) / 2 + %d * (3*s^2 - 6*s + 2) / 6) / h\n", table(2,2), table(2,3), table(2,4))
fprintf("   f'(0.72) = %d\n\n", (table(2,2) + table(2,3) * (2*s - 1)/2 + table(2,4) * (3*s^2 - 6*s + 2)/6)/h)

disp(" (b)")
fprintf("   estimate using x4, x5, x6\n")
fprintf("   s = (x - x4)/h = 1.15\n")
s = (1.33-1.1)/0.2;
fprintf("   f(x) = %d + s*%d + s*(s-1)*%d / 2\n", table(5,1), table(5,2), table(5,3))
fprintf("   f'(x) = (%d + %d * (2*s - 1) / 2) / h\n", table(5,2), table(5,3))
fprintf("   f'(1.33) = %d\n\n", (table(5,2) + table(5,3) * (2*s - 1)/2)/h)

disp(" (c)")
fprintf("   estimate using x0, x1, x2, x3, x4\n")
fprintf("   s = (x - x0)/h = 1\n")
s = (0.5-0.3)/0.2;
fprintf("   f(x) = %d + s*%d + s*(s-1)*%d / 2 + s*(s-1)*(s-2)*%d / 6 + s*(s-1)*(s-2)*(s-3)*%d / 24\n", table(1,1), table(1,2), table(1,3), table(1,4), table(1,5))
fprintf("   f'(x) = (%d + %d * (2*s - 1) / 2 + %d * (3*s^2 - 6*s + 2) / 6 + %d * (4*s^3 - 18*s^2 + 22*s -6)/24) / h\n", table(1,2), table(1,3), table(1,4), table(1,5))
fprintf("   f'(0.5) = %d\n\n", (table(1,2) + table(1,3) * (2*s - 1)/2 + table(1,4) * (3*s^2 - 6*s + 2)/6 + table(1,5) * (4*s^3 - 18*s^2 + 22*s -6)/24)/h)



function y = f(x)
    y = x + sin(x) /3;
end