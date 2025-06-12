format long
table = zeros(6,6);
x = [0.15 0.21 0.23 0.27 0.32 0.35];
table(1:6,1) = F(x);

for q = 1:5
    for i = 1:6-q
        table(i,q+1) = (table(i+1,q) - table(i,q))/(x(i+q)-x(i));
    end

end

disp(table)
disp("row : xi, for i = 0 ~ 5")
disp("column : f[i j], for j = i ~ i+5\n")

for i = 1:4
    fprintf("from x%d:\n",i-1)
    fprintf("polynomial: %d + %d * (x-x%d) + %d * (x-x%d)(x-x%d)\n", table(i,1),table(i,2),i-1, table(i,3), i-1, i)
    fprintf("f'(x) = %d + %d * (x - x%d + x - x%d)\n", table(i,2), table(i,3), i-1, i)
    fprintf("f'(x) = %d\n\n", table(i,2) + table(i,3) * (0.268 - x(i) + 0.268 - x(i+1)))
end

fprintf("real answer: %d\n", F_pron(0.268))
fprintf("fitting with x2, x3, x4 will have the smallest error\n")
fprintf("this is intuitive because the sum of distance from x2, x3, x4 to 0.268 is the smallest\n")
function y = F(s)
    y = 1 + log10(s);
end

function y = F_pron(s)
    y = log10(exp(1))/s;
end