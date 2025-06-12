format long
table = zeros(9,9);
x1 = [1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8];
table(1:9,1) = [1.543 1.669 1.811 1.971 2.151 2.352 2.577 2.858 3.107]';

for q = 1:8
    for i = 1:9-q
        table(i,q+1) = (table(i+1,q) - table(i,q))/(x1(i+q)-x1(i));
    end
end

f = @(x) table(1,1) + table(1,2)*(x-1) + table(1,3)*(x-1).*(x-1.1) + table(1,4)*(x-1).*(x-1.1).*(x-1.2) ...
+ table(1,5)*(x-1).*(x-1.1).*(x-1.2).*(x-1.3) + table(1,6)*(x-1).*(x-1.1).*(x-1.2).*(x-1.3).*(x-1.4) +  ...
table(1,7).*(x-1).*(x-1.1).*(x-1.2).*(x-1.3).*(x-1.4).*(x-1.5) + ...
table(1,8)*(x-1).*(x-1.1).*(x-1.2).*(x-1.3).*(x-1.4).*(x-1.5).*(x-1.6) + ...
table(1,9)*(x-1).*(x-1.1).*(x-1.2).*(x-1.3).*(x-1.4).*(x-1.5).*(x-1.6).*(x-1.7);

I = integral(f,1,1.8);
fprintf("   Estimated integral using interpolation = %d\n\n", I)

% do 1/3 first
sum = 1/30*(table(1,1)+4*table(2,1)+table(3,1)) + 3/80*(table(3,1)+3*table(4,1)+3*table(5,1)+table(6,1)) + 3/80*(table(6,1)+3*table(7,1)+3*table(8,1)+table(9,1));
fprintf("   using 1/3 first, result = %d\n",sum)
fprintf("   error = %d\n\n", abs(I-sum))

sum = 1/30*(table(4,1)+4*table(5,1)+table(6,1)) + 3/80*(table(1,1)+3*table(2,1)+3*table(3,1)+table(4,1)) + 3/80*(table(6,1)+3*table(7,1)+3*table(8,1)+table(9,1));
fprintf("   using 1/3 in the middle, result = %d\n",sum)
fprintf("   error = %d\n\n", abs(I-sum))

sum = 1/30*(table(7,1)+4*table(8,1)+table(9,1)) + 3/80*(table(1,1)+3*table(2,1)+3*table(3,1)+table(4,1)) + 3/80*(table(4,1)+3*table(5,1)+3*table(6,1)+table(7,1));
fprintf("   using 1/3 at the end, result = %d\n",sum)
fprintf("   error = %d\n\n", abs(I-sum))

fprintf("   using Simpson 3/8 for (x = 1~1.3 and x = 1.3~1.6), and using Simpson 1/3 for (x = 1.6~1.8) will have the smallest error\n")