format long;
h = 1/2;                                        % x(n+1) - x(n)
y = [0 0 1 0 0]';                               % corresponding y value on x
Y = 6*[0 2 -4 2 0 ]';                           % Y
H = [2*h h 0 0 0;                               % H
    h 4*h h 0 0 ;
    0 h 4*h h 0 ;
    0 0 h 4*h h ;
    0 0 0 h 2*h];
S = H\Y;                                        % HS = Y
disp("S :")
disp(S)
a = 1/6/h.*[S(2)-S(1) S(3)-S(2) S(4)-S(3) S(5)-S(4)]';
b = 1/2.*S(1:4);
d = [0  0 1 0]';
c = (1/h).*(y(2:5)-y(1:4)) - (1/6).*(2*h.*S(1:4)+h.*S(2:5));

abcd = [a b c d];
disp("[a b c d] :")
disp(abcd)
temp = 0;
x0 = linspace(-1+temp*h, -1+temp*h+h, 101);      % segment x0 to x1
y0 = polyval(abcd(temp+1,1:4),x0+1);             % plug in y = a(x-x0)^3 + b(x-x0)^2 + c(x-x0) +d
temp = temp+1;

x1 = linspace(-1+temp*h, -1+temp*h+h, 101);
y1 = polyval(abcd(temp+1,1:4),x1+1/2);
temp = temp+1;

x2 = linspace(-1+temp*h, -1+temp*h+h, 101);
y2 = polyval(abcd(temp+1,1:4),x2);
temp = temp+1;
x3 = linspace(-1+temp*h, -1+temp*h+h, 101);
y3 = polyval(abcd(temp+1,1:4),x3-1/2);
temp = temp+1;
plot(x0,y0, x2,y2, x1, y1, x3,y3)  % plot