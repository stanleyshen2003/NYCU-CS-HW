format long;
syms x;
f = x*exp(-x);
fd = diff(f);
matrix = [2 -2 1 1;-3 3 -2 -1 ;0  0 1 0; 1 0 0 0];                                            % M

u1 = linspace(0,1,101);
p1 = matrix*[1 vpa(subs(f,x,1)); 2 vpa(subs(f,x,2));1 vpa(subs(fd,x,1));1 vpa(subs(fd,x,2))]; % MP
x1 = u1.^3*p1(1,1)+u1.^2*p1(2,1)+u1*p1(3,1)+p1(4,1);                                          % u'MP(1) = x(u)
y1 = u1.^3*p1(1,2)+u1.^2*p1(2,2)+u1*p1(3,2)+p1(4,2);                                          % u'MP(2) = y(u)

p2 = matrix*[2 vpa(subs(f,x,2)); 3 vpa(subs(f,x,3));1 vpa(subs(fd,x,2));1 vpa(subs(fd,x,3)) ]; 
x2 = u1.^3*p2(1,1)+u1.^2*p2(2,1)+u1*p2(3,1)+p2(4,1);
y2 = u1.^3*p2(1,2)+u1.^2*p2(2,2)+u1*p2(3,2)+p2(4,2);

plot(x1,y1,x2,y2)

fprintf("when x = %f\ny = %f\n", x1(51),y1(51));