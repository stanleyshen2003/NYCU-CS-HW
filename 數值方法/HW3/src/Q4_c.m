format long;
u1 = linspace(0,1);
matrix = (1/6)*[-1 3 -3 1;3 -6 3 0;-3 0 3 0; 1 4 1 0];

pts = [10 10;50 15;75 60;90 100;105 140;150 200;180 140;190 120;160 100;130 80];
count = 1;
p1 = matrix*[pts(count,1:2);pts(count+1,1:2);pts(count+2,1:2) ;pts(count+3,1:2)];
x1 = u1.^3*p1(1,1)+u1.^2*p1(2,1)+u1*p1(3,1)+p1(4,1);
y1 = u1.^3*p1(1,2)+u1.^2*p1(2,2)+u1*p1(3,2)+p1(4,2);
count = count +1;

u2 = u1;
p2 = matrix*[pts(count,1:2);pts(count+1,1:2);pts(count+2,1:2) ;pts(count+3,1:2)];
x2 = u2.^3*p2(1,1)+u2.^2*p2(2,1)+u2*p2(3,1)+p2(4,1);
y2 = u2.^3*p2(1,2)+u2.^2*p2(2,2)+u2*p2(3,2)+p2(4,2);
count = count +1;

u3 = linspace(1,2);
p3 = matrix*[pts(count,1:2);pts(count+1,1:2);pts(count+2,1:2) ;pts(count+3,1:2)];
x3 = (u3-1).^3*p3(1,1)+(u3-1).^2*p3(2,1)+(u3-1)*p3(3,1)+p3(4,1);
y3 = (u3-1).^3*p3(1,2)+(u3-1).^2*p3(2,2)+(u3-1)*p3(3,2)+p3(4,2);
count = count +1;

p4 = matrix*[pts(count,1:2);pts(count+1,1:2);pts(count+2,1:2) ;pts(count+3,1:2)];
x4 = (u3-1).^3*p4(1,1)+(u3-1).^2*p4(2,1)+(u3-1)*p4(3,1)+p4(4,1);
y4 = (u3-1).^3*p4(1,2)+(u3-1).^2*p4(2,2)+(u3-1)*p4(3,2)+p4(4,2);
count = count +1;

p5 = matrix*[pts(count,1:2);pts(count+1,1:2);pts(count+2,1:2) ;pts(count+3,1:2)];
x5 = (u3-1).^3*p5(1,1)+(u3-1).^2*p5(2,1)+(u3-1)*p5(3,1)+p5(4,1);
y5 = (u3-1).^3*p5(1,2)+(u3-1).^2*p5(2,2)+(u3-1)*p5(3,2)+p5(4,2);
count = count +1;

p6 = matrix*[pts(count,1:2);pts(count+1,1:2);pts(count+2,1:2) ;pts(count+3,1:2)];
x6 = (u3-1).^3*p6(1,1)+(u3-1).^2*p6(2,1)+(u3-1)*p6(3,1)+p6(4,1);
y6 = (u3-1).^3*p6(1,2)+(u3-1).^2*p6(2,2)+(u3-1)*p6(3,2)+p6(4,2);
count = count +1;

u7 = linspace(2,3);
p7 = matrix*[pts(count,1:2);pts(count+1,1:2);pts(count+2,1:2) ;pts(count+3,1:2)];
x7 = (u7-2).^3*p7(1,1)+(u7-2).^2*p7(2,1)+(u7-2)*p7(3,1)+p7(4,1);
y7 = (u7-2).^3*p7(1,2)+(u7-2).^2*p7(2,2)+(u7-2)*p7(3,2)+p7(4,2);
count = count +1;

plot(x1, y1, x2, y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7)