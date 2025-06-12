found = 0;
times = 3333;
arrayx = zeros(50);
arrayy = zeros(50);
arrayz = zeros(50);
arrayex = zeros(50);
arrayey = zeros(50);
arrayez = zeros(50);
A = -100000000 + 2000000000.*rand(3*times,1);
for n=1:times-1
    format long
    x = A(3*n); y = A(3*n+1) ; z = A(3*n+2);
    [x,y,z] = newton(x,y,z);
    isin = false;
    for j = 1:found 
        if abs(arrayx(j)-x)<0.1 || isnan(x)
            isin = true;
            break;
        end
    end
    if ~isin && abs(x-3*y-z^2+3)<0.1
        disp("ans: ")
        disp([x,y,z])
        disp("error: ")
        ex =x-3*y-z^2+3;
        ey =2*x^3+y-z^2*5+2;
        ez=4*x^2+y+z-7;
        found = found+1;
        arrayx(found) = x;
        arrayy(found) = y;
        arrayz(found) = z;
        arrayex(found) = ex;
        arrayey(found) = ey;
        arrayez(found) = ez;
    end
   
end
disp(found);
for N=1:found
    disp("ans: ")
    disp([arrayx(N),arrayy(N),arrayz(N)]);
    disp("error: ")
    disp([arrayex(N),arrayey(N),arrayez(N)]);
end

function [x,y,z] = newton(x,y,z)
    format long;
    loop = 0;
    for N = 1:100
        D = [1,-3,-2*z;6*x^2,1,-10*z;8*x,1,1];  % jacobean
        f1 = x-3*y-z^2+3 ;                      % compute f(x)
        f2 = 2*x^3+y-z^2*5+2 ;
        f3 = 4*x^2+y+z-7;
        s = [x y z]' ;
        s = s - D\[f1 f2 f3]' ;                 % use inverse an solve
        x = s(1);
        y = s(2);
        z = s(3);
        loop = loop + 1;
        if(loop>1000000)
            break;
        end
    end
end




function [x ,y ,z] = fixed(x,y,z)
    format long;
    for N = 1:10000
        x = -3+3*y+z^2;
        y = -2-2*x^3+5*z^2;
        z = 7-y-4*x^2;
        if abs(x)<0.1 && abs(y)<0.1 && abs(z)<0.1
            break;
        end
    end
    
end

%compute error
function output = compute(a)
    format long;
    output = true;
    error = 0.1;
    if abs(a(1)-a(2)*3-power(a(3),2)+3)>error
        output = false;
    elseif abs(power(a(1),3)*2+a(2)-5*power(a(3),2)+2)>error
        output = false;
    elseif abs(4*power(a(1),4)+a(2)+a(3)-7)>error
        output = false;
    end
end

%compute f(a)
function foutput = f(a)
    foutput = [a(1)-3*a(2)-power(a(3),2)+3,2*power(a(1),3)+a(2)-5*power(a(3),2)+2,4*power(a(1),2)+a(2)+a(3)-7];
end

%for iterating