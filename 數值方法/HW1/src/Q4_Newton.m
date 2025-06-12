
x = input("x:"); y = input("y:");z = input("z:");
[x,y,z] = newton3(x,y,z);
disp("ans: ")
disp([x y z])
disp("error")
disp([x-3*y-z^2+3  2*x^3+y-z^2*5+2 4*x^2+y+z-7])

function [x, y, z] = newton3(x, y, z)
    for N = 1:100000
        if abs(x)<0.001 && abs(y)<0.001 && abs(z)<0.001
            break;
        end
        D = [1,-3,-2*z;6*x^2,1,-10*z;8*x,1,1];
        f1 = x-3*y-z^2+3 ;
        f2 = 2*x^3+y-z^2*5+2 ;
        f3 = 4*x^2+y+z-7;
        s = [x y z]' ;
        s = s - D\[f1 f2 f3]' ;
        x = s(1);
        y = s(2);
        z = s(3);
        
    end
end