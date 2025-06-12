format long
firstAns = bisection(-2,0);
secondAns = bisection(0,2);
disp('bisection : between -2 & 0');
disp(firstAns);
disp('bisection : between 0 & 2');
disp(secondAns);

firstAns = secant(-2,0);
secondAns = secant(0,2);
disp('secant : between -2 & 0');
disp(firstAns);
disp('secant : between 0 & 2');
disp(secondAns);

firstAns = newton(-1);
secondAns = newton(1);
disp("Newton's : between -2 & 0, starting from -1");
disp(firstAns);
disp("Newton's : between 0 & 2, starting from 1");
disp(secondAns);

function bisectionAns = bisection(a,b)
    format long;
    mid = b;
    while abs(compute(mid))>0.00001                         %set accuracy   
        mid = (a+b)/2;
        if compute(mid)*compute(b)>0
            b = mid;
        else 
            a = mid;
        end
    end
    bisectionAns = b;
end

function secantAns = secant(a,b)
    format long;
    next = b;
    while abs(compute(next))>0.00001
        next = b-compute(b)*(b-a)/(compute(b)-compute(a));  % "next" is the intersection of the line with point a & b and y=0 
        if compute(next)*compute(b)>0                       % let the true solution always in [a,b]
            b = next;
        else 
            a = next;
        end
    end
    secantAns = next;
end

function newtonAns = newton(a)
    format long;
    next = a;
    while abs(compute(next))>0.00001
        next = next - compute(next)/computeD(next);         % x(n+1) = x(n) - f(xn)/f'(xn)
    end
    newtonAns = next;
end

function computeAns = compute(m)                            % compute f(x)
    computeAns = power(m,2) + sin(m) - exp(m)/4 - 1;
end

function computeDAns = computeD(input)                      % compute f'(x)
    computeDAns = 2*input + cos(input) + exp(input)/4;
end
