format long
ans1 = newton(3);
disp('ans :');
disp(ans1);

function newtonAns = newton(a)
    format long;
    next = a;
    while abs(compute(next))>0.00001
        next = next-compute(next)/computeD(next);
    end
    newtonAns = next;
end

function computeDAns = computeD(x)          % compute f'(x)
    format long;
    computeDAns = power(x-2,2)*(x-4)*(5*x-16);
end

function computeAns = compute(x)            % compute f(x)
    format long;
    computeAns = power(x-2,3)*power(x-4,2);
end
