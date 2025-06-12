format long;
x = input('select start point: ');
y = input('select method: ');
ans1 = fixed(x,y);
disp(ans1);

testx = linspace(-9000,-0.000001,100000);
corr = 0;
for i = 1:100000
    ans1 = fixed(testx(i),y);
    if class(ans1)=="double" 
        if abs(compute(ans1))<0.000001
            corr = corr+1;
        end
    end
end
disp(corr);

function fixedAns = fixed(a,b)
    format long;
    iteration = 0;
    arrayForLoop = ones(100);
    loop = 0;
    for i=1:100
        arrayForLoop(i)=inf;
    end
    while abs(compute(a))>0.000001
        a = g(a,b);
        if iteration>10000              % set limit of iterations
            fixedAns = 'diverge(too many iteration)';
            break;
        elseif a==inf
            fixedAns = "diverge(inf)";
            break;
        end
        for i=1:100
            if a==arrayForLoop(i)
                fixedAns = "diverge(loop)";
                loop = 1;
                break;
            end
        end
        arrayForLoop(mod(iteration,100)+1) = a;
        iteration = iteration+1;
        %disp(a);
    end
    if iteration<=1000000 && a~=inf && ~loop
        fixedAns = a;
    end
end

function computeAns = compute(a)          % compute f(x)
    format long;
    computeAns = power(a,3)-4;
end
function gAns = g(a,b)                    % 3 different g(x)
    format long;
    if b == 1
        gAns = (4+2*power(a,3))/power(a,2)-2*a;
    elseif b == 2
        gAns = sqrt(4/a);
    else
        gAns = (16+power(a,3))/(5*power(a,2));
    end
end