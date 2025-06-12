format long
[answer, minh] = adaptive_trapezoidal(0.2,1);
minh = minh/2;
fprintf("   when terminate, h = %d\n", minh)

function [I, minh] = adaptive_trapezoidal(a, b)
    fa = f(a);
    fb = f(b);
    I = (b-a)/2 * (fa + fb);
    minh = b-a;
    intervalLeft = (b-a)/4 * (fa + f((a+b)/2));                 % integrate h/2 with left
    intervalRight = (b-a)/4 * (f((a+b)/2) + fb);                % integrate h/2 with right
    if abs(I - (intervalLeft + intervalRight)) > 0.02        % if "next" iteration have error more than 0.02
        [integralLeft, hleft] = adaptive_trapezoidal(a, (a+b)/2);
        [integralRight, hright] = adaptive_trapezoidal((a+b)/2, b);
        I = integralRight + integralLeft;
        minh = min(hleft, hright);
    end
end

function y = f(x)
    y = 1/x^2;
end