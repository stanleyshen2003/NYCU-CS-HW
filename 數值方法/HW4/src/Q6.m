disp(compute_area(-0.2, 1.4, 0.4, 2.6))


function I = compute_area(xleft, xright, yleft, yright)
    t = [-0.77459667 0 0.77459667];
    w = [0.55555555 0.88888889 0.55555555];
    newy = ((yright-yleft)*t+yright+yleft)/2;
    newx = ((xright-xleft)*t+xright+xleft)/2;
    factor = (xright-xleft)*(yright-yleft)/4;
    I = 0;
    for i = 1:3
        for j = 1:3
            I = I + w(i)*w(j)*f(newx(i), newy(j));
        end
    end
    I = I * factor;
end

function z = f(x,y)
    z = exp(x)*sin(2*y);
end

