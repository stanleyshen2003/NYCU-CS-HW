format long;
s = (0.231-0.12)/0.12;
ans1 = 0.79168 + s*-0.01834 + s*(s-1)*-0.01129/2;
disp("(a)")
disp("   ans for (a) :")
disp(ans1)


disp("(b)")
ans2 = ans1 + s*(s-1)*(s-2)*0.00134/6;
disp("   ans for (b) :")
disp(ans2)


disp("(c)")
disp("   En(x) = value of next term that would be added to Pn(x)")
disp("   Error for (a) :")
disp(s*(s-1)*(s-2)*0.00134/6)
disp("   Error for (b) :")
disp(s*(s-1)*(s-2)*(s-3)*0.00038/24)


disp("(d)")
s24 = (0.42-0.24)/0.12;
error24 = s*(s-1)*(s-2)*0.00172/6;
disp("   Error when x0 = 0.24")
disp(error24)

s36 = (0.42-0.36)/0.12;
error36 = s*(s-1)*(s-2)*0.002/6;
disp("   Error when x0 = 0.36")
disp(error36)

disp("   It is better to start with x0 = 0.24 because it will have smaller error if getting f(0.42) quadratically")