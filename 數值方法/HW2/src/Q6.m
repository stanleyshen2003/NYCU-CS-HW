array = [4 -1 0 0 0 0 100;-1 4 -1 0 0 0 200;0 -1 4 -1 0 0 200;0 0 -1 4 -1 0 200;0 0 0 -1 4 -1 200;0 0 0 0 -1 4 100];

w = 3;
amount = (w-1)/2+1;
%  you can input the array and w(band width) on your own

for i = 1:size(array,1)-1                       
    for j = i+1:i+amount-1
        for k = i+1:i+amount-1
            array(j,k) = array(j,k)-array(j,i)*array(i,k)/array(i,i);
        end
        array(j,size(array,2)) = array(j,size(array,2)) - array(j,i)*array(i,size(array,2))/array(i,i);
        array(j,i) = 0;
    end
end
% display now result
disp(array)
for i = size(array,1):-1:2
    for j = i-amount+1:i-1
        for k = i-amount+1:i-1
            array(j,k) = array(j,k)-array(j,i)*array(i,k)/array(i,i);
        end
        array(j,size(array,2)) = array(j,size(array,2)) - array(j,i)*array(i,size(array,2))/array(i,i);
        array(j,i) = 0;
    end
end
% display now result
disp(array)
x = zeros(size(array,1),1);
for i = 1:size(array,1)
    x(i,1) = array(i,size(array,2))/array(i,i);
end
% display answer
disp(x);