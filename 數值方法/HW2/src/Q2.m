
array = [4 -1 100;4 -1 200;4 -1 200;4 -1 200;4 -1 200;4 -1 100];

arr = compute(array);
disp(arr)
function arr = compute(array)
    for i = 2:size(array,1)      
        % only 1 row has to be reduced for each i
        array(i,1) = array(i,1)-array(i,2)*array(i-1,2)/array(i-1,1);
        array(i,3) = array(i,3)-array(i,2)*array(i-1,3)/array(i-1,1);
    end
    for i = size(array,1)-1:-1:1
        % don't need to compute (i,2) because it wouldn't affact the result
        array(i,3) = array(i,3)-array(i+1,3)*array(i,2)/array(i+1,1);
    end
    arr = zeros(size(array,1),1);
    for i = 1:size(array,1)
        arr(i,1) = array(i,3)/array(i,1);
    end
end