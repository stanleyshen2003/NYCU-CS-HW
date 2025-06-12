__kernel void convolution(
    __global float* inputImage,
    __global float* filter,
    __global float* outputImage,
    int filterWidth)
{
    int i = get_global_id(0); // Row index
    int j = get_global_id(1); // Column index
    int imageHeight = get_global_size(1);
    int imageWidth = get_global_size(0);
    int halffilterSize = filterWidth / 2;
    float sum = 0;

    int row_start, row_end, col_start, col_end;
    row_start = max(0, (halffilterSize - j));
    col_start = max(0, (halffilterSize - i));
    row_end =  max(0, j + halffilterSize - imageHeight - 1);
    col_end = max(0, i + halffilterSize - imageWidth - 1);

    // from serial implemetnation
    int row_base = (j - halffilterSize + row_start) * imageWidth + i;
    int filter_base = row_start * filterWidth + halffilterSize;
    for (int k = -halffilterSize + row_start; k <= halffilterSize - row_end; k++) {
        for (int l = -halffilterSize + col_start; l <= halffilterSize - col_end; l++) {

            int filter_index = filter_base + l;

            if (filter[filter_index])
                sum = mad(inputImage[row_base + l], filter[filter_index], sum);
        }
        row_base += imageWidth;
        filter_base += filterWidth;
    }
    outputImage[j * imageWidth + i] = sum;
}