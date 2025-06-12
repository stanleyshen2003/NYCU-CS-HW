__kernel void convolution(
    __global float* inputImage,
    __global float* filter,
    __global float* outputImage,
    int filterWidth,
    int imageHeight,
    int imageWidth)
{
    int i = get_global_id(0); // Row index
    int j = get_global_id(1); // Column index

    int halffilterSize = filterWidth / 2;
    float sum = 0;

    // from serial implemetnation
    for (int k = -halffilterSize; k <= halffilterSize; k++) {
        for (int l = -halffilterSize; l <= halffilterSize; l++) {
            int row = i + k;
            int col = j + l;

            // Check boundaries
            if (row >= 0 && row < imageHeight && col >= 0 && col < imageWidth) {
                sum += inputImage[row * imageWidth + col] * filter[(k + halffilterSize) * filterWidth + (l + halffilterSize)];
            }
        }
    }
    outputImage[i * imageWidth + j] = sum;
}