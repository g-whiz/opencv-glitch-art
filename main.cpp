#include <atomic>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

typedef cv::Point3_<uint8_t> Pixel;
const int CHAN = 0;

int main(int argc, char **argv) {
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    Mat image;
    image = imread( argv[1], 1 );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    int chanBins = 256;
    int histSize[] = {chanBins};
    // RGB channels vary between 0 and 255
    float chanRanges[] = { 0, 256 };
    const float* ranges[] = { chanRanges };
    MatND hist;
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0};
    calcHist( &image, 1, channels, Mat(), // do not use mask
              hist, 1, histSize, ranges,
              true, // the histogram is uniform
              false );

    std::array<std::atomic_int, 256> indices;

    int idx = 0;
    for (int i = 0; i < 256; i++) {
        int nextIdx = idx + (int) hist.at<float>(i);
        indices[i].store(idx);
        idx = nextIdx;
    }

    Mat sortedImage;
    sortedImage = image.clone();

    image.forEach<Pixel>([&indices, &image, &sortedImage](Pixel &p, const int *position) -> void {
        int newPosition[2];
        int idx = indices[p.x].fetch_add(1);

        newPosition[0] = idx % image.cols;
        newPosition[1] = idx / image.cols;
        sortedImage.at<Pixel>(newPosition) = p;
    });

    namedWindow("Display Image", WINDOW_GUI_EXPANDED );
    imshow("Display Image", sortedImage);
    waitKey(0);
    return 0;
}