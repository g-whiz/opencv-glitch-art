#include <atomic>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

typedef cv::Point3_<uint8_t> Pixel;

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


    std::array<std::atomic_int, 256> histogram, indices;
    for (int i = 0; i < 256; i++) {
        histogram[i].store(0);
    }

    image.forEach<Pixel>([&histogram](Pixel &p, const int *position) -> void {
        histogram[p.x]++;
    });

    int idx = 0;
    for (int i = 0; i < 256; i++) {
        int nextIdx = idx + histogram[i].load();
        indices[i].store(idx);
        idx = nextIdx;
    }

    Mat sortedImage;
    sortedImage = image.clone();

    image.forEach<Pixel>([&indices, &image, &sortedImage](Pixel &p, const int *position) -> void {
        int position2D[2];
        //position of pixel in the 1D array of the sorted image's pixels
        int position1D = indices[p.x].fetch_add(1);

        position2D[0] = position1D / image.cols;
        position2D[1] = position1D % image.cols;
        sortedImage.at<Pixel>(position2D).y = p.y;
        sortedImage.at<Pixel>(position2D).z = p.z;
    });

    namedWindow("Display Image", WINDOW_GUI_EXPANDED );
    imshow("Display Image", sortedImage);
    waitKey(0);
    return 0;
}