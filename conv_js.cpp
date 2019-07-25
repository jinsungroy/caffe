#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <time.h>
using namespace std;
using namespace cv;

double relu(double x)
{
    return (x > 0) ? x : 0;
}

double saturation(double x)
{
    return (x > 255) ? 255 : (x < 0) ? 0 : x;
}

void convolution(const double *input_feature, const double *weights, const double *bias, double *output_feature, int output_channel, int input_channel,
                 int k_size, int output_w, int output_h, int input_w, int input_h, int channels, int stride)
{
    for (int ch = 0; ch < channels; ch++)
    {
        for (int row = 0; row < output_h; row++)
        {
            for (int col = 0; col < output_w; col++)
            {
                for (int output_filter = 0; output_filter < output_channel; output_filter++)
                {
                    double temp = 0;

                    for (int input_filter = 0; input_filter < input_channel; input_filter++)
                    {
                        for (int kernel_row = 0; kernel_row < k_size; kernel_row++)
                        {
                            for (int kernel_col = 0; kernel_col < k_size; kernel_col++)
                            {
                                temp = temp + input_feature[ch * (input_h * input_w) + (input_filter * input_h * input_w + ((row * stride) + kernel_row) * input_w + ((col * stride) + kernel_col))] * weights[output_filter * input_channel * k_size * k_size + input_filter * k_size * k_size + kernel_row * k_size + kernel_col];
                            }
                        }
                    }
                    output_feature[ch * (output_h * output_w) + (output_filter * output_h * output_w + row * output_w + col)] = saturation(relu(temp + bias[output_filter]));
                }
            }
        }
    }
    return;
}

void maxPooling(const double *input_feature, double *output_feature, int input_channel, int input_w, int input_h, int channels)
{
    int output_w = input_w / 2;
    int output_h = input_h / 2;

    for (int ch = 0; ch < channels; ch++)
    {
        for (int depth = 0; depth < input_channel; depth++)
        {
            for (int row = 0; row < input_h / 2; row++)
            {
                for (int col = 0; col < input_w / 2; col++)
                {
                    double max1, max2, max;
                    double array00, array01, array10, array11;

                    array00 = input_feature[ch * input_h * input_w + depth * input_h * input_w + (2 * row) * input_w + (2 * col)];
                    array01 = input_feature[ch * input_h * input_w + depth * input_h * input_w + (2 * row) * input_w + (2 * col) + 1];
                    array10 = input_feature[ch * input_h * input_w + depth * input_h * input_w + ((2 * row) + 1) * input_w + (2 * col)];
                    array11 = input_feature[ch * input_h * input_w + depth * input_h * input_w + ((2 * row) + 1) * input_w + (2 * col) + 1];
                    max1 = array00 > array01 ? array00 : array01;
                    max2 = array10 > array11 ? array10 : array11;
                    max = max1 > max2 ? max1 : max2;

                    // Write the max value on the output layer
                    output_feature[ch * (output_h * output_w) + depth * output_h * output_w + row * output_w + col] = max;
                }
            }
        }
    }
}

void padding(double *input_image, double *output_images, int channel, int input_w, int input_h, int padding_size)
{
    int output_w = input_w + padding_size * 2;
    int output_h = input_h + padding_size * 2;

    for (int ch = 0; ch < channel; ch++)
    {
        for (int row = 0; row < output_h; row++)
        {
            for (int col = 0; col < output_w; col++)
            {
                if (row < padding_size || col < padding_size || row > (output_h - padding_size-2) || col > (output_w - padding_size-2))
                {
                    output_images[ch * output_h * output_w + row * output_w + col] = 0;
                }
                else
                {
                    output_images[ch * output_h * output_w + row * output_w + col] = input_image[ch * input_h * input_w + (row - padding_size + 1) * input_w + (col - padding_size + 1)];
                }
            }
        }
    }
}

void mat2Im(Mat src, double *dst)
{
    for (int ch = 0; ch < src.channels(); ch++)
    {
        for (int row = 0; row < src.rows; row++)
        {
            for (int col = 0; col < src.cols; col++)
            {
                dst[ch * src.rows * src.cols + row * src.cols + col] = double(src.at<Vec3b>(row, col)[ch]);
            }
        }
    }
}

void im2Mat(double *src, Mat dst, int channels, int rows, int cols)
{
    for (int ch = 0; ch < channels; ch++)
    {
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                dst.at<Vec3b>(row, col)[ch] = src[ch * rows * cols + row * cols + col];
            }
        }
    }
}

int main()
{
    /*              Define Parameters                */
    clock_t start_point, end_point;
    int padding_size = 5;
    int conv_kSize = 3;
    int conv_stride = 1;
    int pool_size = 2;
    //double weight[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1}; // Edge detection
    double weight[9] = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625}; // Smoothing
    double bias[1] = {0};
    int number_of_images = 1;
    /*              Define Arrays, Mats               */
    double *I;
    double *I_padded;
    double *I_conv1;
    double *I_pool1;
    Mat Mat_image;
    Mat Mat_padded;
    Mat Mat_conv1;
    Mat Mat_pool1;
    Mat Mat_conv_CV;
    Mat weight_cv = (Mat_<double>(3,3) << 0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625);
    
    /*              Read Image              */
    Mat_image = imread("lena512.tiff", IMREAD_COLOR);
    if (Mat_image.empty())
    {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    /*              Test Matrix Size Error              */
    int image_channel = Mat_image.channels();
    int image_w = Mat_image.cols;
    int image_h = Mat_image.rows;
    int pad_w = image_w + (padding_size * 2);
    int pad_h = image_h + (padding_size * 2);
    // int conv1_w = (pad_w - conv_kSize) / conv_stride + 1;
    // int conv1_h = (pad_h - conv_kSize) / conv_stride + 1;
    int conv1_w = (image_w - conv_kSize) / conv_stride + 1;
    int conv1_h = (image_h - conv_kSize) / conv_stride + 1;
    int pool1_w = conv1_w / pool_size;
    int pool1_h = conv1_h / pool_size;

    printf("%d %d %d %d %d %d %d %d %d\n", image_channel, image_w, image_h, pad_w, pad_h, conv1_w, conv1_h, pool1_w, pool1_h);

    if ((pad_w - conv_kSize + 1) % conv_stride != 0 || (pad_h - conv_kSize +1) % conv_stride != 0)
    {
        cout << "Conv size error" << endl;
        return 0;
    }
    // if (conv1_w % pool_size == 1 || conv1_h % pool_size == 1)
    // {
    //     cout << "Pool size error" << endl;
    //     return 0;
    // }

    /*              Memory Allocation               */
    Mat_padded = Mat(pad_h, pad_w, CV_8UC3);
    Mat_conv1 = Mat(conv1_h, conv1_w, CV_8UC3);
    Mat_conv_CV = Mat(conv1_h, conv1_w, CV_8UC3);
    Mat_pool1 = Mat(pool1_h, pool1_w, CV_8UC3);
    I = (double *)malloc(image_w * image_h * image_channel * sizeof(double));
    I_padded = (double *)malloc(pad_w * pad_h * image_channel * sizeof(double));
    I_conv1 = (double *)malloc(conv1_w * conv1_h * image_channel * sizeof(double));
    I_pool1 = (double *)malloc(pool1_w * pool1_h * image_channel * sizeof(double));
    
    /*              Convert Mat to array                */
    mat2Im(Mat_image, I);


    /*              Algorithm                */
    // padding(I, I_padded, image_channel, image_w, image_h, padding_size);
    start_point = clock();
    convolution(I, weight, bias, I_conv1, 1, 1, conv_kSize, conv1_w, conv1_h, image_w, image_h, image_channel, conv_stride);
    end_point = clock();
    maxPooling(I_conv1, I_pool1, 1, conv1_w, conv1_h, image_channel);

    /*              Convert Array to Mat                */
    im2Mat(I, Mat_image, image_channel, image_h, image_w);
    // im2Mat(I_padded, Mat_padded, image_channel, pad_h, pad_w);
    im2Mat(I_conv1, Mat_conv1, image_channel, conv1_h, conv1_w);
    im2Mat(I_pool1, Mat_pool1, image_channel, pool1_h, pool1_w);

    cout << Mat_padded.rows << endl;
    cout << Mat_padded.cols << endl;
    filter2D(Mat_image, Mat_conv_CV, -1, weight_cv, Point(0, 0), 0, BORDER_CONSTANT);
    /*              Plot Images                */
    imshow("test", Mat_image);
    waitKey();
    // imshow("test1", Mat_padded);
    waitKey();
    imshow("test2", Mat_conv1);
    waitKey();
    imshow("test3", Mat_pool1);
    waitKey();
    imshow("testtt", Mat_conv_CV);
    waitKey();

    cout << "Convolution 수행 시간: " << double(end_point - start_point) / 1000.0 << "ms" << endl;

    
/*              Save Images                */
    // imwrite("./output_pad.jpg", Mat_padded);
    imwrite("./output_conv.jpg", Mat_conv1);
    imwrite("./output_pool.jpg", Mat_pool1);

    ofstream fp;
    ofstream fp2;
    fp.open("./conv_out.txt");
    fp2.open("./conv_out_compare.txt");
    for (int ch = 0; ch < Mat_conv1.channels(); ch++)
    {
        for (int row = 0; row < Mat_conv1.rows;row++)
        {
            for (int col = 0; col < Mat_conv1.cols; col++)
            {
                cout << Mat_conv1.at<Vec3b>(row, col)[ch] - Mat_conv_CV.at<Vec3b>(row, col)[ch] << " ";
                // cout << double(Mat_conv_CV.at<Vec3b>(row, col)[ch]) << endl;
            }
        }
    }    
    
    fp.close();

/*              Memory Deallocation                */
    free(I);
    free(I_padded);
    free(I_conv1);
    free(I_pool1);

   return 0;
}
