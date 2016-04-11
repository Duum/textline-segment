#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

#define XWID 5
#define YWID 5

class doublearray
{
    public:
        int row;
        int col;
        double **data;

    public:
        doublearray(){}
        doublearray( const doublearray& other):
            row( other.row ), col( other.col ), data (other.data)
        {}

        doublearray(const int _row, const int _col)
        {
            row = _row;
            col = _col;
            data = new double *[row];
            for(int i = 0 ; i < row; i++)
                data[i] = new double [col];
        }

        doublearray(const Mat& img){
            row = img.rows;
            col = img.cols;

            data = new double* [img.rows];
            for(int i = 0; i < img.rows; i++)
            {
                 data[i] = new double[img.cols];
                 for(int j = 0; j < img.cols; j++)
                       data[i][j] = (double)(img.at<uchar>(i,j));
            }
        }

        double& operator()(const int row, const int col);

        ~doublearray()
        {
            for(int i = 0; i < row; i++)
                delete[] data[i];
            delete[] data;
        }
};

double& doublearray::operator()(const int row, const int col)
{
    return (data[row][col]);
}

doublearray load_data(Mat img)
{
    doublearray obj;
    obj.row = img.rows;
    obj.col = img.cols;

    obj.data = new double* [img.rows];
    for(int i = 0; i < img.rows; i++)
    {
         obj.data[i] = new double[img.cols];
         for(int j = 0; j < img.cols; j++)
               obj.data[i][j] = (double)(img.at<uchar>(i,j));
    }

    return obj;
}

doublearray Gaussian(doublearray inp, double sigma_u, double sigma_v, double theta)
{
    double sigma_x, sigma_phi, phi, tan_phi;

    double const_term = sqrt( sigma_u * sigma_u * cos(theta) + sigma_v * sigma_v * sin(theta) );

    sigma_x     = (sigma_u * sigma_v)/const_term;
    phi         = atan( const_term / ( (sigma_u*sigma_u - sigma_v*sigma_v) * cos(theta) * sin(theta) ) );
    sigma_phi   = const_term / sin(phi);
    tan_phi     = tan(phi);

    int wid_x = 1 + (XWID/2);
    int wid_y = 1 + (YWID/2);
    double weightX[ wid_x ];
    double weightY[ wid_y ];
    double a = 0.75;

    for(int i = 0; i < wid_x; i++)
        weightX[i] = (1.0 / ( sqrt(2*M_PI) * sigma_x)) * exp(-0.5 * (double)(i*i)/(sigma_x * sigma_x));
    for(int i = 0; i < wid_y; i++)
        weightY[i] = (1.0 / ( sqrt(2*M_PI) * sigma_phi)) * exp(-0.5 * (double)(i*i)/(sigma_phi * sigma_phi));

    doublearray newImg( inp );

    for(int i = wid_x; i < (inp.row - wid_x); i++)
        for(int j = wid_y; i < (inp.col - wid_y); j++)
            {
                newImg.data[i][j] = weightX[0]*inp(i, j);
                for(int k = 1; k < wid_x; k++)
                    newImg.data[i][j] += weightX[k]*(inp(i+k,j) + inp(i-k,j));
            }

    doublearray outImg( newImg );
    for(int i = wid_x; i < (inp.row - wid_x); i++)
        for(int j = wid_y; i < (inp.col - wid_y); j++)
            {
                outImg.data[i][j] = newImg(i, j);
                for(int k = 1; k < wid_x; k++)
                {
                    outImg.data[i][j] += weightX[k]* a * (newImg( (int)( i - k/tan_phi ), j - k) + newImg( (int)( i + k/tan_phi ), j + k) );
                    outImg.data[i][j] += weightY[k]* (1-a) * ( newImg( (int)( i - k/tan_phi ) - 1, j - k) + newImg( (int)( i + k/tan_phi ) + 1, j + k) );
                }
            }

    return outImg;
}

int main()
{
    Mat img = imread("rice.bmp");

    if( img.empty())
    {
        cout << "File not available for reading"<<endl;
        return -1;
    }

    doublearray myData = load_data(img);


    return 0;
}

