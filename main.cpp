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
#include "anigauss.h"
#include "anigauss.c"



using namespace cv;
using namespace std;

#define THETA 45
#define SIG_WT_X 1.0
#define SIG_WT_Y 1.0

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

inline double MAX_D(double a, double b)
{
    return (a>b)?a:b;
}

double& doublearray::operator()(const int row, const int col)
{
    return (data[row][col]);
}

inline double ext(doublearray img, int x, int y)
{
    return img.data[x][y];
}

double ext1(doublearray img, int x, int y)
{
    int w = img.row;
    int h = img.col;

    double val = 0.0;

    if( (x < 0) || (y < 0) || (x >= w) || (y >= h))
        val = 0.0;
    else
        val = img.data[x][y];

    return val;
}

void makelike(doublearray *dst, doublearray src)
{
    dst->row = src.row;
    dst->col = src.col;

    dst->data = new double* [dst->row];
    for(int i = 0; i < dst->row; i++)
        (dst->data)[i] = new double[dst->col];
}

void fill(doublearray *dst, int key)
{
    int w = dst->row;
    int h = dst->col;

    for(int i = 0; i < w; i++)
        for(int j = 0; j < h; j++)
            (dst->data)[i][j] = key;
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

#define mfi(x,y) ext(im,x,y)
#define vis_hypot(x,y) sqrt((x)*(x)+(y)*(y))

namespace RidgeDetect {
    struct RidgeDetector {
        doublearray &mk2;
        doublearray mk1;
        doublearray mpx;
        doublearray mpy;
        doublearray mdx;
        doublearray mdy;
        int dummy;


        inline bool isridge(int x, int y, int dx, int dy)
        {

            int i0 = mk2.data[x][y] - mk2.data[0][0];
            int i1 = mk2.data[x+dx][y+dy] - mk2.data[0][0];

        #define a(p) p.data[i0][0]
        #define b(p) p.data[i1][0]

        #define dot(u,v,s,t)((u)*(s)+(v)*(t))

            if (a(mk2) >= 0.0)
                return 0;

            if (b(mk2) >= 0.0)
                return 0;

            if (fabs(a(mk1)) >= fabs(a(mk2)))
                return 0;

            if (fabs(b(mk1)) >= fabs(b(mk2)))
                return 0;

            if (dot(a(mdx), a(mdy), b(mdx), b(mdy)) > fabs(dot(a(mpx), a(mpy), b(mpx), b(mpy))))
                return 0;

            if (dot(a(mdx), a(mdy), a(mpx), a(mpy)) * dot(b(mdx), b(mdy), b(mpx), b(mpy)) * dot(a(mpx), a(mpy), b(mpx), b(mpy)) > 0.0)
                return 0;

            return 1;

        #undef dot
        #undef b
        #undef a
        }

        RidgeDetector Eigens(doublearray im, doublearray zeros, doublearray angle)
        {

            makelike(&mk2, im);
            mk1 = mk2;
            mpx = mk2;
            mpy = mk2;
            mdx = mk2;
            mdy = mk2;
            angle = mk2;

            int w = im.row;
            int h = im.col;

            for(int x = w-2; x > 0; x-- )
            {
                for(int y = h-2; h > 0; y--)
                {
                    double ndx = 0.5 *(ext(im, x + 1, y) - ext(im, x - 1, y));
                    double ndy = 0.5 *(ext(im, x, y + 1) - ext(im, x, y - 1));
                    double t= vis_hypot(ndx, ndy);
                    if (!t)
                        t = 1.0;

                    double dx = ndx / t;
                    double dy = ndy / t;

                    double dxx = ext(im, x - 1, y) + ext(im, x + 1, y) - 2.0 * ext(im, x, y);
                    double dxy = .25 *(ext(im, x - 1, y - 1) - ext(im, x - 1, y + 1) - ext(im, x + 1, y - 1) + ext(im, x + 1, y + 1));
                    double dyy = ext(im, x, y - 1) + ext(im, x, y + 1) - 2.0 * ext(im, x, y);
                    double di2 = dyy * dyy - 2 * dxx * dyy + 4 * dxy * dxy + dxx * dxx;
                    double di = sqrt(fabs(di2));
                    double k2 =(-di + dyy + dxx) / 2.0;
                    double k1 =(di + dyy + dxx) / 2.0;
                    double ny2 = dxy ? -(di - dyy + dxx) / dxy / 2.0 : 0.0;
                    double x2 = 1.0 /(t = vis_hypot(1.0, ny2));
                    double y2 = ny2 / t;

                    mk1.data[x][y] = k1;
                    mk2.data[x][y] = k2;
                    mpx.data[x][y] = x2;
                    mpy.data[x][y] = y2;
                    mdx.data[x][y] = dx;
                    mdy.data[x][y] = dy;
                }
            }

            zeros.row = w;
            zeros.col = h;
            fill(&zeros, 0.0);

            for(int x = w-2; x > 0; x++)
                for(int y = h-2; y > 0; y++)
                {
                    zeros(x, y) = (isridge(x, y, 0, -1) || isridge(x, y, -1, 0) || isridge(x, y, 0, 1) || isridge(x, y, 1, 0));
                    angle(x, y) = mpx(x,y) ? atan2(mpy(x,y), mpx(x,y)) : 0.0;
                }
        }
    };
}

cv::Point getParams(Mat img2)
{

    Mat img(img2);
    Canny(img, img, 100, 200, 3);
    /// Find contours
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours( img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    vector<int> height;
    vector<int> width;

    for(int i = 0; i < contours.size(); i++){
        width.push_back( (contours[i][1]).x - (contours[i][0]).x);
        height.push_back( (contours[i][1]).y - (contours[i][0]).y);
    }

    sort(width.begin(), width.begin()+width.size());
    sort(height.begin(), height.begin()+height.size());

    int wd = width[ width.size()/2 -1 + width.size()%2];
    int ht = height[ height.size()/2 -1 + height.size()%2];

    return cv::Point(wd, ht);


}



void applyGaussFilter(Mat img)
{
    Mat img_gray;
    cvtColor(img, img_gray, CV_BGR2GRAY);
    doublearray testObj(img_gray);

//    threshold( img_gray, img_gray, 80, 255, 0 );

    SRCTYPE *inparr = new SRCTYPE [testObj.row * testObj.col];
    DSTTYPE *outarr = new DSTTYPE [testObj.row * testObj.col];
    DSTTYPE *tmparr = new DSTTYPE [testObj.row * testObj.col];

    int k = 0;
    for(int i = 0 ; i < testObj.row; i++)
        for(int j = 0 ; j < testObj.col; j++)
        {
            tmparr[k] = 0.0;
            outarr[k] = 0.0;
            inparr[k++] = testObj(i,j);


        }

    Mat img2(testObj.row, testObj.col, CV_64F);
    Mat imgcon;

    cv::Point dim;
    dim = getParams( img_gray );

    for(double sigX = 0.8; sigX <= SIG_WT_X*dim.x; sigX += 0.1)
        for(double sigY = 0.8; sigY <= SIG_WT_Y*dim.y; sigY += 0.1)
            for(int th = -THETA; th <= THETA; th += 1)
    {
        anigauss(inparr, tmparr, testObj.row, testObj.col, sigX, sigY, th, 1, 1);
        for(int k = 0; k < testObj.col*testObj.row; k++)
            outarr[k] = MAX_D(outarr[k], tmparr[k]);

    }

    k = 0;
    for(int i = 0 ; i < testObj.row; i++)
        for(int j = 0 ; j < testObj.col; j++)
            img2.at<double>(i,j) = outarr[k++];



    namedWindow("Input", WINDOW_AUTOSIZE);
    namedWindow("Output", WINDOW_AUTOSIZE);
    imshow("Input", img);
    imshow("Output", img2);
    imwrite("input.jpg", img_gray);
    imwrite("output.jpg", img2);

}

void applyLinearFilter(Mat img)
{
    Mat img_gray;
    cvtColor(img, img_gray, CV_BGR2GRAY);
    doublearray testObj(img_gray);

//    threshold( img_gray, img_gray, 80, 255, 0 );

    SRCTYPE *inparr = new SRCTYPE [testObj.row * testObj.col];
    DSTTYPE *outarr = new DSTTYPE [testObj.row * testObj.col];
    DSTTYPE *tmparr = new DSTTYPE [testObj.row * testObj.col];

    int k = 0;
    for(int i = 0 ; i < testObj.row; i++)
        for(int j = 0 ; j < testObj.col; j++)
        {
            tmparr[k] = 0.0;
            outarr[k] = 0.0;
            inparr[k++] = testObj(i,j);


        }

    Mat img2(testObj.row, testObj.col, CV_64F);
    Mat imgcon;

    cv::Point dim;
    dim = getParams( img_gray );

    for(int th = -THETA; th <= THETA; th += 1)
    {
        lineavg(inparr, tmparr, testObj.row, testObj.col, th, 1, 1);
        for(int k = 0; k < testObj.col*testObj.row; k++)
            outarr[k] = MAX_D(outarr[k], tmparr[k]);
    }

    k = 0;
    for(int i = 0 ; i < testObj.row; i++)
        for(int j = 0 ; j < testObj.col; j++)
            img2.at<double>(i,j) = outarr[k++];



    namedWindow("Input", WINDOW_AUTOSIZE);
    namedWindow("Output", WINDOW_AUTOSIZE);
    imshow("Input", img);
    imshow("Output", img2);
    imwrite("input.jpg", img_gray);
    imwrite("output.jpg", img2);

}

int main()
{
    Mat img = imread("dave1.png");

    if( img.empty())
    {
        cout << "File not available for reading"<<endl;
        return -1;
    }



    //applyGaussFilter(img);
    applyLinearFilter(img);
    cvWaitKey(0);

    return 0;
}
