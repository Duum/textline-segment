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

class floatarray
{
    public:
        int row;
        int col;
        float **data;

    public:
        floatarray(){}
        floatarray( const floatarray& other):
            row( other.row ), col( other.col ), data (other.data)
        {}

        floatarray(const Mat& img){
            row = img.rows;
            col = img.cols;

            data = new float* [img.rows];
            for(int i = 0; i < img.rows; i++)
            {
                 data[i] = new float[img.cols];
                 for(int j = 0; j < img.cols; j++)
                       data[i][j] = (float)(img.at<uchar>(i,j));
            }
        }

        float& operator()(const int row, const int col);

        ~floatarray()
        {
            for(int i = 0; i < row; i++)
                delete[] data[i];
            delete[] data;
        }
};

float& floatarray::operator()(const int row, const int col)
{
    return (data[row][col]);
}

inline float ext(floatarray img, int x, int y)
{
    return img.data[x][y];
}

float ext1(floatarray img, int x, int y)
{
    int w = img.row;
    int h = img.col;

    float val = 0.0;

    if( (x < 0) || (y < 0) || (x >= w) || (y >= h))
        val = 0.0;
    else
        val = img.data[x][y];

    return val;
}

void makelike(floatarray *dst, floatarray src)
{
    dst->row = src.row;
    dst->col = src.col;

    dst->data = new float* [dst->row];
    for(int i = 0; i < dst->row; i++)
        (dst->data)[i] = new float[dst->col];
}

void fill(floatarray *dst, int key)
{
    int w = dst->row;
    int h = dst->col;

    for(int i = 0; i < w; i++)
        for(int j = 0; j < h; j++)
            (dst->data)[i][j] = key;
}

floatarray load_data(Mat img)
{
    floatarray obj;
    obj.row = img.rows;
    obj.col = img.cols;

    obj.data = new float* [img.rows];
    for(int i = 0; i < img.rows; i++)
    {
         obj.data[i] = new float[img.cols];
         for(int j = 0; j < img.cols; j++)
               obj.data[i][j] = (float)(img.at<uchar>(i,j));
    }

    return obj;
}

#define mfi(x,y) ext(im,x,y)
#define vis_hypot(x,y) sqrt((x)*(x)+(y)*(y))

namespace RidgeDetect {
    struct RidgeDetector {
        floatarray &mk2;
        floatarray mk1;
        floatarray mpx;
        floatarray mpy;
        floatarray mdx;
        floatarray mdy;
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

        RidgeDetector Eigens(floatarray im, floatarray zeros, floatarray angle)
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
                    float ndx = 0.5 *(ext(im, x + 1, y) - ext(im, x - 1, y));
                    float ndy = 0.5 *(ext(im, x, y + 1) - ext(im, x, y - 1));
                    float t= vis_hypot(ndx, ndy);
                    if (!t)
                        t = 1.0;

                    float dx = ndx / t;
                    float dy = ndy / t;

                    float dxx = ext(im, x - 1, y) + ext(im, x + 1, y) - 2.0 * ext(im, x, y);
                    float dxy = .25 *(ext(im, x - 1, y - 1) - ext(im, x - 1, y + 1) - ext(im, x + 1, y - 1) + ext(im, x + 1, y + 1));
                    float dyy = ext(im, x, y - 1) + ext(im, x, y + 1) - 2.0 * ext(im, x, y);
                    float di2 = dyy * dyy - 2 * dxx * dyy + 4 * dxy * dxy + dxx * dxx;
                    float di = sqrt(fabs(di2));
                    float k2 =(-di + dyy + dxx) / 2.0;
                    float k1 =(di + dyy + dxx) / 2.0;
                    float ny2 = dxy ? -(di - dyy + dxx) / dxy / 2.0 : 0.0;
                    float x2 = 1.0 /(t = vis_hypot(1.0, ny2));
                    float y2 = ny2 / t;

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


int main()
{
    Mat img = imread("rice.bmp");

    if( img.empty())
    {
        cout << "File not available for reading"<<endl;
        return -1;
    }

    floatarray testObj(img);

    return 0;
}
