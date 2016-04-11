#ifndef ANIGAUSS_H
#define ANIGAUSS_H

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

#ifndef PI
#ifdef M_PI
#define PI M_PI
#else
#define PI 3.14159265358979323846
#endif
#endif


/* define the input buffer type, e.g. "float" */
typedef double SRCTYPE;
//#define SRCTYPE double


/* define the output buffer type, should be at least "float" */
typedef double DSTTYPE;
//#define DSTTYPE double


/* the function prototypes */
void anigauss(SRCTYPE *input, DSTTYPE *output, int sizex, int sizey,
    double sigmav, double sigmau, double phi, int orderv, int orderu);
void YvVfilterCoef(double sigma, double *filter);
void TriggsM(double *filter, double *M);

static void f_iir_xline_filter(SRCTYPE *src, DSTTYPE *dest, int sx, int sy,
    double *filter);
static void f_iir_yline_filter(DSTTYPE *src, DSTTYPE *dest, int sx, int sy,
    double *filter);
static void f_iir_tline_filter(DSTTYPE *src, DSTTYPE *dest, int sx, int sy,
    double *filter, double tanp);
static void f_iir_derivative_filter(DSTTYPE *src, DSTTYPE *dest, int sx, int sy,
    double phi, int order);
static void f_iir_linear_filter(DSTTYPE *src, DSTTYPE *dest, int sx, int sy,
    double phi);



/*
 *  the main function:
 *    anigauss(inbuf, outbuf, bufwidth, bufheight, sigma_v, sigma_u, phi,
 *       derivative_order_v, derivative_order_u);
 *
 *  v-axis = short axis
 *  u-axis = long axis
 *  phi = orientation angle in degrees
 *
 *  for example, anisotropic data smoothing:
 *    anigauss(inptr, outptr, 512, 512, 3.0, 7.0, 30.0, 0, 0);
 *
 *  or, anisotropic edge detection:
 *    anigauss(inptr, outptr, 512, 512, 3.0, 7.0, 30.0, 1, 0);
 *
 *  or, anisotropic line detection:
 *    anigauss(inptr, outptr, 512, 512, 3.0, 7.0, 30.0, 2, 0);
 *
 *  or, in-place anisotropic data smoothing:
 *    anigauss(bufptr, bufptr, 512, 512, 3.0, 7.0, 30.0, 0, 0);
 *
 */



#endif
