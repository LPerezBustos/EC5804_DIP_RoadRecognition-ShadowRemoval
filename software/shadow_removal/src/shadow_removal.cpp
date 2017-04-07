#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

#define SHADOWS_TO_REMOVE 4
#define CONNECTED_1_Y 291
#define CONNECTED_1_X 266
#define CONNECTED_2_Y 423
#define CONNECTED_2_X 318
#define CONNECTED_3_Y 582
#define CONNECTED_3_X 313
#define CONNECTED_4_Y 82
#define CONNECTED_4_X 441

using namespace std;
using namespace cv;

int main(){
	// find the image
	char IMAGE_NAME[] = "road_and_shadow.jpg";
	Mat src = imread(IMAGE_NAME);
	if (!src.data){
		cout << "Error al cargar la imagenes" << endl;
		exit(1);
	}
	// show RGB planes
	vector<Mat> bgr_planes;
	split(src, bgr_planes);
   namedWindow("Original", CV_WINDOW_AUTOSIZE), imshow("Original", src);
   namedWindow("BLUE PLANE", CV_WINDOW_AUTOSIZE), imshow("BLUE PLANE", bgr_planes[0]);
   namedWindow("GREEN PLANE", CV_WINDOW_AUTOSIZE), imshow("GREEN PLANE", bgr_planes[1]);
   namedWindow("RED PLANE", CV_WINDOW_AUTOSIZE), imshow("RED PLANE", bgr_planes[2]);
	// convert and show HSV planes
   Mat srcHSV, hsv_planes[3], ndi_image;
   cvtColor(src, srcHSV, CV_BGR2HSV);
	split(srcHSV, hsv_planes);
	imwrite("shadow_removal_HSV.jpg", srcHSV);
   namedWindow("HSV", CV_WINDOW_AUTOSIZE), imshow("HSV", srcHSV);
   namedWindow("HUE PLANE", CV_WINDOW_AUTOSIZE), imshow("HUE PLANE", hsv_planes[0]);
   namedWindow("SATURATION PLANE", CV_WINDOW_AUTOSIZE), imshow("SATURATION PLANE", hsv_planes[1]);
   namedWindow("INTENSITY PLANE", CV_WINDOW_AUTOSIZE), imshow("INTENSITY PLANE", hsv_planes[2]);
	// convert to 32bits float
	hsv_planes[0].convertTo(hsv_planes[0], CV_32F, 1/255.0);
	hsv_planes[1].convertTo(hsv_planes[1], CV_32F, 1/255.0);
	hsv_planes[2].convertTo(hsv_planes[2], CV_32F, 1/255.0);
	// normalized difference index
	ndi_image = (hsv_planes[1]-hsv_planes[2])/(hsv_planes[1]+hsv_planes[2]);
	ndi_image += 1.0; // set all the values to positive
	ndi_image.convertTo(ndi_image, CV_8U, 255.0/2.0); // convert to range {0,255}
   namedWindow("NDI_IMAGE", CV_WINDOW_AUTOSIZE), imshow("NDI_IMAGE", ndi_image);
	imwrite("shadow_removal_NDI.jpg", ndi_image);
	// show histogram of NDI IMAGE (GRAY SCALE)
	// **** initial parameters ****
	int histSize = 256;
	float range[] = {0,255};
	const float *ranges[] = {range};
	// **** calculate histogram ****
	MatND hist;
	calcHist( &ndi_image, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false );
	double total;
	total = ndi_image.rows * ndi_image.cols;
	for(int h = 0; h < histSize; h++){
			  float binVal = hist.at<float>(h);
		  }
	// **** plot the histogram ****
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/histSize );
	Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 0,0,0) );
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	int i;
	for(i = 1; i < histSize; i++){
	  line(histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
							 Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
							 Scalar( 255, 0, 0), 2, 8, 0 );
	}
	imwrite("shadow_removal_HIST.jpg", histImage);
	namedWindow("NDI_IMAGE HISTOGRAM", CV_WINDOW_AUTOSIZE), imshow("NDI_IMAGE HISTOGRAM", histImage);
	// find OTSU's threshold
	Mat otsu_image(ndi_image.size(), CV_8UC1);
	double umbral = threshold(ndi_image, otsu_image, 0, 255, THRESH_BINARY+THRESH_OTSU);
	namedWindow("BINARY IMAGE", CV_WINDOW_AUTOSIZE), imshow("BINARY IMAGE", otsu_image);
	cout << "The OTSU threshold is " << umbral << endl;
	imwrite("otsu_image.jpg", otsu_image);
	// find sample pixels
	// *** SOME METHOD TO FIND SEED PIXELS IN SHADOWS ***
	// find connected components using paper method
	int dilation_size = 1;
	cout << "structure element in shadow classfication stage is (" << 2*dilation_size+1 << "x" << 2*dilation_size+1 << ")" << endl;
	Mat dilation_element = getStructuringElement(MORPH_RECT, Size(2*dilation_size+1, 2*dilation_size+1),
						Point(dilation_size, dilation_size));
	// seed pixels in connected components
	Mat Io = Mat::zeros(otsu_image.rows, otsu_image.cols, CV_8UC1);
	Io.at<uchar>(CONNECTED_1_X,CONNECTED_1_Y) = 255;
	Io.at<uchar>(CONNECTED_2_X,CONNECTED_2_Y) = 255;
	Io.at<uchar>(CONNECTED_3_X,CONNECTED_3_Y) = 255;
	Io.at<uchar>(CONNECTED_4_X,CONNECTED_4_Y) = 255;
	imwrite("shadow_removal_seed.jpg", Io);
	// shadow classfication
	Mat Idilate(otsu_image.size(), CV_8UC1);
	Mat Ishadow(otsu_image.size(), CV_8UC1);
	Mat diff(otsu_image.size(), CV_8UC1);
	Idilate = Io.clone();
	i = 0;
	do{
		Ishadow = Idilate.clone();
		dilate(Idilate, Idilate, dilation_element);
		Idilate = Idilate & otsu_image;
		diff = (Idilate != Ishadow);
		i++;
	} while(countNonZero(diff));
	cout << "number of iterations in shadow classfication stage is " << i << endl;
	imwrite("shadow_removal_CONNECTED_COMP.jpg", Ishadow);
	namedWindow("CONNECTED COMPONENTS", CV_WINDOW_AUTOSIZE), imshow("CONNECTED COMPONENTS", Ishadow);
	// buffer area estimation
	Mat shadow_borders = Idilate.clone();
	dilate(Idilate, shadow_borders, dilation_element);
	shadow_borders = shadow_borders - Idilate;
	imwrite("shadow_removal_borders.jpg", shadow_borders);
	namedWindow("BUFFER AREA ESTIMATION", CV_WINDOW_AUTOSIZE), imshow("BUFFER AREA ESTIMATION", shadow_borders);
	Scalar border_mean, border_stddev, border_variance, shadow_mean, shadow_stddev, shadow_variance;
	meanStdDev(srcHSV, shadow_mean, shadow_stddev, Ishadow);
	cout << "Mean and StdDev of first shadow" << endl;
	cout << shadow_mean[0] << " " << shadow_mean[1] << " " << shadow_mean[2] << endl;
	cout << shadow_stddev[0] << " " << shadow_stddev[1] << " " << shadow_stddev[2] << endl;
	meanStdDev(srcHSV, border_mean, border_stddev, shadow_borders);
	cout << "Mean and StdDev of buffer area in first shadow" << endl;
	cout << border_mean[0] << " " << border_mean[1] << " " << border_mean[2] << endl;
	cout << border_stddev[0] << " " << border_stddev[1] << " " << border_stddev[2] << endl;
	pow(border_stddev, 2, border_variance);
	pow(shadow_stddev, 2, shadow_variance);
	Scalar variance_ratio;
	divide(shadow_variance, border_variance, variance_ratio);
	cout << "Variance ratio --> H S V" << endl;
	cout << variance_ratio[0] << " " << variance_ratio[1] << " " << variance_ratio[2] << endl;
	int j;
	for(i=0; i<Ishadow.rows; i++){
		for(j=0; j<Ishadow.cols; j++){
			uchar pix = Ishadow.at<uchar>(i,j);
			if(pix){
				Scalar pix_value = srcHSV.at<Vec3b>(i,j);
				srcHSV.at<Vec3b>(i,j)[0] = border_mean[0] + variance_ratio[0]*(pix_value.val[0] - shadow_mean[0]);
				srcHSV.at<Vec3b>(i,j)[1] = border_mean[1] + variance_ratio[1]*(pix_value.val[1] - shadow_mean[1]);
				srcHSV.at<Vec3b>(i,j)[2] = border_mean[2] + variance_ratio[2]*(pix_value.val[2] - shadow_mean[2]);
				/*Scalar pix_value2 = srcHSV.at<Vec3b>(i,j);
				cout << pix_value[2] << " " << pix_value2[2] << endl;*/
			}
		}
	}
	namedWindow("HSV image without Shadows", CV_WINDOW_AUTOSIZE), imshow("HSV image without Shadows", srcHSV);
   cvtColor(srcHSV, src, CV_HSV2BGR);
	namedWindow("RGB image without Shadows", CV_WINDOW_AUTOSIZE), imshow("RGB image without Shadows", src);
	imwrite("shadow_removal_road_final.jpg", src);
	cvWaitKey(0);
	return 0;
}
