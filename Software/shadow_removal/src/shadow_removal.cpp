#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
//#include "types_c.h"

using namespace std;
using namespace cv;

int main(){
	//Nombre de la imagen que se va a cargar
	char IMAGE_NAME[] = "road_and_shadow.jpg";

	//Cargamos las imagenes y se comprueba que lo ha hecho correctamente
	Mat src = imread(IMAGE_NAME);
	if (!src.data){
		cout << "Error al cargar la imagenes" << endl;
		exit(1);
	}

	vector<Mat> bgr_planes;
	split(src, bgr_planes);
   namedWindow("Original", CV_WINDOW_AUTOSIZE), imshow("Original", src);
   namedWindow("BLUE PLANE", CV_WINDOW_AUTOSIZE), imshow("BLUE PLANE", bgr_planes[0]);
   namedWindow("GREEN PLANE", CV_WINDOW_AUTOSIZE), imshow("GREEN PLANE", bgr_planes[1]);
   namedWindow("RED PLANE", CV_WINDOW_AUTOSIZE), imshow("RED PLANE", bgr_planes[2]);

   Mat srcHSV;
   cvtColor(src, srcHSV, CV_BGR2HSV);
	vector<Mat> hsv_planes;
	split(srcHSV, hsv_planes);
   namedWindow("HSV", CV_WINDOW_AUTOSIZE), imshow("HSV", srcHSV);
   namedWindow("HUE PLANE", CV_WINDOW_AUTOSIZE), imshow("HUE PLANE", hsv_planes[0]);
   namedWindow("SATURATION PLANE", CV_WINDOW_AUTOSIZE), imshow("SATURATION PLANE", hsv_planes[1]);
   namedWindow("INTENSITY PLANE", CV_WINDOW_AUTOSIZE), imshow("INTENSITY PLANE", hsv_planes[2]);

   Mat ndi_image = Mat::zeros(srcHSV.rows, srcHSV.cols, CV_8UC1);
   int tono, saturacion, valor;
   for (int y = 0; y < srcHSV.cols; y++){
       for(int x = 0; x < srcHSV.rows; x++){
          tono = srcHSV.at<cv::Vec3b>(y,x)[0];
          saturacion = srcHSV.at<cv::Vec3b>(y,x)[1];
          valor = srcHSV.at<cv::Vec3b>(y,x)[2];
          ndi_image.at<cv::Vec2d>(x,y) = (saturacion - valor)/(saturacion + valor);
       }
   }

	cvWaitKey(0);
	return 0;
}
