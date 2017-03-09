#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int main(){
	//Nombre de la imagen que se va a cargar
	char IMAGE_NAME[] = "road_and_shadow_2.jpg";

	//Cargamos las imagenes y se comprueba que lo ha hecho correctamente
	Mat src = imread(IMAGE_NAME);
	if (!src.data){
		cout << "Error al cargar la imagenes" << endl;
		exit(1);
	}

	/*vector<Mat> bgr_planes, H, S, V;
	split(src, bgr_planes);

   namedWindow("Original", CV_WINDOW_AUTOSIZE), imshow("Original", src);
   namedWindow("BLUE PLANE", CV_WINDOW_AUTOSIZE), imshow("BLUE PLANE", bgr_planes[0]);
   namedWindow("GREEN PLANE", CV_WINDOW_AUTOSIZE), imshow("GREEN PLANE", bgr_planes[1]);
   namedWindow("RED PLANE", CV_WINDOW_AUTOSIZE), imshow("RED PLANE", bgr_planes[2]);*/
   Mat hsv_image
   for (int y = 0; y < src.cols; y++){
       for(int x = 0; x < img.rows; x++){
          image.at<cv::Vec3b>(y,x)[0]
          uchar *blue = ((uchar*)(img->imageData + img->widthStep*y))[x*3];
          uchar *green = ((uchar*)(img->imageData + img->widthStep*y))[x*3+1];
          uchar *red = ((uchar*)(img->imageData + img->widthStep*y))[x*3+2];
       }
   }

	cvWaitKey(0);
	return 0;
}
