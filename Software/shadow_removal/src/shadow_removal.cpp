#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int main(){
	//Nombre de la imagen que se va a cargar
	char NombreImagen[] = "road_and_shadow.jpg";

	//Cargamos las imagenes y se comprueba que lo ha hecho correctamente
	Mat src = imread(NombreImagen);
	if (!src.data){
		cout << "Error al cargar la imagenes" << endl;
		exit(1);
	}

	Mat dst;

	cvWaitKey(0);
	return 0;
}
