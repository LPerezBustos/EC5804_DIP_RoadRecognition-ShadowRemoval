#include <opencv2/opencv.hpp> //Include file for every supported OpenCV function
#include <iostream>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/ml/ml.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

int main( int argc, char** argv ) {


    //***ENTRENAMIENTO***//
    Ptr<TrainData> datos_t;      //Tipo de datos de entrenamiento para OpenCV

    /*--Cargar los datos de entenamiento desde el archivo "trainingdata_positives.csv"
    Tiene la forma (float)B,(float)G,(float)R,LABEL(-1 ó 1),  ex: "0.78824,0.79608,0.79608,1"
    En este archivo, cada fila contiene una muestra y el ultimo valor corresponde a la etiqueta "1" positivo, "-1" negativo
    La instrucción loadFromCSV usa varios parámetros, que indican la ruta del archivo de entrenamiento y tres valores para indicar la estructura de los datos.
    En este caso,  0 = offset (no hay fila para encabezados, con lo que la primera fila corresponde a la primera muestra)
    y los siguientes valores indican donde comienzan y donde terminan las muestras para separarlas de la etiqueta;  Segun la documentación de OpenCV  usar -1, -1
    indicará que solo hay una etiqueta y que la misma estará ubicada en el ultimo valor de la fila.
    --*/
    datos_t = cv::ml::TrainData::loadFromCSV("trainingdata_positives.csv",0,-1,-1);



    Ptr<SVM> svm = SVM::create();  //Genera una nueva instancia a la clase SVM
    svm->setKernel(SVM::LINEAR);  //Se define el tipo de kernel LINEAL
    svm->setTermCriteria(TermCriteria(TermCriteria::COUNT, 3000, 1e-2)); //Se define el criterio de parada para el entrenamiento. EN este caso se usa COUNT que implica un número finito de iteraciones (la decision fue empirica)


    //Finalmente se ejecuta la instruccion trainAuto, que se encargará de ejecutar el algoritmo de optimización para encontrar los vectores de soporte.
    //Aunque están especificados todos los valores dentro de la funcion, realmente solo son utilizados algunos pocos puesto que se está utilizando un kernel LINEAL
    //el primero toma como entrada los datos de entrenamiento
    //el segundo indica en cuantos subconjuntos será dividido el set de entrenamiento para que el algoritmo pueda medir el rendimiento de la clasificacion durante el entrenamiento
    //el tercero define el parametro C, que es un parámetro de holgura para poder aceptar un numero mayor de muestras que estan cerca del margen del hiperplano de separacion.
    //los parametros GAMMA, P, NU, COEF, y DEGREE  son necesarios para la ejecucion del metodo, pero no son utilizados en este ejemplo ya que son parametros para otros tipos de KERNEL.
    //el ultimo parametro es un booleano que en caso de ser verdadero implica que se está trabajando con un clasificador de dos clases y por tanto optimiza los vectores de soporte para encontrar una solucion más balanceada.

    svm->trainAuto(datos_t,5,SVM::getDefaultGrid(SVM::C),SVM::getDefaultGrid(SVM::GAMMA),SVM::getDefaultGrid(SVM::P),SVM::getDefaultGrid(SVM::NU),SVM::getDefaultGrid(SVM::COEF),SVM::getDefaultGrid(SVM::DEGREE),true);

    //**CLASIFICACION**/

    Mat img = imread("road.jpg");  //se carga la imagen y se convierte a punto flotante de 32bits (necesario por la libreria para el clasificador)
    cv::Mat img32(img.rows, img.cols, CV_32FC3);
    Mat markerMask(img32.rows,img32.cols,CV_32FC3),dst(img32.rows,img32.cols,CV_32FC3);
    Mat img132[3];
    img.convertTo(img32, CV_32FC3, 1/255.0);
    split(img32,img132);

    imshow("original",img32); //se muestra la imagen original (ya en 32bits)

    cout<<"Antes del for: R->"<<img32.rows<<" C->"<<img32.cols<<endl; //mostrar en pantalla el tamaño de la imagen ROWS, COLS.
    int i,j; //variables para iterar sobre los pixeles de la imagen
    float r=0.; //variable que indica el resultado de la etiqueta del clasificador (1.0 o -1.0)

     for(i=0;i<img32.rows;i++)
        for(j=0;j<img32.cols;j++)
        {
           float testData[3] = {img132[0].at<float>(i,j),img132[1].at<float>(i,j),img132[2].at<float>(i,j)}; //se genera un vector de 1x3 con la informacion BGR de cada pixel
           Mat sample(1, 3, CV_32FC1,testData);

            //El metodo PREDICT, evalúa de qué clase es la muestra y asigna la etiqueta a la variable "r"

            if ((r=svm->predict(sample))==1)
            {
                markerMask.at<Vec3f>(i,j) = Vec3f(0.,1.,0.);; //si la muestra es positiva, se genera un pixel verde en una imagen "mascara" conservando su posicion
            }
            else{
                markerMask.at<Vec3f>(i,j) = Vec3f(0.,0.,0.);; //si la muestra es negativa, se genera un pixel negro en la imagen "mascara" conservando su posicion
            }

        }

            //Algunas operaciones morfologicas simples para mejorar el resultado

            Mat element = getStructuringElement( 0,
                                       Size( 2*3 + 1, 2*3+1 ),
                                       Point( 3, 3 ) );
            erode( markerMask, markerMask,element);
            erode( markerMask, markerMask,element);
            dilate( markerMask, markerMask,element);
            dilate( markerMask, markerMask,element);


            imshow("marker", markerMask);  //Se muestra la imagen Máscara
            addWeighted( img32, 0.7, markerMask, 0.3, 0.0, dst); //Se superpone la mascara sobre la imagen original
    				markerMask.convertTo(markerMask, CV_8UC3, 255.0);
					imwrite("Road-Mask.jpg", markerMask);
            imshow("Reconocimiento", dst); //Se muestra el resultado final
    				dst.convertTo(dst, CV_8UC3, 255.0);
					imwrite("Final-Road-Recognition.jpg", dst);
            waitKey(0);
}
