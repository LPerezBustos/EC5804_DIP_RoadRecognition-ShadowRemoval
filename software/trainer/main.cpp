#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <cstdio>
#include <iostream>
#include <fstream>

using namespace cv; //Cargar ambiente de openCV
using namespace std; //cargar ambiente estandar de C

//Método para mostrar ayuda del programa
static void help()
{
    cout << "\nEste programa es una herramienta para obtener los datos de entrenamiento en el reconocimiento de caminos\n"
            << endl;
   cout << "\nUso: Seleccione el tamaño del pincel con el slider superior, luego marcar toda la zona correspondiente al pavimento (no incluir marcas de vía)\nSi desea, puede presionar \"c\" para visualizar la media y varianza, o puede guardar directamente los datos con \"g\".\n"
            << endl;

    cout << "Accesos directos: \n"
        "\tESC - finalizar programa\n"
        "\tr - restaurar imagen original\n"
        "\tc - calcular Promedio y Varianza\n"
        "\tt - Alternar tipo de muestra (Positiva/Negativa)\n"
        "\tg - guardar clasificacion\n"
        "\th - ayuda\n"
        << endl;
}

Mat markerMask, img;

//*** Datos para definir linea del marcador
Point prevPt(-1, -1);
int thickness = 10;

//***  Datos para definir letras en imagen
string muestra_tipo="Negativa";
Scalar muestra_color=Scalar(0,255,0);

//*** Archivos para guardar datos de entrenamiento
ofstream positivos,negativos;

static void on_trackbar(int,void*)
{

}

//Definicion de eventos para poder dibujar sobre la imagen con el mouse
static void onMouse( int event, int x, int y, int flags, void* )
{
    if( x < 0 || x >= img.cols || y < 0 || y >= img.rows )
        return;
    if( event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON) )
        prevPt = Point(-1,-1);
    else if( event == EVENT_LBUTTONDOWN )
        prevPt = Point(x,y);
    else if( event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON) )
    {
        Point pt(x, y);
        if( prevPt.x < 0 )
            prevPt = pt;
        line( markerMask, prevPt, pt, Scalar::all(255), thickness, 8, 0 );
        line( img, prevPt, pt, Scalar::all(255), thickness, 8, 0 );
        prevPt = pt;
        imshow("image", img);

    }
}

int main( int argc, char** argv )
{
    //parser para leer la imagen con el comando input desde command line.  Por defecto abrirá una imagen dentro de la carpeta training_data/xxx.jpg
    cv::CommandLineParser parser(argc, argv, "{help h | | }{ @input | training_data/urs.jpg | }");
    if (parser.has("help"))
    {
        help();
        return 0;
    }

    string filename = parser.get<string>("@input"); //leer string del nombre de archivo

    Mat img0 = imread(filename); //cargar imagen
    Scalar mean,stddev; //Variable tipo cv::Scalar que contendrá los valores de la media y la varianza

    if( img0.empty() ) //Verificar qe la imagen se pudo abrir
    {
        cout << "No se pudo abrir la imagen: " << filename << ". Uso: meanstd <training_data/nombre_imagen>\n";
        return 0;
    }

    help();  //muestra la ayuda en command prompt


    namedWindow( "image", 1 ); //crear ventana donde se mostrará la imagen cargada.

    createTrackbar("Size","image",&thickness,100,on_trackbar); //slider para definir tamaño del marcador

    img0.copyTo(img); //se clona la imagen original "img0" a "img" para no perderla en caso de que se desee restaurar

    cvtColor(img, markerMask, COLOR_BGR2GRAY); //clonar img y copiar la imagen que guardará la mascara para calcular el promedio
    markerMask = Scalar::all(0);


    rectangle(img,Point2d(28,3),Point2d(170,38),Scalar(0,0,0),-1,LINE_8,0);  //Mostrar texto con tipo de muestra (no necesario)
    putText(img,muestra_tipo,Point2d(30,30),FONT_HERSHEY_COMPLEX,1.0,muestra_color,1,LINE_8);  //Mostrar texto con tipo de muestra (no necesario)

    imshow( "image", img );
    setMouseCallback( "image", onMouse, 0 ); //esperar por evento de mouse


    for(;;)
    {
        char c = (char)waitKey(0);

        if( c == 27 )
            break; //Si presiona ESC se finaliza el programa

        if( c == 'h' )
            help(); //Mostrar ayuda

        if( c == 'c' )
        {

            meanStdDev(img0,mean,stddev,markerMask);  //Mostrar calculo de promedio y stddev
            cout<<"Promedio[B,G,R]"<<mean<<endl;
            cout<<"Stddev[B,G,R]"<<stddev<<endl;
        }
        if( c == 'g' )
        {
            meanStdDev(img0,mean,stddev,markerMask);  //Guardar los resultados en la muestra
            string pos_file="trainingdata_positives.csv";  //definir archivos donde se van a guardar
            string neg_file="trainingdata_negatives.csv";
            if(muestra_tipo=="Positiva")
            {
                positivos.open(pos_file.c_str(),ios::app);
                positivos << mean[0]<<","<< mean[1]<<","<< mean[2]<<","<<stddev[0]<<","<<stddev[1]<<","<<stddev[2]<<endl;
                positivos.close();
                cout<<"Guardado en: \""<<pos_file<<"\""<<endl;
            }
            else
            {
                negativos.open(neg_file.c_str(),ios::app);
                negativos << mean[0]<<","<< mean[1]<<","<< mean[2]<<","<<stddev[0]<<","<<stddev[1]<<","<<stddev[2]<<endl;
                negativos.close();
                cout<<"Guardado en: \""<<neg_file<<"\""<<endl;
            }


        }
        if( c == 'r' )
        {
            markerMask = Scalar::all(0);  //Restaurar imagen
            img0.copyTo(img);
            rectangle(img,Point2d(28,3),Point2d(170,38),Scalar(0,0,0),-1,LINE_8,0);
            putText(img,muestra_tipo,Point2d(30,30),FONT_HERSHEY_COMPLEX,1.0,muestra_color,1,LINE_8);
            imshow( "image", img );
        }

        if( c == 't')
        {
            if (muestra_tipo=="Positiva"){ //alternar tipo de muestra (afecta el archivo donde se va a guardar)
                muestra_tipo="Negativa";
                muestra_color=Scalar(0,0,255);
            }else{
                muestra_tipo="Positiva";
                muestra_color=Scalar(0,255,0);
            }
            rectangle(img,Point2d(28,3),Point2d(170,38),Scalar(0,0,0),-1,LINE_8,0);
            putText(img,muestra_tipo,Point2d(30,30),FONT_HERSHEY_COMPLEX,1.0,muestra_color,1,LINE_8);
            imshow("image",img);
        }
    }

    return 0;
}
