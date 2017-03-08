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
    /*Mat img2(74,75,CV_32FC1);
    cv::Mat img = cv::imread("training-data/1.jpg",0);
    img.convertTo(img2,CV_32FC1,1.0/255.0);
	if( img.empty() )
        return -1;
    cv::namedWindow( "Example1", cv::WINDOW_AUTOSIZE );
    cv::imshow( "Example1", img );
    cv::namedWindow( "Example2", cv::WINDOW_AUTOSIZE );
    cv::imshow( "Example2", img2 );
    Mat prueba[3];
    Mat training_pb;
    Mat trainingMat_32F[3];
    */
    Ptr<TrainData> datos_t;
    cout<<"Cargando datos ->";
    datos_t = cv::ml::TrainData::loadFromCSV("trainingdata_positives.csv",0,-1,-1);
    cout<<"Archivo Cargado ->"<<endl;


/*
    prueba[0]=imread("training-data/1.jpg",0).reshape(0,1);
    prueba[1]=imread("training-data/2.jpg",0).reshape(0,1);
    prueba[2]=imread("training-data/3.jpg",0).reshape(0,1);

    prueba[0].convertTo(trainingMat_32F[0],CV_32FC1,1.0/255.0);
    prueba[1].convertTo(trainingMat_32F[1],CV_32FC1,1.0/255.0);
    prueba[2].convertTo(trainingMat_32F[2],CV_32FC1,1.0/255.0);

    training_pb.push_back(trainingMat_32F[0]);
    training_pb.push_back(trainingMat_32F[1]);
    training_pb.push_back(trainingMat_32F[2]);
    imshow("Training Mat",training_pb);

    // Data for visual representation
    int width = 512, height = 512;
    Mat image = Mat::zeros(height, width, CV_8UC3);

    int labels[3][1] = {{1},{-1},{-1}};
    Mat labelsMat(3,1,CV_32SC1,labels);
*/
    // Train the SVM
    Ptr<SVM> svm = SVM::create();
    cout<<"Create -> ";
    waitKey(1);
    svm->setType(SVM::C_SVC);
    cout<<"setType -> ";
    waitKey(1);
    svm->setKernel(SVM::LINEAR);
    cout<<"setKernel ->";
    waitKey(1);
    //svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1));
    cout<<"setTermCriteria -> ";
    waitKey(1);
    //svm->train(datos_t, ROW_SAMPLE, labels_t);
    svm->trainAuto(datos_t,10,SVM::getDefaultGrid(SVM::C),SVM::getDefaultGrid(SVM::GAMMA),SVM::getDefaultGrid(SVM::P),SVM::getDefaultGrid(SVM::NU),SVM::getDefaultGrid(SVM::COEF),SVM::getDefaultGrid(SVM::DEGREE),false);
    cout<<"Trained!";

    // Testing predictor
    Mat img = imread("training-data/ur4.jpg"); //RM
    Mat markerMask;
    cvtColor(img, markerMask, COLOR_BGR2GRAY); //clonar img y copiar la imagen que guardará la mascara para calcular el promedio
    markerMask = Scalar::all(0);
    int i,j;
    for(i=0;i<img.rows;i++)
        for(j=0;j<img.cols;j++)
        {
            cout<<img.at<Vec3b>(i,j);

            /*if (predict((static_cast<Vec3b>img.at<Vec3b>(i,j))==1))
                markerMask.at<Vec3b>(i,j) = Scalar(255);
            else
                markerMask.at<Vec3b>(i,j) = Scalar(0);
            /*if(img0_mask.at<Vec3b>(i, j)[0]!=0&&img0_mask.at<Vec3b>(i, j)[1]!=0&&img0_mask.at<Vec3b>(i, j)[2]!=0)
            {
            cout<<"B: "<<static_cast<Vec3b>(img0_mask.at<Vec3b>(i,j))<<endl;
            }*/
        }


 //   predict_img.convertTo(predict_img,CV_32FC1);
  //  float res=svm->predict(predict_img);
   // if (res>0)
   //     cout<<endl<<"Real Madrid es: Ganador";
    //else
    //    cout<<endl<<"Real Madrid es: Perdedor";


}
