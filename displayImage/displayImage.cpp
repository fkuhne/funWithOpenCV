/*
  'Meu' primeiro programa para OpenCV.

  O meu esta entre aspas pois esse nao e exatamente 'meu' programa. Copiei ele
    do site do opencv: http://docs.opencv.org/2.4/doc/tutorials/introduction/
    linux_gcc_cmake/linux_gcc_cmake.html. Mudei algumas coisas so pra ficar mais
    ao meu gosto. Vou continuar trabalhando em cima deste exemplo pra ir
    aprendendo as APIs.
*/

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char** argv)
{

  if(argc != 2)
  {
    cout << "Usage:\n\t" << argv[0] << " <image path>" << endl << endl;
    return -1;
  }

  // Mat eh um container de imagem em forma de matriz
  cv::Mat image = cv::imread(argv[1], 1);

  if(image.empty())
  {
    cout << "No image data. << endl";
    return -1;
  }

//  cv::putText(image, argv[1], cv::Point(50,100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,200,200), 4);

  cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
  cv::imshow("Display Image", image);

  cv::waitKey(0);

  return 0;
}
