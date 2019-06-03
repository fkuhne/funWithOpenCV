/*

  Programa simples de captura de video.

  Baseado em exemplos da documentacao do OpenCV.

  Se informado um argumento, vai abrir o arquivo correspondente (JPG ou AVI).
    Caso nenhum argumento seja informado, vai abrir o dispositivo padrao de
    captura (estou assumindo que e o /dev/video0, mas nao testei isso).

  Felipe Kuhne, 20/01/2016.

  Licenca: GPL v3.

*/

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

const String windowName="Video Capture";

int main(int argc, char **argv)
{
  VideoCapture capture;

  // Read the video stream
  if(argc == 1)
    // If no args passed, open /dev/video0
    capture.open(0);
  else
    // Otherwise, open the file
    capture.open(string(argv[1]));

  if(!capture.isOpened())
  {
    cout << "Could not open video stream. Exiting!" << endl;
    return -1;
  }

  // Get the frame rate
  int fps = (int)capture.get(CV_CAP_PROP_FPS);
  cout << "Frame rate = " << fps << " FPS." << endl;
  int frameInterval = 1000 / fps;

  // Create a window to later display the video frames
  namedWindow(windowName, WINDOW_KEEPRATIO);

  // Cature the video frames
  //   What is the frame rate? Can I choose it or it is automatic?
  while(true)
  {
    Mat frame;

    //capture.read(frame);
    capture >> frame;

    if(frame.empty())
    {
      cout << "Could not capture frame. Exiting!" << endl;
      break;
    }

    // Show the frame
    imshow(windowName, frame);

    // Try to mimic the video frame rate. Of course that this is not exact.
    // Wait for ESC press within the video frame rate
    int c = waitKey(frameInterval);
    if((char)c == 27)
      break;

  }

  // Wait for one more key stroke to exit application
  waitKey(0);

  return 0;
}
