//
//  main.cpp
//  opencv-test1
//
//  Created by Felipe Kuhne on 5/20/16.
//  Copyright © 2016 Felipe Kuhne. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;


//hide the local functions in an anon namespace
namespace {
    void help(char** av)
    {
        cout << "Create a help text." << endl
        << "\texample: " << av[0] << " right%%02d.jpg" << endl;
    }
    
    Mat detectEdges(const Mat &frame)
    {
        Mat edges;
        cvtColor(frame, edges, CV_BGR2GRAY);
        GaussianBlur(edges, edges, Size(5,5), 2, 2);
  /*      
        threshold(edges, edges, 80, 255, 0);
        
        Mat element = getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1));
        erode(edges, edges, element);
        
        dilate(edges, edges, element);
   */     
        // pesquisar opening
        
        
        Canny(edges, edges, 0, 30, 3);
        
        //dilate(edges, edges, element);
        
        return edges;
    }
    
    int process(VideoCapture& capture)
    {
        string window_name = "video | esc to quit";
        namedWindow(window_name, /*WINDOW_KEEPRATIO*/CV_WINDOW_NORMAL); //resizable window;
        resizeWindow(window_name, 100, 100);
        
        Mat firstFrame;
        capture >> firstFrame;
        
        while(1)
        {
            Mat frame, edges, diff;
            capture >> frame;
            if (frame.empty())
                continue;
            
            
          //  absdiff(frame, firstFrame, diff);
            
            //imshow(window_name, diff);
            
            imshow(window_name, detectEdges(frame));
            
            char key = (char)waitKey(30); //delay N millis, usually long enough to display and capture input
            switch(key)
            {
                case 27: //escape key
                    return 0;
                default:
                    break;
            }
        }
        return 0;
    }
}

int main(int ac, char** av)
{
    if(ac == 1)
    {
        VideoCapture capture(0); // Open the default camera
        if(!capture.isOpened())
        {
            cerr << "Failed to open the video device!\n" << endl;
            help(av);
            return 1;
        }
        
        return process(capture);
    }
    else if(ac == 2)
    {
        /// Load the source image
        Mat src = imread(av[1], CV_LOAD_IMAGE_UNCHANGED);
        cout << "src size = " << src.rows << " rows, "<< src.cols << " cols." << endl;
        
        
        string window_name = "image | esc to quit";
        namedWindow(window_name, /*WINDOW_KEEPRATIO*/CV_WINDOW_NORMAL); //resizable window;

        double alpha = 0.5;
        double beta = 1.0 - alpha;
        Mat edges = detectEdges(src);
        cout << "edges size = " << src.rows << " rows, "<< src.cols << " cols." << endl;

        Mat dst;
        
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        
        findContours(edges, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
        
        Scalar color(255,255,255);
        for(int idx = 0; idx >= 0; idx = hierarchy[idx][0])
            drawContours(src, contours, idx, color, CV_FILLED, 8, hierarchy);
        
        imshow(window_name, src);
        
        char key = (char)waitKey(0);
        switch(key)
        {
            case 27: //escape key
                return 0;
            default:
                break;
        }
    }
    
}
