#ifndef _ROISELECTOR_H_
#define _ROISELECTOR_H_

#pragma once

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <stdio.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>

class ROISelector
{
public:
	/* Constructor */
	ROISelector();

	int set(const int camId);
	int set(const std::string& fileName);

	cv::Rect& get();

private:
	bool selectObject;
	bool done;
	cv::Rect selection;
	cv::Point origin;
	cv::Mat image;

	std::string ROIWindow;

	void onMouse(int event, int x, int y);
	static void registerMyMouseCallback(int event, int x, int y, int, void* userdata);

	void drawROIRectangle();
	int ROISelectorLoop(cv::VideoCapture& cap);
	int ROISelectorLoop(const cv::Mat& image);
};

#endif
