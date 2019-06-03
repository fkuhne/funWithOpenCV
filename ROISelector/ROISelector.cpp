#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "ROISelector.h"

using namespace std;
using namespace cv;

const int min_roi_size_x = 300;
const int min_roi_size_y = 300;

/* Class constructor */
ROISelector::ROISelector()
{
	selectObject = false;
	done = false;
	selection = Rect(0,0,0,0);
	origin = Point(0,0);
	image = Mat(1, 1, CV_64F, 0.0);
	ROIWindow = string("ROI Selector");
	namedWindow(ROIWindow.c_str(), 0);
}

void ROISelector::registerMyMouseCallback(int event, int x, int y, int, void* userdata)
{
	ROISelector* roi = reinterpret_cast<ROISelector*>(userdata);
	roi->onMouse(event, x, y);
}

void ROISelector::onMouse(int event, int x, int y)
{
	if (selectObject && !done)
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = abs(x - origin.x);
		selection.height = abs(y - origin.y);
		selection &= Rect(0, 0, image.cols, image.rows);
	}

//	cout << x << "," << y << endl;

	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
	{
		origin = Point(x, y);
		selection = Rect(x, y, 0, 0);
		selectObject = true;
		cout << "origin: " << origin.x << "," << origin.y << endl;
		break;
	}
	case CV_EVENT_LBUTTONUP:
	{
		if (selectObject)
		{
			if (selection.width > min_roi_size_x && selection.height > min_roi_size_y)
			{
				done = true;
				cout << "selection: " << selection.width << "," << selection.height << endl;
                
                /* This little code can be removed when we don't need it anymore. */
                imshow("Selected image", image(selection));
                char k = waitKey(2000);
            }

			else
			{
				selection = Rect(0, 0, 0, 0);
				selectObject = false;
			}
		}
		break;
	}
	}
}

void ROISelector::drawROIRectangle()
{
	if (selection.width > min_roi_size_x && selection.height > min_roi_size_y)
		rectangle(image, Point(selection.x - 1, selection.y - 1),
			Point(selection.x + selection.width + 1, selection.y + selection.height + 1), Scalar(0, 255, 0));

	else if (selection.width > 0 && selection.height > 0)
		rectangle(image, Point(selection.x - 1, selection.y - 1),
			Point(selection.x + selection.width + 1, selection.y + selection.height + 1), Scalar(0, 0, 255));

	imshow(ROIWindow.c_str(), image);

	return;

}

int ROISelector::ROISelectorLoop(const Mat& thisImage)
{
	if (thisImage.empty())
	{
		cout << "Image empty!" << endl;
		return -1;
	}

	while (!done)
	{
		/* copy the imagem to the class image frame, keeping the original intact,
		 * since at each loop the rectangle need to be redrawn. */
		thisImage.copyTo(image);

		/* Draw the rectangle at the current image frame, according to mouse events. */
		drawROIRectangle();

		char k = waitKey(50);
		if (k == 27) // ESC
			return 1;
	}

	return 0;
}

int ROISelector::ROISelectorLoop(VideoCapture& cap)
{
	while (!done)
	{
		/* Get current frame from the camera. */
		cap >> image;

		if (image.empty())
			return -1;

		/* Draw the rectangle at the current image frame, according to mouse events. */
		drawROIRectangle();

		char k = waitKey(50);
		if (k == 27) // ESC
            return 1;
	}

	return 0;
}

int ROISelector::set(const string& fileName)
{
	Mat thisImage;
	imread(fileName, CV_LOAD_IMAGE_COLOR).copyTo(thisImage);
	if (thisImage.empty())
	{
		cout << "Couldn't load " << fileName << endl;
		return -1;
	}

	cout << "Opening " << fileName << endl;

	/* Register the mouse callback, which is the function that will be called when
	* some mouse event happens. */
	setMouseCallback(ROIWindow.c_str(), registerMyMouseCallback, this);

	/* Start the selection loop. */
	return ROISelectorLoop(thisImage);
}

int ROISelector::set(const int camId)
{
	/* Open the video feed. */
	VideoCapture cap(camId);
	if (!cap.isOpened())
	{
		cout << "Couldn't open camera " << camId << endl;
		return -1;
	}

	/* Register the mouse callback, which is the function that will be called when
	* some mouse event happens. */
	setMouseCallback(ROIWindow.c_str(), registerMyMouseCallback, this);

	/* Start the selection loop. */
	return ROISelectorLoop(cap);
}

Rect& ROISelector::get()
{
	/* Returns the selected rectangle. This variable will always exist so we don't
	 * need to test it. */
	return selection;
}
