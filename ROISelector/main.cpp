#include <iostream>
#include <opencv2/imgproc.hpp>
#include "ROISelector.h"

#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	const char* keys =
		"{ h help   |  | print help message }"
		"{ c camera |  | camera as input }"
		"{ i image  |  | image as input }";

	CommandLineParser cmd(argc, argv, keys);
	if (cmd.has("help") || argc == 1)
	{
		cmd.printMessage();
		return EXIT_SUCCESS;
	}

	ROISelector roi;

	int returnCode = 0;

	if (cmd.has("camera"))
	{
		int camId = cmd.get<int>("c");
		cout << "camId = " << camId << endl;
		returnCode = roi.set(camId);
	}
	else if (cmd.has("image"))
	{
		string fileName = cmd.get<string>("i");
		cout << "File name = " << fileName << endl;
		returnCode = roi.set(fileName);
	}

	cout << "returnCode = " << returnCode << endl;

	if(returnCode == 0)
	{
		Rect sel = roi.get();
		cout << "imgRoi coordinates: " << sel.x << "," << sel.y << "," << sel.x + sel.width << "," << sel.y + sel.height << endl;
	}

	return 0;
}
