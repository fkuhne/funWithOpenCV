#include <iostream>
#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

const int GOOD_PTS_MAX = 300;
const float GOOD_PORTION = 0.2f;

static Mat drawGoodMatches(
	const Mat& img1,
	const Mat& img2,
	const std::vector<KeyPoint>& keypoints1,
	const std::vector<KeyPoint>& keypoints2,
	std::vector<DMatch>& matches,
	std::vector<Point2f>& scene_corners_
)
{
	//-- Sort matches and preserve top 10% matches
	std::sort(matches.begin(), matches.end());
	std::vector< DMatch > good_matches;
	double minDist = matches.front().distance;
	double maxDist = matches.back().distance;

	const int ptsPairs = std::min(GOOD_PTS_MAX, (int)(matches.size() * GOOD_PORTION));
	for (int i = 0; i < ptsPairs; i++)
	{
		good_matches.push_back(matches[i]);
	}
	std::cout << "\nMax distance: " << maxDist << std::endl;
	std::cout << "Min distance: " << minDist << std::endl;
	std::cout << "Found " << matches.size() << " matches." << std::endl;

	std::cout << "Calculating homography using " << ptsPairs << " point pairs." << std::endl;

	// drawing the results
	Mat img_matches;

	drawMatches(img1, keypoints1, img2, keypoints2,
		good_matches, img_matches
		,Scalar::all(-1), Scalar::all(-1),
		std::vector<char>(), /*DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS*/DrawMatchesFlags::DRAW_RICH_KEYPOINTS
	);

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints1[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}
	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point(0, 0);
	obj_corners[1] = Point(img1.cols, 0);
	obj_corners[2] = Point(img1.cols, img1.rows);
	obj_corners[3] = Point(0, img1.rows);
	std::vector<Point2f> scene_corners(4);

	Mat H = findHomography(obj, scene, RANSAC);
	perspectiveTransform(obj_corners, scene_corners, H);

	scene_corners_ = scene_corners;

	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line(img_matches,
		scene_corners[0] + Point2f((float)img1.cols, 0), scene_corners[1] + Point2f((float)img1.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);
	line(img_matches,
		scene_corners[1] + Point2f((float)img1.cols, 0), scene_corners[2] + Point2f((float)img1.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);
	line(img_matches,
		scene_corners[2] + Point2f((float)img1.cols, 0), scene_corners[3] + Point2f((float)img1.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);
	line(img_matches,
		scene_corners[3] + Point2f((float)img1.cols, 0), scene_corners[0] + Point2f((float)img1.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);
	return img_matches;
}

////////////////////////////////////////////////////
// This program demonstrates the usage of SURF_OCL.
// use cpu findHomography interface to calculate the transformation matrix
int main(int argc, char* argv[])
{
	const char* keys =
		"{ h help     |                  | print help message  }"
		"{ l left     | capa_menor.JPG   | specify left image  }"
		"{ r right    | cena.JPG         | specify right image }"
		"{ o output   | SURF_output.jpg  | specify output save path }"
		"{ m cpu_mode |                  | run without OpenCL }";

	CommandLineParser cmd(argc, argv, keys);
	if (cmd.has("help"))
	{
		std::cout << "Usage: surf_matcher [options]" << std::endl;
		std::cout << "Available options:" << std::endl;
		cmd.printMessage();
		return EXIT_SUCCESS;
	}

	std::string leftName = cmd.get<std::string>("l");
	UMat img1;
	imread(leftName, IMREAD_GRAYSCALE).copyTo(img1);
	if (img1.empty())
	{
		std::cout << "Couldn't load " << leftName << std::endl;
		cmd.printMessage();
		return EXIT_FAILURE;
	}

	std::string rightName = cmd.get<std::string>("r");
	UMat img2;
	imread(rightName, IMREAD_GRAYSCALE).copyTo(img2);
	if (img2.empty())
	{
		std::cout << "Couldn't load " << rightName << std::endl;
		cmd.printMessage();
		return EXIT_FAILURE;
	}

	std::string outpath = cmd.get<std::string>("o");

	//instantiate detectors/matchers
	Ptr<Feature2D> surfDetector = SURF::create(800);
	Ptr<DescriptorExtractor> surfExtractor = surfDetector;
	Ptr<DescriptorMatcher> surfMatcher = DescriptorMatcher::create("BruteForce");
	//BFMatcher surfMatcher;

	std::vector<KeyPoint> keypoints1;
	surfDetector->detect(img1.getMat(ACCESS_READ), keypoints1);
	UMat _descriptors1;
	Mat descriptors1 = _descriptors1.getMat(ACCESS_RW);
	surfExtractor->compute(img1.getMat(ACCESS_READ), keypoints1, descriptors1);

	std::vector<KeyPoint> keypoints2;
	surfDetector->detect(img2.getMat(ACCESS_READ), keypoints2);
	UMat _descriptors2;
	Mat descriptors2 = _descriptors2.getMat(ACCESS_RW);
	surfExtractor->compute(img2.getMat(ACCESS_READ), keypoints2, descriptors2);

	std::cout << "FOUND " << keypoints1.size() << " keypoints on first image" << std::endl;
	std::cout << "FOUND " << keypoints2.size() << " keypoints on second image" << std::endl;

	std::vector<DMatch> matches;
	surfMatcher->match(descriptors1, descriptors2, matches);

	std::vector<Point2f> corner;
	Mat img_matches = drawGoodMatches(img1.getMat(ACCESS_READ), img2.getMat(ACCESS_READ), keypoints1, keypoints2, matches, corner);

	//-- Show detected matches

	namedWindow("surf matches", 0);
	imshow("surf matches", img_matches);
	imwrite(outpath, img_matches);

	waitKey(0);
	return EXIT_SUCCESS;
}
