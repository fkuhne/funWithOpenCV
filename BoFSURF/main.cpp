#include <iostream>
#include <stdio.h>
#include <vector>
#include "dirent.h"
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/ml.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;

typedef struct {
	int index;
	string name;
} classId;

int getClassId(const char *dirName, classId& classItem)
{
	/* Check for the existence of dash in the directory name. */
	if((dirName[0] != '-') && (dirName[1] != '-'))
	{
		cerr << "Incorrect syntax for directory name." << endl;
		return EXIT_FAILURE;
	}

	/* Get pointer to the  dash. */
	char *p = strchr(dirName, '-');
	if(p == NULL)
	{
		cerr << "Incorrect syntax for directory name." << endl;
		return EXIT_FAILURE;
	}

	/* Get the dash position so that it can be used to extract the index (before
	 * the dash) and the name (after the dash). */
	int dashPosition = p - dirName;
	classItem.index = stoi(string(dirName).substr(0, dashPosition));
	classItem.name = string(dirName).substr(dashPosition+1);

	return EXIT_SUCCESS;
}

/* Build a vector of classId's structures, from the subdirectories
 * present inside the samples directory.
 * The expected syntax for each subdirectory is "index-name". */
int buildClassVector(const string& samplesDirName, vector<classId>& classVector)
{
	classVector.clear();

	/* Opening directory containing the list of directories, each one with
	 *   a class of object. */
	DIR *samplesDir = NULL;
	if((samplesDir = opendir(samplesDirName.c_str())) == NULL)
	{
		cerr << "Error while opening '" << samplesDirName << "' directory." << endl;
		return EXIT_FAILURE;
	}

	struct dirent *samplesDirInfo = NULL;
	while((samplesDirInfo = readdir(samplesDir)) != NULL)
	{
		/* Skip '.' and '..'. */
		if(!strncmp(samplesDirInfo->d_name, ".", strlen(samplesDirInfo->d_name)) ||
		   !strncmp(samplesDirInfo->d_name, "..", strlen(samplesDirInfo->d_name)))
			continue;

		classId classItem;
		if(getClassId(samplesDirInfo->d_name, classItem) != EXIT_SUCCESS)
			continue;

		classVector.push_back(classItem);
	}

	if(classVector.empty())
	{
		cerr << "No class detected." << endl;
		return EXIT_FAILURE;
	}

	/*cout << "testing..." << endl;

	for (vector<classId>::iterator it = classVector.begin() ; it != classVector.end(); ++it)
		cout << "Class = " << it->name << ", Index = " << it->index << "." << endl;*/

	return EXIT_SUCCESS;
}

/* This function receives the feature detector, the descriptor extractor and the
 * name of the directory that contains all images that will be used to build the
 * vocabulary. It will the process everything - it will take a while! - and
 * record the vocabulary in an XML file. */
int buildVocabulary(const Ptr<FeatureDetector>& featureDetector,
										const Ptr<DescriptorExtractor>& descriptorExtractor,
										const string& samplesDirName)
{
	/* Opening directory containing the list of directories, each one with
	 *   a class of object. */
	DIR *samplesDir = NULL;
	if((samplesDir = opendir(samplesDirName.c_str())) == NULL)
	{
		cerr << "Error while opening '" << samplesDirName << "' directory." << endl;
		return EXIT_FAILURE;
	}

	cout << "Building vocabulary..." << endl;

	/* Each row of this matrix corresponds to the descriptors of an image file. */
	Mat featuresUnclustered;
	/* This is a simple counter. I have to check if its value is exactly the same
	 * as featuresUnclustered.rows(). */
	//int vocabularySize = 0;

	/* Let us now iterate over the directories, containing the image files of all
	 * objects that we will detect later. To facilitate the identification of the
	 * object class, each directory begins with a number, which corresponds
	 * exactly to the object class. */
	struct dirent *samplesDirInfo = NULL;
	while((samplesDirInfo = readdir(samplesDir)) != NULL)
	{
		/* Skip . and .., present on Unix systems. */
		if(!strncmp(samplesDirInfo->d_name, ".", strlen(samplesDirInfo->d_name)) ||
		   !strncmp(samplesDirInfo->d_name, "..", strlen(samplesDirInfo->d_name)))
			continue;

		string dirName(samplesDirName + '/' + samplesDirInfo->d_name);
		cout << "Opening " << dirName << "." << endl;

		/* Now we have an inner loop: we have to iterate ober the files inside each
		 * directory. */
		DIR *filesDir = NULL;
		if((filesDir = opendir(dirName.c_str())) == NULL)
		{
			cerr << "Error while opening " << dirName << "." << endl;
			continue; /* Go to the next directory. */
		}

		struct dirent *filesDirInfo = NULL;
		// Reads all image files in the directory
		while((filesDirInfo = readdir(filesDir)) != NULL)
		{
			/* Skip . and .., present on Unix systems. */
			if(!strncmp(filesDirInfo->d_name, ".", strlen(filesDirInfo->d_name)) ||
			   !strncmp(filesDirInfo->d_name, "..", strlen(filesDirInfo->d_name)))
				continue;

			cout << "Extracting features from file " << filesDirInfo->d_name
			     << "... " << flush;

			UMat image;
			string fileName(dirName + "/" + filesDirInfo->d_name);
			imread(fileName, IMREAD_GRAYSCALE).copyTo(image);
			if(image.empty())
			{
				cout << " SKIPPING." << endl;
				continue;
			}

			/* This is the usual process of feature detection and
			 * descriptor extraction. We are currently using SURF, but other
			 * might be used. */
			vector<KeyPoint> keyPoints;
			featureDetector->detect(image.getMat(ACCESS_READ), keyPoints);
			UMat _descriptors;
			Mat descriptors = _descriptors.getMat(ACCESS_RW);
			descriptorExtractor->compute(image.getMat(ACCESS_READ), keyPoints,
			                             descriptors);
			if(!descriptors.empty())
			{
				featuresUnclustered.push_back(descriptors);
				cout << "OK" << endl;
				//vocabularySize += descriptors.rows;
			}
		} // End of images loop

		closedir(filesDir);
	} // End of directory loop

	closedir(samplesDir);

	/* OK, so now we have a bunch of descriptors. We need then to build a
	 * vocabulary, or dictionary, containing a "bag of words" (BoW), of "bag of
	 * features" (BoF). */
	//define Terminate Criteria
	TermCriteria tc(CV_TERMCRIT_ITER, 100, 1e-6);
	//retries number default value from class
	int retries = 3;
	//necessary flags - default value from class
	int flags = KMEANS_PP_CENTERS;
	//Create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainer(2000, tc, retries, flags);

	cout << "Clustering vocabulary... " << flush;
	//cluster the feature vectors
	Mat vocabulary = bowTrainer.cluster(featuresUnclustered);

	cout << "OK." << endl;

  cout << "vocabulary.rows = " << vocabulary.rows << endl;

	//store the vocabulary
	FileStorage fs("vocabulary.xml", FileStorage::WRITE);
	fs << "vocabulary" << vocabulary;
	fs.release();

	return EXIT_SUCCESS;
}

int trainSVMClassifier(const Ptr<FeatureDetector>& featureDetector,
	                     Ptr<BOWImgDescriptorExtractor>& bowDE,
											 const string& samplesDirName)
{
	DIR *samplesDir = NULL;
	if((samplesDir = opendir(samplesDirName.c_str())) == NULL)
	{
		cerr << "Error while opening directory samples." << endl;
		return EXIT_FAILURE;
	}

	cout << "Training the SVM classifier." << endl;

	Mat labels(0, 1, CV_32FC1);
	Mat trainingData(0, bowDE->descriptorSize(), CV_32FC1);

	struct dirent *samplesDirInfo = NULL;
	while((samplesDirInfo = readdir(samplesDir)) != NULL)
	{
		if(!strncmp(samplesDirInfo->d_name, ".", strlen(samplesDirInfo->d_name)) ||
		   !strncmp(samplesDirInfo->d_name, "..", strlen(samplesDirInfo->d_name)))
			continue;

		// each of the classes is a directory name containing images belonging to
		// that class.
		string dirName(samplesDirName + "/" + samplesDirInfo->d_name);

		DIR *filesDir = NULL;
		if((filesDir = opendir(dirName.c_str())) == NULL)
		{
			cerr << "Error while opening " << dirName << "." << endl;
			continue;
		}

		// Get the number in the beggining of the directory name
		stringstream ss(samplesDirInfo->d_name);
		string token;
		getline(ss, token, '-');
		if(token.empty())
			continue;
		ss.str(token);
		int classIndex;
		ss >> classIndex;

		cout << "Opening " << dirName << ". Class Index = " << classIndex << "."
		     << endl;

		struct dirent *filesDirInfo = NULL;
		while((filesDirInfo = readdir(filesDir)) != NULL)
		{
			if(!strncmp(filesDirInfo->d_name, ".", strlen(filesDirInfo->d_name)) ||
			   !strncmp(filesDirInfo->d_name, "..", strlen(filesDirInfo->d_name)))
				continue;

			string fileName(dirName + "/" + filesDirInfo->d_name);
			UMat image;
			imread(fileName, IMREAD_GRAYSCALE).copyTo(image);
			if(image.empty())
			{
				cout << "Couldn't load " << fileName << "." << endl;
				continue;
			}

			cout << "Extracting features from file " << filesDirInfo->d_name << "..."
			     << flush;

			vector<KeyPoint> keyPoints;
			featureDetector->detect(image.getMat(ACCESS_READ), keyPoints);

			UMat _descriptors;
			Mat descriptors = _descriptors.getMat(ACCESS_RW);
			/* Extract the actual bag of word descriptors. Using this function you
			 * create descriptors which you then gather into a matrix which serves as
			 * the input for the classifier functions. */
			bowDE->compute(image.getMat(ACCESS_READ), keyPoints, descriptors);
			if(!descriptors.empty())
			{
				trainingData.push_back(descriptors);
				labels.push_back(classIndex);
			}

			cout << " OK" << endl;
		} // End of images loop inside its directory

		closedir(filesDir);

	} // end of directory loop

	closedir(samplesDir);

	// Now train the SVM classifier:

	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);

	ParamGrid c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid;
	c_grid = SVM::getDefaultGrid(SVM::C);
	gamma_grid = SVM::getDefaultGrid(SVM::GAMMA);
	p_grid = SVM::getDefaultGrid(SVM::P);
	p_grid.logStep = 0;
	nu_grid = SVM::getDefaultGrid(SVM::NU);
	nu_grid.logStep = 0;
	coef_grid = SVM::getDefaultGrid(SVM::COEF);
	coef_grid.logStep = 0;
	degree_grid = SVM::getDefaultGrid(SVM::DEGREE);
	degree_grid.logStep = 0;

	svm->trainAuto(TrainData::create(trainingData, ROW_SAMPLE, labels), 10,
	               c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid);

	cout << "Saving SVM classifier..." << flush;
	svm->save("classifier.xml");
	cout << " OK." << endl;

	return EXIT_SUCCESS;
}

/* TODO: Have to implement the calls to this function, inside a for iterating
 * over the classId vector. */
int trainSVMClassifierForClass(const Ptr<FeatureDetector>& featureDetector,
	                     Ptr<BOWImgDescriptorExtractor>& bowDE,
											 const string& samplesDirName, const classId& classItem)
{
	DIR *samplesDir = NULL;
	if((samplesDir = opendir(samplesDirName.c_str())) == NULL)
	{
		cerr << "Error while opening directory samples." << endl;
		return EXIT_FAILURE;
	}

	cout << "Training the SVM classifier for class " << classItem.name << " ("
			 << classItem.index << ")." << endl;

	Mat labels(0, 1, CV_32FC1);
	/* TODO: have to double check if bowDE->descriptorSize() is equal to
	 * vocabulary.rows. */
	Mat trainingData(0, bowDE->descriptorSize(), CV_32FC1);

	struct dirent *samplesDirInfo = NULL;
	while((samplesDirInfo = readdir(samplesDir)) != NULL)
	{
		if(!strncmp(samplesDirInfo->d_name, ".", strlen(samplesDirInfo->d_name)) ||
		   !strncmp(samplesDirInfo->d_name, "..", strlen(samplesDirInfo->d_name)))
			continue;

		// each of the classes is a directory name containing images belonging to
		// that class.
		string dirName(samplesDirName + "/" + samplesDirInfo->d_name);

		DIR *filesDir = NULL;
		if((filesDir = opendir(dirName.c_str())) == NULL)
		{
			cerr << "Error while opening " << dirName << "." << endl;
			continue;
		}

		classId thisClassItem;
		if(getClassId(samplesDirInfo->d_name, thisClassItem) != EXIT_SUCCESS)
			continue;

		cout << "Opening " << dirName << ". Class Index = " << thisClassItem.index << "."
		     << endl;

		struct dirent *filesDirInfo = NULL;
		while((filesDirInfo = readdir(filesDir)) != NULL)
		{
			if(!strncmp(filesDirInfo->d_name, ".", strlen(filesDirInfo->d_name)) ||
			   !strncmp(filesDirInfo->d_name, "..", strlen(filesDirInfo->d_name)))
				continue;

			string fileName(dirName + "/" + filesDirInfo->d_name);
			UMat image;
			imread(fileName, IMREAD_GRAYSCALE).copyTo(image);
			if(image.empty())
			{
				cout << "Couldn't load " << fileName << "." << endl;
				continue;
			}

			cout << "Extracting features from file " << filesDirInfo->d_name << "..."
			     << flush;

			vector<KeyPoint> keyPoints;
			featureDetector->detect(image.getMat(ACCESS_READ), keyPoints);

			UMat _descriptors;
			Mat descriptors = _descriptors.getMat(ACCESS_RW);
			/* Extract the actual bag of word descriptors. Using this function you
			 * create descriptors which you then gather into a matrix which serves as
			 * the input for the classifier functions. */
			bowDE->compute(image.getMat(ACCESS_READ), keyPoints, descriptors);
			if(!descriptors.empty())
			{
				trainingData.push_back(descriptors);
				/* Here's the thing: build one classifier for each class. For the images
				 * belonging to the chosen class, the label equals 1. For all others,
				 * label equals -1. It givez us a collection of binary classifiers. */
				labels.push_back((thisClassItem.index == classItem.index) ? 1 : -1);
			}

			cout << " OK: " << labels.row(labels.rows-1) << endl;
		} // End of images loop inside its directory

		closedir(filesDir);

	} // end of directory loop

	closedir(samplesDir);

	// Now train the SVM classifier:

	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);

	ParamGrid c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid;
	c_grid = SVM::getDefaultGrid(SVM::C);
	gamma_grid = SVM::getDefaultGrid(SVM::GAMMA);
	p_grid = SVM::getDefaultGrid(SVM::P);
	p_grid.logStep = 0;
	nu_grid = SVM::getDefaultGrid(SVM::NU);
	nu_grid.logStep = 0;
	coef_grid = SVM::getDefaultGrid(SVM::COEF);
	coef_grid.logStep = 0;
	degree_grid = SVM::getDefaultGrid(SVM::DEGREE);
	degree_grid.logStep = 0;

	svm->trainAuto(TrainData::create(trainingData, ROW_SAMPLE, labels), 10,
	               c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid);

	cout << "Saving SVM classifier..." << flush;
	string classifierName = string("classifier" + to_string(classItem.index) + ".xml.gz");
	svm->save(classifierName);
	cout << " OK." << endl;

	return EXIT_SUCCESS;
}


int imagePredict(const Ptr<FeatureDetector>& featureDetector,
	               Ptr<BOWImgDescriptorExtractor>& bowDE, const string& fileName)
{
	cout << "Loading SVM classifier... " << flush;
	FileStorage fs("classifier.xml", FileStorage::READ);
	Ptr<SVM> svm = StatModel::load<SVM>("classifier.xml");
	cout << "OK" << endl;
  fs.release();

	UMat image;
	imread(fileName, IMREAD_GRAYSCALE).copyTo(image);
	if(image.empty())
	{
		cout << "Couldn't load " << fileName << "." << endl;
		return EXIT_FAILURE;
	}

	vector<KeyPoint> keyPoints;
	cout << "Detecting keypoints... " << flush;
	featureDetector->detect(image.getMat(ACCESS_READ), keyPoints);
	cout << "OK." << endl;

	UMat _descriptors;
	cout << "Extracting descriptors... " << flush;
	Mat descriptors = _descriptors.getMat(ACCESS_RW);
	bowDE->compute(image.getMat(ACCESS_READ), keyPoints, descriptors);
	cout << "OK." << endl;
	if(!descriptors.empty())
	{
		cout << "Computing response: " << flush;
		float response = svm->predict(descriptors);
		cout << response << endl;
	}

	return EXIT_SUCCESS;
}


int main(int argc, char* argv[])
{
	const char* keys =
		"{ h help       | | print help message }"
		"{ v vocabulary | | build vocabulary }"
		"{ c classifier | | train classifier }"
		"{ p predict    | | predict the class of an image }"
		"{ f filename   | | image to be identified }";

	CommandLineParser cmd(argc, argv, keys);
	if (cmd.has("help") || argc < 2)
	{
		cout << "Usage: "<< argv[0] << "[options]" << endl;
		cout << "Available options:" << endl;
		cmd.printMessage();
		return EXIT_SUCCESS;
	}

	Ptr<Feature2D> surfDetector = SURF::create(1600);
	Ptr<DescriptorExtractor> surfExtractor = surfDetector;
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
	//Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	Ptr<BOWImgDescriptorExtractor> bowDE;

	bowDE = makePtr<BOWImgDescriptorExtractor>(surfDetector, matcher);

	if(cmd.has("vocabulary"))
	{
		FileStorage fs("vocabulary.xml", FileStorage::READ);
		if(fs.isOpened())
		{
			fs.release();
			cout << "Vocabulary already exists. Remove the file vocabulary.xml to \
			         create it again." << endl;
		}
		else
		{
			string samplesDir(cmd.get<string>("v"));
			buildVocabulary(surfDetector, surfExtractor, samplesDir);
		}

		return EXIT_SUCCESS;
	}

	if(cmd.has("classifier"))
	{
		FileStorage fs("vocabulary.xml", FileStorage::READ);
		if(fs.isOpened())
		{
			cout << "Loading vocabulary... " << flush;
			Mat vocabulary;
			fs["vocabulary"] >> vocabulary;
			cout << "OK." << endl;
			cout << "vocabulary.rows = " << vocabulary.rows << endl;
			fs.release();

			//Set the dictionary with the vocabulary we created in the first step
			bowDE->setVocabulary(vocabulary);
			cout << "bowDE->descriptorSize() = " << bowDE->descriptorSize() << endl;

			string samplesDir(cmd.get<string>("c"));

			vector<classId> classList;
			buildClassVector(samplesDir, classList);

			for (vector<classId>::iterator it = classList.begin() ; it != classList.end(); ++it)
				trainSVMClassifierForClass(surfDetector, bowDE, samplesDir, *it);

			//trainSVMClassifier(surfDetector, bowDE, samplesDir);
		}
		else
			cout << "Vocabulary doesn't exist yet. Run " << argv[0] << " -v first."
			     << endl;

		return EXIT_SUCCESS;
	}

	if(cmd.has("predict"))
	{
		FileStorage fs("vocabulary.xml", FileStorage::READ);
		if(!fs.isOpened())
		{
			cout << "Vocabulary doesn't exist yet. Run " << argv[0] << " -v first."
			     << endl;
			return EXIT_SUCCESS;
		}
		else
		{
			cout << "Loading vocabulary... " << flush;
			Mat vocabulary;
			fs["vocabulary"] >> vocabulary;
			cout << "OK." << endl;
			fs.release();
			//Set the dictionary with the vocabulary we created in the first step
			bowDE->setVocabulary(vocabulary);
		}

		fs.open("classifier.xml", FileStorage::READ);
		if(!fs.isOpened())
    {
			cout << "No SVM classifier file found. Run " << argv[0] << " -c first."
			     << endl;
			return EXIT_SUCCESS;
	  }
		fs.release();

		string fileName = cmd.get<string>("p");
		imagePredict(surfDetector, bowDE, fileName);
	}

	return EXIT_SUCCESS;
}
