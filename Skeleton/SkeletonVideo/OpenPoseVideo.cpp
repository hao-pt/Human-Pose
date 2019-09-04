#include <iostream>
#include "opencv2/dnn/dnn.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::dnn;

#define MPI

#ifdef MPI
const int POSE_PAIRS[14][2] =
{
	{0,1}, {1,2}, {2,3},
	{3,4}, {1,5}, {5,6},
	{6,7}, {1,14}, {14,8}, {8,9},
	{9,10}, {14,11}, {11,12}, {12,13}
};
// Network specification and weights file
string protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt";
string weightsFile = "pose/mpi/pose_iter_160000.caffemodel";
// Number of keypoints
int nPoints = 15;
#endif

#ifdef COCO
const int POSE_PAIRS[17][2] =
{
	{1,2}, {1,5}, {2,3},
	{3,4}, {5,6}, {6,7},
	{1,8}, {8,9}, {9,10},
	{1,11}, {11,12}, {12,13},
	{1,0}, {0,14},
	{14,16}, {0,15}, {15,17}
};
// Network specification and weights file
string protoFile = "pose/coco/pose_deploy_linevec.prototxt";
string weightsFile = "pose/coco/pose_iter_440000.caffemodel";
// Number of keypoints
int nPoints = 18;
#endif

int main(int argc, char **argv)
{

	cout << "USAGE : ./openpose <VideoFile> " << endl;
	// Default videoFile
	string videoFile = "sample_video.mp4";
	// Take arguments from commmand line
	if (argc == 2)
	{
		videoFile = argv[1];
	}

	// Frame size and Threshold
	int inWidth = 368;
	int inHeight = 368;
	float thresh = 0.01;
	
	// Start Read Video
	cv::VideoCapture cap(0);

	// Check if Open success or not
	if (!cap.isOpened())
	{
		cerr << "Unable to connect to camera" << endl;
		return 1;
	}

	
	Mat frame, frameCopy;
	// Get frame size
	int frameWidth = cap.get(CAP_PROP_FRAME_WIDTH);
	int frameHeight = cap.get(CAP_PROP_FRAME_HEIGHT);

	// Init video writer
	//VideoWriter video("Output-Skeleton.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(frameWidth, frameHeight));

	// Read pretrained Caffe model
	Net net = readNetFromCaffe(protoFile, weightsFile);
	// Flag to process frame or not
	// this helps reduce the number of frames to process
	bool isProcess = true;
	double t = 0;
	while (waitKey(1) < 0)
	{
		// start tick each single frame
		double t = (double)cv::getTickCount();

		// Read frame
		cap >> frame;
		//// Resize frame of video to 1 / 4 size for faster processing
		//resize(frame, frame, Size(0, 0), 0.5, 0.5);
		frameCopy = frame.clone();
		
		if (isProcess) {
			// Put to
			Mat inpBlob = blobFromImage(frame, 
				1.0 / 255, //scale
				Size(inWidth, inHeight), // blob size
				Scalar(0, 0, 0), // mean substraction
				false, false); // swapRB and crop

			// Feed inpBlob through network
			net.setInput(inpBlob);
			Mat output = net.forward();
			/* Output
			1. The first dimension is the image ID (if feeding more than 1 image through network),
			2. The second dimension indicates the index of a keypoint.
			The model produces Confidence Maps and Part Affinity maps which are all concatenated.
			For COCO model it consists of 57 parts – 18 keypoint confidence Maps + 1 background + 19*2 Part Affinity Maps.
			Similarly, for MPI, it produces 44 points.
			We will be using only the first few points which correspond to Keypoints.
			3. The third dimension is the height of the output map.
			4. The fourth dimension is the width of the output map.
			*/
			// Size of output
			int H = output.size[2];
			int W = output.size[3];

			// find the position of the body parts
			vector<Point> points(nPoints);
			for (int n = 0; n < nPoints; n++)
			{
				// Probability map of corresponding body's part.
				Mat probMap(H, W, CV_32F, output.ptr(0, n));

				Point2f p(-1, -1);
				Point maxLoc;
				double prob;
				// Find the min value, min_pos, max_value as prob and max_pos of probMap
				minMaxLoc(probMap, 0, &prob, 0, &maxLoc);
				// Threshold to refine the false positive prediction
				if (prob > thresh)
				{
					p = maxLoc;
					// Scale p due to different size between original image and probMap
					p.x *= (float)frameWidth / W;
					p.y *= (float)frameHeight / H;

					// Draw keypoint as circle at point p with radius = 8 and color = (0, 255, 255), 
					// thickness = -1 meaning fill the circle
					circle(frameCopy, cv::Point((int)p.x, (int)p.y), 8, Scalar(0, 255, 255), -1);
					// Put keypoint number
					cv::putText(frameCopy, cv::format("%d", n), cv::Point((int)p.x, (int)p.y), cv::FONT_HERSHEY_COMPLEX, 1.1, cv::Scalar(0, 0, 255), 2);
				}
				// Save keypoint
				points[n] = p;
			}

			int nPairs = sizeof(POSE_PAIRS) / sizeof(POSE_PAIRS[0]);

			// Draw skeleton
			for (int n = 0; n < nPairs; n++)
			{
				// lookup 2 connected body/hand parts
				Point2f partA = points[POSE_PAIRS[n][0]];
				Point2f partB = points[POSE_PAIRS[n][1]];

				if (partA.x <= 0 || partA.y <= 0 || partB.x <= 0 || partB.y <= 0)
					continue;

				line(frame, partA, partB, Scalar(0, 255, 255), 8);
				circle(frame, partA, 8, Scalar(0, 0, 255), -1);
				circle(frame, partB, 8, Scalar(0, 0, 255), -1);
			}

			t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
			cv::putText(frame, cv::format("time taken = %.2f sec", t), cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, .8, cv::Scalar(255, 50, 0), 2);
			
			// Display keypoint and skeleton
			// imshow("Output-Keypoints", frameCopy);
			imshow("Output-Skeleton", frame);

			// Write file
			//video.write(frame);
		}

		// flip the flag
		isProcess = (~isProcess);
	}
	// When everything done, release the video capture and write object
	cap.release();
	//video.release();

	return 0;
}