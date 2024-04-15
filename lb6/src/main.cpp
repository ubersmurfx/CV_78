#include <opencv2/opencv.hpp>
#include <iostream>


using namespace cv;
using namespace std;

void laser(Mat &frame, Mat& mask) {
	cvtColor(frame, mask, COLOR_BGR2HSV);
	inRange(mask, Scalar(0, 0, 130), Scalar(90, 115, 255), mask);
}

int main(int argc, char** argv)  {
	bool debug = false;
	VideoCapture cap("../Video/calib_2.avi");
	//const string source = argv[1];

	if (!cap.isOpened())
 	{
 		cout << "Could not open the output video for write: " << endl;
 		return -1;
	}

	int frameWidth = cap.get(CAP_PROP_FRAME_WIDTH);
	int frameHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
	double frameRate = cap.get(CAP_PROP_FPS);
	
	
	int cent_x = frameWidth / 2;
	int cent_y = frameHeight / 2;
	int k = frameHeight / frameWidth;
	int camAngle = 74;
	double camAngleX = camAngle * CV_PI / 180;
	double focusX = (frameWidth / 2) / (tan(camAngleX / 2));

	if (debug) 
		{ cout << "camAngleX: " << camAngleX << "  focusX: " << focusX << endl; }

	if (debug) 
		{ cout << "w: " << frameWidth << "  h: " << frameHeight << "  fps: " << frameRate << endl; }


	VideoWriter mapperVid("mapperVid.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), frameRate, Size(frameWidth, frameHeight));

	Mat frame, mask, out;
	int letX, letY;

	while (cap.read(frame))
	{
		laser(frame, mask);
		out = Mat(mask.size(), CV_8UC3, Scalar(0, 0, 0));

		// Рисуем сетку на изображении out с шагом 15 пикселей
		if (out.cols > out.rows)
			for (int i = 0; i < out.cols; i += 15)
			{
				line(out, Point(i, 0), Point(i, out.rows), Scalar(255, 255, 255), 1);	
    			if (i < out.rows)
    			{
        			line(out, Point(0, i), Point(out.cols, i), Scalar(255, 255, 255), 1);
    			}
			}
		else
			for (int i = 0; i < out.rows; i += 15)
			{
				line(out, Point(0, i), Point(out.cols, i), Scalar(255, 255, 255), 1);
    			if (i < out.cols)
    			{
        			line(out, Point(i, 0), Point(i, out.rows), Scalar(255, 255, 255), 1);
    			}
			}

		// Обработка маски и отображение точек на изображении out
		for (int y = 0; y < out.rows; y++)
		{
    		for (int x = 0; x < out.cols; x++)
    		{
        		if (mask.at<uchar>(y, x) == 255)
        		{
            		int letY = clamp(int(5000000 / ((y - cent_y) * focusX)), 0, out.rows - 1);
            		circle(out, Point(x, letY), 2, Scalar(0, 255, 0));
        		}
    		}
		}
		mapperVid.write(out);
		if (debug) 
		{
			imshow("video", frame);
			imshow("find the laser", mask);
			imshow("mapper", out);
			waitKey(-1);
		}
	}
	return 0;
}