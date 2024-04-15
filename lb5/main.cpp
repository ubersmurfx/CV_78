#include <iostream>
#include <unistd.h>
#include "opencv2/aruco.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
     
using namespace std;
using namespace cv;

void arUcoMarkerGenerate(Ptr<aruco::Dictionary> dictionary, short int id, short int size, std::string path);
void arUcoBoardGenerate(Ptr<aruco::Dictionary> dictionary, short int row, short int col, float length, float sep, std::string path);
int cameraCalibration(Ptr<aruco::Dictionary> dictionary, short int row, short int col, float length, float sep, std::string paramsCam);
void markerDetector(Ptr<aruco::Dictionary> dict);
void cubaning(Mat &image, Vec3d rotV, Vec3d transV, float len);

Mat intrinsicMatrix, distCoeffs;


int cameraCalibration(
		Ptr<aruco::Dictionary> dictionary, 
		short int row, 
		short int col,
		float length,
		float sep, 
		std::string paramsCam
	) {
    	Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();
    	Ptr<aruco::GridBoard> markerBoard = aruco::GridBoard::create(
			row, 
			col, 
			length, 
			sep, 
			dictionary, 
			0
		);

		vector<std::string> images;
    	std::string vault = "./src/img/calibrate/*.jpg";
    	glob(vault, images);
		cout << images.size();
    	vector<vector<vector<Point2f>>> corners;
    	vector<vector<int>> ids;
    	cv::Size imageSize;

    for (int i = 0; i < images.size(); i++) {
        vector<vector<Point2f>> cur_corners, rej_corners;
        vector<int> cur_ids;
        Mat cur_image = imread(images[i]);
        imshow(std::to_string(i), cur_image);

        aruco::detectMarkers(cur_image, dictionary, cur_corners, cur_ids, params, rej_corners);
        corners.push_back(cur_corners);
        ids.push_back(cur_ids);
        imageSize = cur_image.size();
        aruco::drawDetectedMarkers(cur_image, cur_corners, cur_ids, Scalar(255, 0, 0));
        imshow(std::to_string(i) + "frame", cur_image);
        waitKey(0);
    }
    vector<vector<Point2f>> allCornersConcat, rej_candidates;
    vector<int> allIdsConcat, markerPointPerImage;
    markerPointPerImage.reserve(corners.size());

    for (size_t i = 0; i < corners.size(); i ++) {
        markerPointPerImage.push_back((int)corners[i].size());
        for (size_t j = 0; j < corners[i].size(); j++) {
            allCornersConcat.push_back(corners[i][j]);
            allIdsConcat.push_back(ids[i][j]);
        }
    }

	vector<Mat> rotV, transV;
    double reprError;
    reprError = aruco::calibrateCameraAruco(
		allCornersConcat, 
		allIdsConcat, 
		markerPointPerImage, 
		markerBoard, 
		imageSize, 
		intrinsicMatrix, 
		distCoeffs, 
		rotV, 
		transV, 
		0
	);


	FileStorage filestor("./src/cur_params.yaml", FileStorage::WRITE);
    if (filestor.isOpened()) {
        filestor << "intrinsicMatrix" << intrinsicMatrix;
        filestor << "distCoeffs" << distCoeffs;
	}
    return 0;
}

void cubaning(Mat &image, Vec3d rotV, Vec3d transV, float len) {
    vector<Point2f> imageP;
    vector<Point3f> worldPoint(8, Point3d(0, 0, 0));

    worldPoint[0] = Point3d(len/2, len/2, 0);
    worldPoint[1] = Point3d(len/2, -len/2, 0);
    worldPoint[2] = Point3d(-len/2, -len/2, 0);
    worldPoint[3] = Point3d(-len/2, len/2, 0);
    worldPoint[4] = Point3d(len/2, len/2, len);
    worldPoint[5] = Point3d(len/2, -len/2, len);
    worldPoint[6] = Point3d(-len/2, -len/2, len);
    worldPoint[7] = Point3d(-len/2, len/2, len);

    projectPoints(worldPoint, rotV, transV, intrinsicMatrix, distCoeffs, imageP);

    for (size_t i = 0; i < worldPoint.size(); i++) {
        putText(image, std::to_string(i), imageP[i], 1, 2, Scalar(0, 0, 0), 0.5, 1);
    }

    for (int num = 0; num < worldPoint.size(); num += 2) {
        line(image, imageP[num], imageP[num+1], Scalar(0, 255, 0), 2);
    }

    for (int num = 0; num < worldPoint.size()/2; num++) {
        Scalar color = (num % 2 == 0) ? Scalar(0, 0, 255) : Scalar(255, 0, 0);
        line(image, imageP[num], imageP[num+4], color, 2);
    }

    line(image, imageP[5], imageP[6], Scalar(255, 0, 0), 2);
    line(image, imageP[4], imageP[7], Scalar(0, 0, 255), 2);
    
}

void markerDetector(Ptr<aruco::Dictionary> dict) {
    VideoCapture cap(0);
    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);
    int fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G');
    cap.set(CAP_PROP_FOURCC, fourcc);

    Mat image;
    vector<vector<Point2f>> markerCorners, rejCandidates;
    vector<int> markerId;
    Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();
    
    while (waitKey(1) != 27) {
        cap.read(image);
        if (image.empty()) {
            continue;
        }

        aruco::detectMarkers(image, dict, markerCorners, markerId, params, rejCandidates);
        Mat outImage = image.clone();

        for (size_t i = 0; i < markerId.size(); i++) {
            aruco::drawDetectedMarkers(outImage, markerCorners, markerId);
            vector<Vec3d> rotV, transV;
            aruco::estimatePoseSingleMarkers(markerCorners, 0.01, intrinsicMatrix, distCoeffs, rotV, transV);
            cubaning(outImage, rotV[i], transV[i], 0.01);
            cout << "I see aruco" << endl;
        }
        
        imshow("cubaning", outImage);
    }
}

void arUcoMarkerGenerate(Ptr<aruco::Dictionary> dictionary, short int id, short int size, std::string path) {
	Mat markerImage;
	aruco::drawMarker(dictionary, id, size, markerImage, 1);
	imwrite(path + "marker" + to_string(id) + ".png", markerImage);
}

void arUcoBoardGenerate(
		Ptr<aruco::Dictionary> dictionary,
		short int row, 
		short int col, 
		float length, 
		float sep, 
		std::string path
	) {
		Mat arUcoBoard;
		Ptr<aruco::GridBoard> markerBoard = aruco::GridBoard::create(
			row, 
			col, 
			length, 
			sep, 
			dictionary, 
			0
		);
		markerBoard->draw(Size(1920, 1280), arUcoBoard, 10, 1);
		imwrite(path + "board" + std::to_string(row) + std::to_string(col) + ".png", arUcoBoard);
}

int main(int, char**)
{
	Mat markerImage, arUcoBoardImage;
	short int arUcoBoardRows = 5;
	short int arUcoBoardCols = 7;
	float markerLength = 0.01;
	float markerSeparation = 0.01;
	short int markerId = 23;
	short int markerSize = 200;
	string imagePath = "src/img/";
	string paramsArUco = "./src/camera_params.yaml";
	
	Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
	arUcoMarkerGenerate(
		dictionary, 
		markerId, 
		markerSize, 
		imagePath
	);

	arUcoBoardGenerate(
		dictionary,
		arUcoBoardRows, 
		arUcoBoardCols, 
		markerLength, 
		markerSeparation, 
		imagePath
	);

	cameraCalibration(dictionary, arUcoBoardRows, arUcoBoardCols, markerLength, markerSeparation, paramsArUco);
    markerDetector(dictionary);
	
 	return 0;
}