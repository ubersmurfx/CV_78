#include <iostream>
#include <opencv2/opencv.hpp>


// Timers's class
class CV_EXPORTS TickMeter
{
public:
  TickMeter();
  void start();
  void stop();

  int64 getTimeTicks() const;
  double getTimeMicro() const;
  double getTimeMilli() const;
  double getTimeSec()   const;
  int64 getCounter() const;

  void reset();
private:
  int64 counter;
  int64 sumTime;
  int64 startTime;
};

std::ostream& operator << (std::ostream& out, const TickMeter& tm);

TickMeter::TickMeter() { reset(); }
int64 TickMeter::getTimeTicks() const { return sumTime; }
double TickMeter::getTimeMicro() const { return  getTimeMilli() * 1e3; }
double TickMeter::getTimeMilli() const { return getTimeSec() * 1e3; }
double TickMeter::getTimeSec() const { return (double)getTimeTicks() / cv::getTickFrequency(); }
int64 TickMeter::getCounter() const { return counter; }
void TickMeter::reset() { startTime = 0; sumTime = 0; counter = 0; }

void TickMeter::start() { startTime = cv::getTickCount(); }
void TickMeter::stop()
{
  int64 time = cv::getTickCount();
  if (startTime == 0)
    return;
  ++counter;
  sumTime += (time - startTime);
  startTime = 0;
}

std::ostream& operator << (std::ostream& out, const TickMeter& tm) { return out << tm.getTimeSec() << "sec"; }


void box_filter(cv::Mat& src, cv::Mat& dst, int kernel_size) {
    assert(kernel_size % 2 == 1);
    dst.create(src.size(), src.type());
    unsigned int sum = 0;

    for (int col = (kernel_size-1) / 2; col < dst.cols - (kernel_size-1) / 2; col++) {
        for (int row = (kernel_size-1) / 2; row < dst.rows - (kernel_size-1) / 2; row++) {
            sum = 0;
            for (int temp_col = col - (kernel_size-1) / 2; temp_col <= col + (kernel_size-1) / 2; temp_col++) {
                for (int temp_row = row - (kernel_size-1) / 2; temp_row <= row + (kernel_size-1) / 2; temp_row++) {
                    sum += (int)src.at<uchar>(temp_row, temp_col);
                }
            }
            dst.at<uchar>(row, col) = std::round(sum / (kernel_size * kernel_size));
        }
    }
}

/*
    for (int col = 0; col < img1.cols; col++) {
        for (int row = 0; row < img1.rows; row++) {
            if (img1.at<uchar>(row, col) == img2.at<uchar>(row, col) ||
            img1.at<uchar>(row, col) == img2.at<uchar>(row, col) + 1 ||
            img1.at<uchar>(row, col) == img2.at<uchar>(row, col) - 1)
             {
                result.at<uchar>(row, col) = 0;
                count++;
            }
        }
    }
    interest = count / (img1.rows * img1.cols);
    */

void compare(cv::Mat& img1, cv::Mat& img2, cv::Mat& result) {
    result = img1.clone();

    double interest = 0;

    for (int col = 0; col < img1.cols; col++) {
        for (int row = 0; row < img1.rows; row++) {
            uint8_t local_diff = std::abs(img1.at< uint8_t >(row, col) - img2.at< uint8_t >(row, col));

			interest += (float)(local_diff) / 255;

			result.at< uint8_t >(row, col) = local_diff;
		}
	}
	
	std::cout << "Match: " << std::round((1 - interest / (result.rows * result.cols)) * 100) << std::endl;    
}


void unsharpMasking(cv::Mat& src, cv::Mat& filter, cv::Mat& dst, double k = 2) {
    dst = src.clone();

    for (int col = 0; col < src.cols; col++)
        for (int row = 0; row < src.rows; row++) {
            dst.at<uchar>(row, col) = src.at<uchar>(row, col) + k * (src.at<uchar>(row, col) - filter.at<uchar>(row, col));
        }
}

void laplasFiltration(cv::Mat& src, cv::Mat& dst, int ksize) { 
    

    dst.create(src.size(), src.type());

    cv::Mat kernel = cv::Mat::zeros(ksize, ksize, CV_32S);
    kernel.at<int>((ksize - 1) / 2, (ksize - 1) / 2) = -1 * (ksize * ksize - 1);
    for (int i = 0; i < ksize; i++)
    {
        for (int j = 0; j < ksize; j++)
        {
            if (i != (ksize - 1) / 2 || j != (ksize - 1) / 2)
            {
                kernel.at<int>(i, j) = 1;
            }
        }
    }

    for (int i = (ksize - 1) / 2; i < src.rows - (ksize - 1) / 2; i++)
    {
        for (int j = (ksize - 1) / 2; j < src.cols - (ksize - 1) / 2; j++)
        {          
            int sum = 0;
            for (int m = 0; m < ksize; m++)
            {
                for (int n = 0; n < ksize; n++)
                {                
                    sum += kernel.at<int>(m, n) * src.at<uchar>(i - (ksize - 1) / 2 + m, j - (ksize - 1) / 2 + n);
                }
            }          
            dst.at<uchar>(i, j) = (uchar)abs(sum);
        }
    }
}

void logTransform(cv::Mat& img1, cv::Mat& result, double c = 70) {
    result = img1.clone();
    for (int index = 0; index < img1.cols * img1.rows; index++)
        result.data[index] = std::round(c * std::log(1 + img1.data[index]));
}

int main(int argc, char** argv) {
    // set timer
    TickMeter timer;

    cv::Mat img = cv::imread("/home/gerzeg/Polytech/TV/lb2/images/captain.jpeg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cout << "Could not read the image" << std::endl;
        return -1;
    }

    int startX = 325;
    int startY = 100;
    int width = 400;
    int height = 400;
    cv::Rect rp(startX, startY, width, height);
    cv::Mat roi = img(rp);

    cv::imshow("ROI", roi);
    cv::waitKey(-1);
/*
    //box_filter
    cv::Mat box_filtered;
    timer.start();
    box_filter(roi, box_filtered, 3);
    timer.stop();
    std::cout << "My time: " << timer.getTimeSec() << std::endl;
    timer.reset();
    cv::imshow("box_filtered", box_filtered);
    cv::waitKey(-1);

    //opencv box_filter
    cv::Mat opencvFiltered;
    timer.start();
    blur(roi, opencvFiltered, cv::Size(3, 3));
    timer.stop();
    std::cout << "Blur time: " << timer.getTimeSec() << std::endl;
    timer.reset();
    cv::imshow("opencvFiltered", opencvFiltered);
    cv::waitKey(-1);
*/
    //Gauss filter
    cv::Mat GaussFiltered;
    GaussianBlur(roi, GaussFiltered, cv::Size(3, 3), (0, 0));
    cv::imshow("Gauss", GaussFiltered);
    cv::waitKey(-1);
/*
    //compare images 1 own box filter and opencv box filter
    cv::Mat comparsion_1;
    compare(box_filtered, opencvFiltered, comparsion_1);
    cv::imshow("comparison BOX 1", comparsion_1);
    cv::waitKey(-1);

    //compare images 2 opencv box filter and gauss filter
    cv::Mat comparsion_2;
    compare(GaussFiltered, box_filtered, comparsion_2);
    cv::imshow("comparison BOX 2", comparsion_2);
    cv::waitKey(-1);

    //unsharp masking
    cv::Mat boxSharp;
    cv::Mat gaussSharp; 
    unsharpMasking(roi, opencvFiltered, boxSharp);
    //logTransform(GaussFiltered, GaussFiltered);
    unsharpMasking(roi, GaussFiltered, gaussSharp);
    cv::imshow("box sharping", boxSharp);
    cv::imshow("gauss sharping", gaussSharp);
    cv::waitKey(-1);

    // comparsion log between box and gauss
    cv::Mat comparsion_3;
    compare(boxSharp, gaussSharp, comparsion_3);
    cv::imshow("comparison BOX 3", comparsion_3);
    cv::waitKey(-1);
*/
    // laplas filter

    cv::imshow("roi", roi);
    cv::waitKey(-1);

    cv::Mat laplas;
    laplasFiltration(roi, laplas, 2);
    cv::imshow("laplasFiltration", laplas);
    cv::waitKey(-1);

    cv::Mat comparsion_3_1;
    compare(GaussFiltered, laplas, comparsion_3_1);
    cv::imshow("comparison BOX 3_1", comparsion_3_1);
    cv::waitKey(-1);

    // unsharp for laplas
    cv::Mat laplasSharp;
    double alpha2 = 0.8;
    laplasSharp = roi - alpha2 * laplas;

    cv::imshow("laplasSharp", laplasSharp);
    cv::waitKey(-1);

    // comparsion log between unsharp laplas and box
    cv::Mat comparsion_4;
    compare(laplas, laplasSharp, comparsion_4);
    cv::imshow("comparison BOX 4", comparsion_4);
    cv::waitKey(-1);
    return 0;

}