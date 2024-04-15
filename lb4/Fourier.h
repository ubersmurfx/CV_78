#pragma once
#include <iostream>
#include "opencv2/opencv.hpp"


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

std::ostream& operator << (std::ostream& out, const ::TickMeter& tm);

TickMeter::TickMeter() { reset(); }
int64 ::TickMeter::getTimeTicks() const { return sumTime; }
double ::TickMeter::getTimeMicro() const { return  getTimeMilli() * 1e3; }
double ::TickMeter::getTimeMilli() const { return getTimeSec() * 1e3; }
double ::TickMeter::getTimeSec() const { return (double)getTimeTicks() / cv::getTickFrequency(); }
int64 ::TickMeter::getCounter() const { return counter; }
void ::TickMeter::reset() { startTime = 0; sumTime = 0; counter = 0; }


void ::TickMeter::start() { startTime = cv::getTickCount(); }
void ::TickMeter::stop()
{
    int64 time = cv::getTickCount();
    if (startTime == 0)
        return;
    ++counter;
    sumTime += (time - startTime);
    startTime = 0;
}

std::ostream& operator << (std::ostream& out, const ::TickMeter& tm) { return out << tm.getTimeSec() << "sec"; }

class Fourier
{
public:
    Fourier() = default;
    Fourier(cv::Mat src);
    ~Fourier() = default;

    void DFT(cv::Mat& src, cv::Mat& dst);
    void IDFT(cv::Mat& src, cv::Mat& dst);
    cv::Mat optimalSize(cv::Mat& image, int channels);
    void tuneSpectrum(cv::Mat& magI);
    std::vector<std::complex<double>> convertMatToVector(cv::Mat img);
    void radix(std::vector<std::complex<double>>& src, bool invert,  std::vector<std::complex<double>>& dstArray);
    void spectrum(cv::Mat& imageAfterDFT, cv::Mat& result);
    void laplas(cv::Mat& src, cv::Mat& dst);
    void sobelX(cv::Mat& src, cv::Mat& dst);
    void sobelY(cv::Mat& src, cv::Mat& dst);
    void boxFilter(cv::Mat& src, cv::Mat& dst);
    cv::Mat high_filter(cv::Mat src, int rad);
	cv::Mat low_filter(cv::Mat src, int rad);
	cv::Mat kernelIm(cv::Mat kernel, int width, int height);
	cv::Mat radixSpec(std::vector<std::complex<double>> src);
};


Fourier::Fourier(cv::Mat src) 
{
	;
}


cv::Mat Fourier::optimalSize(cv::Mat& image, int channels)
{
	cv::Size dftSize;
	dftSize.width = cv::getOptimalDFTSize(image.cols);
	dftSize.height = cv::getOptimalDFTSize(image.rows);
	cv::Rect rectangle(0, 0, image.cols, image.rows);
	if (channels == 1)
	{
		cv::Mat dftImg(dftSize, CV_32FC1, cv::Scalar(0));
		image.copyTo(dftImg(rectangle));
		return dftImg;
	}

	if (channels == 2)
	{
		cv::Mat dftImg(dftSize, CV_32FC2, cv::Scalar(0));
		image.copyTo(dftImg(rectangle));
		return dftImg;
	}
}

void Fourier::DFT(cv::Mat& src, cv::Mat& dst)
{
    CV_Assert(src.channels() == 1 && src.type() == CV_32F);
	int rows = src.rows;
	int cols = src.cols;
	for (int row = 0; row < rows; row++)
		for (int col = 0; col < cols; col++)
			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
				{
					float arg = (float)CV_2PI * (((float)(row * i) / rows) + ((float)(col * j) / cols));
					dst.at<cv::Vec2f>(row, col)[0] = 
                        dst.at<cv::Vec2f>(row, col)[0] + 
                        src.at<cv::Vec<float, 1>>(i, j)[0] * cos(arg);

					dst.at<cv::Vec2f>(row, col)[1] = 
                        dst.at<cv::Vec2f>(row, col)[1] - 
                        src.at<cv::Vec<float, 1>>(i, j)[0] * sin(arg);
				}
}

void Fourier::IDFT(cv::Mat& src, cv::Mat& dst)
{
	int rows = src.rows;
	int cols = src.cols;
	for (int row = 0; row < rows; row++)
		for (int col = 0; col < cols; col++) 
        {  
			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
				{
					float arg = (float)CV_2PI * (((float)(row * i) / rows) + ((float)(col * j) / cols));
					dst.at<float>(row, col) += 
                        src.at<cv::Vec2f>(i, j)[0] * cos(arg) - 
                        src.at<cv::Vec2f>(i, j)[1] * sin(arg);
				}
			dst.at<float>(row, col) = ((float)1 / (rows * cols)) * dst.at<float>(row, col);
        }
}
void Fourier::radix(std::vector<std::complex<double>>& src, bool invert, std::vector<std::complex<double>>& dst) {
    const int n = src.size();
  
    if ((n & (n - 1)) != 0) {
        std::cout << "error in radix size: " << (n & (n - 1));
        return;
    }

    if (n == 1) {
        dst = src;
        return;
    }

    std::vector<std::complex<double>> src0(n / 2);
    std::vector<std::complex<double>> src1(n / 2);

    for (int i = 0; i < n / 2; i++) {
        src0[i] = src[2 * i];
        src1[i] = src[2 * i + 1];
    }

    std::vector<std::complex<double>> src_0(n / 2);
    std::vector<std::complex<double>> src_1(n / 2);

    radix(src0, invert, src_0);
    radix(src1, invert, src_1);

    dst.resize(n);
    double angle = 2 * M_PI / n * (invert ? -1 : 1);
    std::complex<double> w(1);
    std::complex<double> wn(cos(angle), sin(angle));

    for (int i = 0; i < n / 2; i++) {
        dst[i] = src_0[i] + w * src_1[i];
        dst[i + n / 2] = src_0[i] - w * src_1[i];
        w *= wn;
    }

    if (invert) {
        for (int i = 0; i < n; i++) {
            dst[i] /= n;
        }
    }
}
void Fourier::spectrum(cv::Mat& src, cv::Mat& res)
{
	std::vector<cv::Mat> temp;
	cv::split(src, temp);

	cv::Mat magn;
	magnitude(temp[0], temp[1], magn);

	tuneSpectrum(magn);

	magn += cv::Scalar::all(1);
	log(magn, magn); 

	cv::normalize(magn, magn, 0.0f, 1.0f, cv::NORM_MINMAX);
	res = magn.clone();
}


void Fourier::tuneSpectrum(cv::Mat& magI)
{
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));
	cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));
	cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));
	cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy));

	cv::Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

std::vector<std::complex<double>> Fourier::convertMatToVector(cv::Mat img)
{
    std::vector<uchar> imageVector(img.begin<uchar>(), img.end<uchar>());
    std::vector<std::complex<double>> complexVector;

    for (const auto &val : imageVector)
    {
        complexVector.push_back(std::complex<double>(val, 0));
    }
    return complexVector;
}

void Fourier::laplas(cv::Mat& src, cv::Mat& dst)
{
	cv::Mat exsrc(cv::Size(src.cols + 2, src.rows + 2), CV_32FC1, cv::Scalar());
	cv::Rect rectangle(0, 0, src.cols, src.rows);
	src.copyTo(exsrc(rectangle));
	cv::Mat imageAfterDFT(cv::Size(src.cols + 2, src.rows + 2), CV_32FC2, cv::Scalar());
	dft(exsrc, imageAfterDFT, cv::DFT_COMPLEX_OUTPUT);

	cv::Mat laplas(cv::Size(src.cols + 2, src.rows + 2), CV_32FC1, cv::Scalar());
	laplas.at<float>(0, 0) = 0;
	laplas.at<float>(0, 1) = 1;
	laplas.at<float>(0, 2) = 0;
	laplas.at<float>(1, 0) = 1;
	laplas.at<float>(1, 1) = -4;
	laplas.at<float>(1, 2) = 1;
	laplas.at<float>(2, 0) = 0;
	laplas.at<float>(2, 1) = 1;
	laplas.at<float>(2, 2) = 0;
	
	cv::Mat laplasAfterDFT(cv::Size(src.cols + 2, src.rows + 2), CV_32FC2, cv::Scalar());
	dft(laplas, laplasAfterDFT, cv::DFT_COMPLEX_OUTPUT);
	cv::Mat spec1;
	spectrum(laplasAfterDFT, spec1);
	cv::normalize(spec1, spec1, 0, 1, cv::NORM_MINMAX);
	spec1.convertTo(spec1, CV_8UC1, 255);
	cv::imshow("Spec1", spec1);

	cv::Mat forMul(cv::Size(src.cols + 2, src.rows + 2), CV_32FC2, cv::Scalar());
	for (int i = 0; i < src.rows + 2; i++)
	{
		for (int j = 0; j < src.cols + 2; j++)
		{
			float a1 = laplasAfterDFT.at<cv::Vec2f>(i, j)[0];
			float b1 = laplasAfterDFT.at<cv::Vec2f>(i, j)[1];
			float a2 = imageAfterDFT.at<cv::Vec2f>(i, j)[0];
			float b2 = imageAfterDFT.at<cv::Vec2f>(i, j)[1];

			forMul.at<cv::Vec2f>(i, j)[0] = a1 * a2 - b1 * b2;
			forMul.at<cv::Vec2f>(i, j)[1] = a1 * b2 + a2 * b1;
		}
	}
	cv::Mat resultInverse(cv::Size(src.cols + 2, src.rows + 2), CV_32FC1, cv::Scalar());
	dft(forMul, resultInverse, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
	cv::Rect rectangle2(1, 1, src.cols, src.rows);
	resultInverse(rectangle2).copyTo(dst);
	cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
	dst.convertTo(dst, CV_8UC1, 255);
}

void Fourier::sobelY(cv::Mat& src, cv::Mat& dst)
{
	cv::Mat exsrc(cv::Size(src.cols + 2, src.rows + 2), CV_32FC1, cv::Scalar());
	cv::Rect rectangle(0, 0, src.cols, src.rows);
	src.copyTo(exsrc(rectangle));
	cv::Mat imageAfterDFT(cv::Size(src.cols + 2, src.rows + 2), CV_32FC2, cv::Scalar());
	dft(exsrc, imageAfterDFT, cv::DFT_COMPLEX_OUTPUT);

	cv::Mat sobelY(cv::Size(src.cols + 2, src.rows + 2), CV_32FC1, cv::Scalar());
    sobelY.at<float>(0, 0) = 1;
	sobelY.at<float>(0, 1) = 2;
	sobelY.at<float>(0, 2) = 1;
	sobelY.at<float>(1, 0) = 0;
	sobelY.at<float>(1, 1) = 0;
	sobelY.at<float>(1, 2) = 0;
	sobelY.at<float>(2, 0) = -1;
	sobelY.at<float>(2, 1) = -2;
	sobelY.at<float>(2, 2) = -1;
		
	cv::Mat sobelYAfterDFT(cv::Size(src.cols + 2, src.rows + 2), CV_32FC2, cv::Scalar());
	dft(sobelY, sobelYAfterDFT, cv::DFT_COMPLEX_OUTPUT);

	cv::Mat spec2;
	spectrum(sobelYAfterDFT, spec2);
	cv::normalize(spec2, spec2, 0, 1, cv::NORM_MINMAX);
	spec2.convertTo(spec2, CV_8UC1, 255);
	cv::imshow("Spec2", spec2);

	cv::Mat forMul(cv::Size(src.cols + 2, src.rows + 2), CV_32FC2, cv::Scalar());
	for (int i = 0; i < src.rows + 2; i++)
	{
		for (int j = 0; j < src.cols + 2; j++)
		{
			float a1 = sobelYAfterDFT.at<cv::Vec2f>(i, j)[0];
			float b1 = sobelYAfterDFT.at<cv::Vec2f>(i, j)[1];
			float a2 = imageAfterDFT.at<cv::Vec2f>(i, j)[0];
			float b2 = imageAfterDFT.at<cv::Vec2f>(i, j)[1];

			forMul.at<cv::Vec2f>(i, j)[0] = a1 * a2 - b1 * b2;
			forMul.at<cv::Vec2f>(i, j)[1] = a1 * b2 + a2 * b1;
		}
	}

	cv::Mat resultInverse(cv::Size(src.cols + 2, src.rows + 2), CV_32FC1, cv::Scalar());
	dft(forMul, resultInverse, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
	cv::Rect rectangle2(0, 0, src.cols, src.rows);
	resultInverse(rectangle2).copyTo(dst);
	cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
	dst.convertTo(dst, CV_8UC1, 255);
}

void Fourier::sobelX(cv::Mat& src, cv::Mat& dst)
{
	cv::Mat exsrc(cv::Size(src.cols + 2, src.rows + 2), CV_32FC1, cv::Scalar());
	cv::Rect rectangle(0, 0, src.cols, src.rows);
	src.copyTo(exsrc(rectangle));
	cv::Mat imageAfterDFT(cv::Size(src.cols + 2, src.rows + 2), CV_32FC2, cv::Scalar());
	dft(exsrc, imageAfterDFT, cv::DFT_COMPLEX_OUTPUT);

	cv::Mat sobelX(cv::Size(src.cols + 2, src.rows + 2), CV_32FC1, cv::Scalar());
    sobelX.at<float>(0, 0) = 1;
	sobelX.at<float>(0, 1) = 0;
	sobelX.at<float>(0, 2) = -1;
	sobelX.at<float>(1, 0) = 2;
	sobelX.at<float>(1, 1) = 0;
	sobelX.at<float>(1, 2) = -2;
	sobelX.at<float>(2, 0) = 1;
	sobelX.at<float>(2, 1) = 0;
	sobelX.at<float>(2, 2) = -1;
		
	cv::Mat sobelXAfterDFT(cv::Size(src.cols + 2, src.rows + 2), CV_32FC2, cv::Scalar());
	dft(sobelX, sobelXAfterDFT, cv::DFT_COMPLEX_OUTPUT);

	cv::Mat spec2;
	spectrum(sobelXAfterDFT, spec2);
	cv::normalize(spec2, spec2, 0, 1, cv::NORM_MINMAX);
	spec2.convertTo(spec2, CV_8UC1, 255);
	cv::imshow("Spec2", spec2);

	cv::Mat forMul(cv::Size(src.cols + 2, src.rows + 2), CV_32FC2, cv::Scalar());
	for (int i = 0; i < src.rows + 2; i++)
	{
		for (int j = 0; j < src.cols + 2; j++)
		{
			float a1 = sobelXAfterDFT.at<cv::Vec2f>(i, j)[0];
			float b1 = sobelXAfterDFT.at<cv::Vec2f>(i, j)[1];
			float a2 = imageAfterDFT.at<cv::Vec2f>(i, j)[0];
			float b2 = imageAfterDFT.at<cv::Vec2f>(i, j)[1];

			forMul.at<cv::Vec2f>(i, j)[0] = a1 * a2 - b1 * b2;
			forMul.at<cv::Vec2f>(i, j)[1] = a1 * b2 + a2 * b1;
		}
	}

	cv::Mat resultInverse(cv::Size(src.cols + 2, src.rows + 2), CV_32FC1, cv::Scalar());
	dft(forMul, resultInverse, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
	cv::Rect rectangle2(0, 0, src.cols, src.rows);
	resultInverse(rectangle2).copyTo(dst);
	cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
	dst.convertTo(dst, CV_8UC1, 255);
}

void Fourier::boxFilter(cv::Mat& src, cv::Mat& dst)
{
	cv::Mat exsrc(cv::Size(src.cols + 2, src.rows + 2), CV_32FC1, cv::Scalar());
	cv::Rect rectangle(0, 0, src.cols, src.rows);
	src.copyTo(exsrc(rectangle));
	cv::Mat imageAfterDFT(cv::Size(src.cols + 2, src.rows + 2), CV_32FC2, cv::Scalar());
	dft(exsrc, imageAfterDFT, cv::DFT_COMPLEX_OUTPUT);

	cv::Mat boxFilter(cv::Size(src.cols + 2, src.rows + 2), CV_32FC1, cv::Scalar());
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			boxFilter.at<float>(i, j) = 1;
		}
	}

	cv::Mat boxFilterAfterDFT(cv::Size(src.cols + 2, src.rows + 2), CV_32FC2, cv::Scalar());
	dft(boxFilter, boxFilterAfterDFT, cv::DFT_COMPLEX_OUTPUT);
	cv::Mat spec3;
	spectrum(boxFilterAfterDFT, spec3);
	cv::normalize(spec3, spec3, 0, 1, cv::NORM_MINMAX);
	spec3.convertTo(spec3, CV_8UC1, 255);
	cv::imshow("Spec3", spec3);


	cv::Mat forMul(cv::Size(src.cols + 2, src.rows + 2), CV_32FC2, cv::Scalar());
	for (int i = 0; i < src.rows + 2; i++)
	{
		for (int j = 0; j < src.cols + 2; j++)
		{
			float a1 = boxFilterAfterDFT.at<cv::Vec2f>(i, j)[0];
			float b1 = boxFilterAfterDFT.at<cv::Vec2f>(i, j)[1];
			float a2 = imageAfterDFT.at<cv::Vec2f>(i, j)[0];
			float b2 = imageAfterDFT.at<cv::Vec2f>(i, j)[1];

			forMul.at<cv::Vec2f>(i, j)[0] = a1 * a2 - b1 * b2;
			forMul.at<cv::Vec2f>(i, j)[1] = a1 * b2 + a2 * b1;
		}
	}
	cv::Mat resultInverse(cv::Size(src.cols + 2, src.rows + 2), CV_32FC1, cv::Scalar());
	cv::dft(forMul, resultInverse, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
	cv::Rect rectangle2(1, 1, src.cols, src.rows);
	resultInverse(rectangle2).copyTo(dst);
	cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
	dst.convertTo(dst, CV_8UC1, 255);
}

cv::Mat Fourier::low_filter(cv::Mat src, int rad) {
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(src.rows);
    int n = cv::getOptimalDFTSize(src.cols);
    cv::copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
    cv::Mat complex;
    cv::merge(planes, 2, complex);
    cv::dft(complex, complex);
    tuneSpectrum(complex);
    cv::Mat mask = cv::Mat::zeros(complex.rows, complex.cols, CV_32F);
    cv::circle(mask, cv::Point(mask.cols / 2, mask.rows / 2), rad, cv::Scalar(1, 1, 1), -1, 8, 0);
    cv::Mat filtered;
    complex.copyTo(filtered);
    cv::split(filtered, planes);
    planes[0] = planes[0].mul(mask);
    planes[1] = planes[1].mul(mask);
    cv::merge(planes, 2, filtered);
    cv::split(filtered, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat mag = planes[0];
    mag += cv::Scalar::all(1);
    cv::log(mag, mag);
    cv::normalize(mag, mag, 0, 1, cv::NORM_MINMAX);
    tuneSpectrum(filtered);
    cv::Mat inversed;
    cv::idft(filtered, inversed, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    cv::normalize(inversed, inversed, 0, 1, cv::NORM_MINMAX);
    imshow("lf", mag);
    return inversed;
}

cv::Mat Fourier::high_filter(cv::Mat src, int rad) {
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(src.rows);
    int n = cv::getOptimalDFTSize(src.cols);
    cv::copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
    cv::Mat complex;
    cv::merge(planes, 2, complex); 
    cv::dft(complex, complex);
    tuneSpectrum(complex);
    cv::Mat mask = cv::Mat::ones(complex.rows, complex.cols, CV_32F);
    cv::circle(mask, cv::Point(mask.cols / 2, mask.rows / 2), rad, cv::Scalar(0, 0, 0), -1, 8, 0); 
    cv::Mat filtered;
    complex.copyTo(filtered);
    cv::split(filtered, planes);
    planes[0] = planes[0].mul(mask);
    planes[1] = planes[1].mul(mask); 
    cv::merge(planes, 2, filtered);
    cv::split(filtered, planes); 
    cv::magnitude(planes[0], planes[1], planes[0]); 
    cv::Mat mag = planes[0];
    mag += cv::Scalar::all(1); 
    cv::log(mag, mag);
    cv::normalize(mag, mag, 0, 1, cv::NORM_MINMAX);
    tuneSpectrum(filtered);
    cv::Mat inversed;
    cv::idft(filtered, inversed, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT); 
    cv::normalize(inversed, inversed, 0, 1, cv::NORM_MINMAX);
    imshow("hf", mag);
    return inversed;
}

cv::Mat Fourier::kernelIm(cv::Mat kernel, int width, int height) {
    cv::Mat result;
    double minVal, maxVal;
    cv::minMaxLoc(kernel, &minVal, &maxVal);
    kernel = (kernel - minVal) / (maxVal - minVal);
    kernel.convertTo(result, CV_8UC1, 255);
    cv::resize(result, result, cv::Size(width, height));
    return result;
}

cv::Mat Fourier::radixSpec(std::vector<std::complex<double>> src) {
	int rows = 128;
	int cols = 256;
	cv::Mat res(rows, cols, CV_32FC2);
	if (src.size() != rows * cols) {
		std::cerr << src.size() <<
		std::endl;
		return res;
	}

	int k = 0;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			res.at<cv::Vec2f>(i, j)[1] = src[k].imag();
			res.at<cv::Vec2f>(i, j)[0] = src[k].real();
			k++;
		}
	}

	cv::rotate(res, res, cv::ROTATE_90_COUNTERCLOCKWISE);
	cv::rotate(res, res, cv::ROTATE_180);
	return res;
}