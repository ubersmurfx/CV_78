#include <iostream>
#include "opencv2/opencv.hpp"
#include "Fourier.h"
#include "time.h"

// ./lb4 IM/2_200_200.png IM/nomer_gost.png IM/k_sym.png IM/m_sym.png IM/7.png

cv::Mat correlation(cv::Mat& src, cv::Mat& sym) {
	cv::Mat res;
    int cols = src.cols - sym.cols + 1;
    int rows = src.rows - sym.rows + 1;
	std::cout << "image size: " << cols << "x" << rows << std::endl;
    res.create(rows, cols, CV_32FC1);
	
    cv::matchTemplate(src, sym, res, 1);
    cv::normalize(res, res, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
	cv::imshow("res", res);
    double min_val;
    cv::minMaxLoc(res, &min_val, NULL);
	std::cout << "minimum value: " << min_val << std::endl;
    cv::threshold(res, res, min_val + 0.02, 255, cv::THRESH_BINARY_INV);
	return res;
}


int main(int argc, char** argv)
{
	std::string image1, nomer_po_gost, symbol1, symbol2, symbol3;
	std::cout << "argc: " << argc << std::endl;
	if (argc == 6) 
	{
		image1 = argv[1];
		nomer_po_gost = argv[2];
		symbol1 = argv[3];
		symbol2 = argv[4];
		symbol3 = argv[5];
	}
	else
	{
		image1 = "/home/gerzeg/Polytech/TV/lb4/IM/radix.png";
		nomer_po_gost = "/home/gerzeg/Polytech/TV/lb4/IM/kot1.jpeg";
		symbol1 = "/home/gerzeg/Polytech/TV/lb4/IM/lob_kota_2.jpeg";
		symbol2 = "/home/gerzeg/Polytech/TV/lb4/IM/m_sym.png";
		symbol3 = "/home/gerzeg/Polytech/TV/lb4/IM/7.png";
	}


	TickMeter timer;
	
	cv::Mat image = cv::imread(image1, cv::IMREAD_GRAYSCALE);
	cv::imshow("image", image);
	cv::waitKey(-1);

	cv::Mat image_clone = image.clone();

	cv::Mat src;
	image_clone.convertTo(src, CV_32FC1);

	cv::Mat resultForward(cv::Size(image_clone.cols, image_clone.rows), CV_32FC2, cv::Scalar());
	cv::Mat resultInverse(cv::Size(image_clone.cols, image_clone.rows), CV_32FC1, cv::Scalar());
	cv::Mat resultSpectrum(cv::Size(image_clone.cols, image_clone.rows), CV_32FC1, cv::Scalar());
	cv::Mat resultOpenCV(cv::Size(image_clone.cols, image_clone.rows), CV_32FC1, cv::Scalar());
	cv::Mat resultLaplas;
	cv::Mat resultSobelX;
	cv::Mat resultSobelY;
	cv::Mat resultBoxFilter;
	cv::Mat resultLfp;
	cv::Mat resultHfp;
	cv::Mat im_dftopencv;
	cv::Mat resultLfp_inv;
	cv::Mat resultHfp_inv;

	
	
	Fourier im_proc(image);
	src = im_proc.optimalSize(src, 1);
	resultForward = im_proc.optimalSize(resultForward, 2);
	
	// DFT 
	timer.start();
	im_proc.DFT(src, resultForward);
	timer.stop();
	std::cout << "own dft: " << timer.getTimeSec() << std::endl;
	timer.reset();
	cv::waitKey(-1);

	// IDFT
	timer.start();
	im_proc.IDFT(src, resultInverse);
	timer.stop();
	std::cout << "own idft: " << timer.getTimeSec() << std::endl;
	timer.reset();
	cv::normalize(resultInverse, resultInverse, 0, 1, cv::NORM_MINMAX);
	resultInverse.convertTo(resultInverse, CV_8UC1, 255);
	imshow("ImageAfterInverseTransform", resultInverse);
	cv::waitKey(-1);

	timer.start();
	cv::dft(src.clone(), resultOpenCV, cv::DFT_COMPLEX_OUTPUT);
	timer.stop();
	std::cout << "cv::dft: " << timer.getTimeSec() << std::endl;
	timer.reset();
	cv::waitKey(-1);

	cv::Mat opencvSpectr;
	timer.start();
	im_proc.spectrum(resultOpenCV, opencvSpectr);
	timer.stop();
	std::cout << "own spectr: " << timer.getTimeSec() << std::endl;
	timer.reset();
	cv::normalize(opencvSpectr, opencvSpectr, 0, 1, cv::NORM_MINMAX);
	opencvSpectr.convertTo(opencvSpectr, CV_8UC1, 255);
	cv::imshow("opencv", opencvSpectr);
	cv::waitKey(-1);

	std::vector<std::complex<double>> inputRadix, outputRadix;
	inputRadix = im_proc.convertMatToVector(image);

	// for (int i = 0; i < inputRadix.size(); i++) {
	//	std::cout << inputRadix[i] << std::endl;
	//}
	
	std::cout << "radix input size: " << inputRadix.size() << std::endl;
	timer.start();
	im_proc.radix(inputRadix, 0, outputRadix);
	
	
	timer.stop();
	std::cout << "radix time: " << timer.getTimeSec() << std::endl;
	for (int i = 0; i < outputRadix.size(); i++) {
		std::cout << outputRadix[i] << std::endl;
	}
	std::cout << "radix output size: " << outputRadix.size() << std::endl;
	cv::Mat resultRadix;
	resultRadix = im_proc.radixSpec(outputRadix);

	timer.reset();
	cv::waitKey(-1);

	// spectr radix2
	cv::Mat radix;
	timer.start();
	im_proc.spectrum(resultRadix, radix);
	timer.stop();
	std::cout << "radix: " << timer.getTimeSec() << std::endl;
	timer.reset();
	cv::normalize(radix, radix, 0, 1, cv::NORM_MINMAX);
	radix.convertTo(radix, CV_8UC1, 255);
	cv::imshow("radix", radix);
	cv::waitKey(-1);

	// spectrum IDFT
	timer.start();
	im_proc.spectrum(resultForward, resultSpectrum);
	timer.stop();
	std::cout << "own spectr: " << timer.getTimeSec() << std::endl;
	timer.reset();
	cv::normalize(resultSpectrum, resultSpectrum, 0, 1, cv::NORM_MINMAX);
	resultSpectrum.convertTo(resultSpectrum, CV_8UC1, 255);
	cv::imshow("spectrum", resultSpectrum);
	std::cout << src.cols << std::endl;
	cv::waitKey(-1);

	// laplas
	im_proc.laplas(src, resultLaplas);
	cv::imshow("laplas", resultLaplas);

	// sobelX
	im_proc.sobelX(src, resultSobelX);
	imshow("sobelX", resultSobelX);

	// sobelY
	im_proc.sobelY(src, resultSobelY);
	imshow("sobelY", resultSobelY);
	
	// boxFilter
	im_proc.boxFilter(src, resultBoxFilter);
	cv::imshow("boxFilter", resultBoxFilter);

	cv::Mat sobelX = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat sobelY = (cv::Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    cv::Mat box = (cv::Mat_<float>(3, 3) << 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9);
    cv::Mat laplacian = (cv::Mat_<float>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);

	cv::Mat sobelX_kern = im_proc.kernelIm(sobelX, src.cols, src.rows);
    cv::imshow("xs", sobelX_kern);
    cv::Mat sobelY_kern = im_proc.kernelIm(sobelY, src.cols, src.rows);
    cv::imshow("ys", sobelY_kern);
    cv::Mat box_kern = im_proc.kernelIm(box, src.cols, src.rows);
    cv::imshow("bs", box_kern);
    cv::Mat laplacian_kern = im_proc.kernelIm(laplacian, src.cols, src.rows);
    cv::imshow("ls", laplacian_kern);

	// lpf
	// 
	cv::Mat low = im_proc.low_filter(src, 20);
    cv::Mat high = im_proc.high_filter(src, 20);
    // Показать результаты
    cv::imshow("Low-pass", low);
    cv::imshow("High-pass", high);
	cv::waitKey(-1);
/*_____________________________________________________________________________________________*/

	cv::Mat nomer_gost = cv::imread(nomer_po_gost, cv::IMREAD_GRAYSCALE);
    cv::Mat sym1 = cv::imread(symbol1, cv::IMREAD_GRAYSCALE);
    cv::Mat sym2 = cv::imread(symbol2, cv::IMREAD_GRAYSCALE);
    cv::Mat sym3 = cv::imread(symbol3, cv::IMREAD_GRAYSCALE);

    cv::imshow("nomer_gost",nomer_gost);
	cv::waitKey(-1);

	cv::Mat resultCorrelation1;
	cv::Mat resultCorrelation2;
	cv::Mat resultCorrelation3;

    resultCorrelation1 = correlation(nomer_gost, sym1);
    cv::imshow("sym1", sym1);
    cv::imshow("cor1",resultCorrelation1);

	cv::waitKey(-1);
    resultCorrelation2 = correlation(nomer_gost, sym2);
    cv::imshow("sym2", sym2);
    cv::imshow("cor2", resultCorrelation2);

	cv::waitKey(-1);
    resultCorrelation3 = correlation(nomer_gost, sym3);
    cv::imshow("sym3", sym3);
    cv::imshow("cor3", resultCorrelation3);
    cv::waitKey(-1);

	return 0;
}