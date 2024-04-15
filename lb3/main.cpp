#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdint.h>

// https://colorpicker.me/#613ac8
// to fix error
// "libpng warning: iCCP: known incorrect sRGB profile"
// sudo apt install pngcrush
// pngcrush -n -q ntcs_quest_measurement.png
// mogrify ntcs_quest_measurement.png


// function for drawing the target
void indicator(cv::Mat& src, int mx, int my, int radius_val) {
    cv::circle(src, 
        cv::Point(mx, my), 
        radius_val, 
        cv::Scalar(255, 0, 0)
    );
    
    cv::circle(src, 
        cv::Point(mx, my), 
        5 * radius_val, 
        cv::Scalar(255, 0, 0)
    );
    
    cv::line(src, 
        cv::Point(mx, my), 
        cv::Point(mx + 2 * radius_val, my),
        cv::Scalar(255, 0, 0)
    );

    cv::line(src, 
        cv::Point(mx, my), 
        cv::Point(mx - 2 * radius_val, my), 
        cv::Scalar(255, 0, 0)
    );
    
    cv::line(src, 
        cv::Point(mx, my), 
        cv::Point(mx, my + 2 * radius_val), 
        cv::Scalar(255, 0, 0)
    );
    
    cv::line(src, 
        cv::Point(mx, my), 
        cv::Point(mx, my - 2 * radius_val), 
        cv::Scalar(255, 0, 0)
    );
}


void alababah(cv::Mat& src, cv::Mat& res, int threshold_val, bool debug = true) {
    res.create(src.size(), src.type());
    cv::Mat gray_src;
    // conversion to gray color space
    cv::cvtColor(src, gray_src, cv::COLOR_BGR2GRAY);
    cv::threshold(gray_src, res, threshold_val, 255, cv::THRESH_BINARY);
    
    // outline selection
    std::vector <std::vector <cv::Point>> contour;
    cv::findContours(res, contour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    
    // find the largest contour
    std::vector<cv::Point> largest_contour = contour[0];
    for (int index = 0; index < contour.size(); index++) {
        if (contour[index].size() > largest_contour.size()) {
            largest_contour = contour[index];
        }
    }

    // fill the entire image with black 
    for (int index = 0; index < res.cols * res.rows; index++) {
        res.data[index] = 0;
    }

    // draw the outline (white on black)
    cv::polylines(res, largest_contour, true, 255);
    double c_area = cv::contourArea(largest_contour, false);

    // center definition
    cv::Moments mnts = cv::moments(largest_contour);
    double m00 = mnts.m00;
    double m10 = mnts.m10;
    double m01 = mnts.m01;

    // rad in pixels
    int radius_val = 5;
    unsigned int mx = (m10 / m00);
    unsigned int my = (m01 / m00);
    if (debug) {
        std::cout << "contour center X:" << mx << " Y:" << my << std::endl;
    }
    
    indicator(res, mx, my, radius_val);
    indicator(src, mx, my, radius_val);

    cv::imshow("res", res);
    cv::imshow("src", src);
    cv::waitKey(-1);
}

void democratization(cv::Mat& src, bool debug = true) {
    int low_H = 0, low_S = 40, low_V = 40;
    int high_H = 35, high_S = 255, high_V = 255;

    cv::Mat res = src.clone();

    // Convert from BGR to HSV colorspace
    cvtColor(src, res, cv::COLOR_BGR2HSV);
    
    // Detect the object based on HSV Range Values
    cv::inRange(res, cv::Scalar(low_H, low_S, low_V), cv::Scalar(high_H, high_S, high_V), res);
    

    // similar to task 1
    std::vector <std::vector <cv::Point>> contour;
    cv::findContours(res, contour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    
    for (int index = 0; index < res.cols * res.rows; index++) {
        res.data[index] = 0;
    }
    

    double max_c_area = -1;

    for (int contour_index = 0; contour_index < contour.size(); contour_index++) {
        if (cv::contourArea(contour[contour_index], false) > max_c_area) {
            max_c_area = cv::contourArea(contour[contour_index]);
        }
    }

    for (int contour_index = 0; contour_index < contour.size(); contour_index++) {
        if (cv::contourArea(contour[contour_index]) > 0.5 * max_c_area) {
            cv::polylines(res, contour[contour_index], true, 255);
            cv::Moments mnts = cv::moments(contour[contour_index]);
            double m00 = mnts.m00;
            double m10 = mnts.m10;
            double m01 = mnts.m01;

            // rad in pixels
            int radius_val = 5;
            unsigned int mx = (m10 / m00);
            unsigned int my = (m01 / m00);
            if (debug) {
                std::cout << "contour center X:" << mx << " Y:" << my << std::endl;
            }
            indicator(src, mx, my, radius_val);
            indicator(res, mx, my, radius_val);
        }
    }

    cv::imshow("res", res);
    cv::imshow("src", src);
    cv::waitKey(-1);
}

void robots(cv::Mat& src, bool debug = false) {
    cv::dilate(src, src, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4, 4)));
    cv::Mat res = src.clone();
    
    cv::Mat green(src.size(), CV_8UC1);
    cv::Mat blue(src.size(), CV_8UC1);
    cv::Mat red(src.size(), CV_8UC1);
// H (0 - 179) S (0 - 255) V (0 - 255)
    unsigned int low_green_H = 65, low_green_S = 50, low_green_V = 50;
    unsigned int high_green_H = 80, high_green_S = 255, high_green_V = 255;

    unsigned int low_blue_H = 90, low_blue_S = 50, low_blue_V = 50;
    unsigned int high_blue_H = 100, high_blue_S = 255, high_blue_V = 255;

    unsigned int low_red_H = 165, low_red_S = 50, low_red_V = 50;
    unsigned int high_red_H = 179, high_red_S = 240, high_red_V = 240;

    double green_max_c_area = -1, blue_max_c_area = -1, red_max_c_area = -1, lightBulb_max_c_area = -1, lightBulb_c_max_area_index = 0;
    
    std::vector <std::vector <cv::Point>> green_contour, blue_contour, red_contour;
    std::vector<cv::Point> green_contour_points, blue_contour_points, red_contour_points;

    double temp;
    const double max_distance = std::sqrt((src.cols)*(src.cols) + (src.rows)*(src.rows));
    double green_min_distance = max_distance, blue_min_distance = max_distance, red_min_distance = max_distance;
    double green_md_index = 0, blue_md_index = 0, red_md_index = 0;



    // Convert from BGR to HSV colorspace
    cv::cvtColor(res, res, cv::COLOR_BGR2HSV);

    cv::Mat lightBulb = src.clone();
	cv::cvtColor(src, lightBulb, cv::COLOR_BGR2GRAY);
	cv::threshold(lightBulb, lightBulb, 251, 255, cv::THRESH_BINARY);

	std::vector <std::vector<cv::Point>> lightBulb_contour;
	cv::findContours(lightBulb, lightBulb_contour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	for (int contour_index = 0; contour_index < lightBulb_contour.size(); contour_index++) {
        if (cv::contourArea(lightBulb_contour[contour_index], false) > lightBulb_max_c_area) {
            lightBulb_max_c_area = cv::contourArea(lightBulb_contour[contour_index]);
            lightBulb_c_max_area_index = contour_index;
        }
    }

    cv::Moments mnts = cv::moments(lightBulb_contour[lightBulb_c_max_area_index]);
    double lB_m00 = mnts.m00;
    double lB_m10 = mnts.m10;
    double lB_m01 = mnts.m01;

    unsigned int lB_mx = (lB_m10 / lB_m00);
    unsigned int lB_my = (lB_m01 / lB_m00);

    if (debug) {
        indicator(lightBulb, lB_mx, lB_my, 10);
        cv::imshow("lightBulb", lightBulb);
    }

    // paint the lamp
    cv::rectangle(res, cv::Point(lB_mx - 40, lB_my - 60), cv::Point(lB_mx + 40, lB_my + 0), cv::Scalar(255, 0, 0), -1);

    if (debug) {
        indicator(lightBulb, lB_mx, lB_my, 10);
        cv::imshow("lightBulb", lightBulb);
    }

    cv::inRange(res, cv::Scalar(low_green_H, low_green_S, low_green_V), cv::Scalar(high_green_H, high_green_S, high_green_V), green);
	cv::inRange(res, cv::Scalar(low_blue_H, low_blue_S, low_blue_V), cv::Scalar(high_blue_H, high_blue_S, high_blue_V), blue);
    cv::inRange(res, cv::Scalar(low_red_H, low_red_S, low_red_V), cv::Scalar(high_red_H, high_red_S, high_red_V), red);
	


    // similar to task 1 and task2
    cv::findContours(green, green_contour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	cv::findContours(blue, blue_contour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	cv::findContours(red, red_contour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    

    for (int contour_index = 0; contour_index < green_contour.size(); contour_index++) {
        if (cv::contourArea(green_contour[contour_index], false) > green_max_c_area) {
            green_max_c_area = cv::contourArea(green_contour[contour_index]);
        }
    }

    for (int contour_index = 0; contour_index < blue_contour.size(); contour_index++) {
        if (cv::contourArea(blue_contour[contour_index], false) > blue_max_c_area) {
            blue_max_c_area = cv::contourArea(blue_contour[contour_index]);
        }
    }

    for (int contour_index = 0; contour_index < red_contour.size(); contour_index++) {
        if (cv::contourArea(red_contour[contour_index], false) > red_max_c_area) {
            red_max_c_area = cv::contourArea(red_contour[contour_index]);
        }
    }

    if (debug) {
        std::cout << "max green zone: " << green_max_c_area << " max blue zone: " << blue_max_c_area << " max red zone: " << red_max_c_area << std::endl;
    }

    for (int contour_index = 0; contour_index < green_contour.size(); contour_index++) {
        if (cv::contourArea(green_contour[contour_index], false) < 0.1 * green_max_c_area) {
            green_contour.erase(green_contour.begin() + contour_index);
            contour_index--;
        }
        else
        {
            cv::Moments mnts = cv::moments(green_contour[contour_index]);
            double m00 = mnts.m00;
            double m10 = mnts.m10;
            double m01 = mnts.m01;

            unsigned int mx = (m10 / m00);
            unsigned int my = (m01 / m00);
            indicator(src, mx, my, 10);
            green_contour_points.push_back(cv::Point(mx, my));
        }
    }

    for (int contour_index = 0; contour_index < blue_contour.size(); contour_index++) {
        if (cv::contourArea(blue_contour[contour_index], false) < 0.1 * blue_max_c_area) {
            blue_contour.erase(blue_contour.begin() + contour_index);
            contour_index--;
        }
        else
        {
            cv::Moments mnts = cv::moments(blue_contour[contour_index]);
            double m00 = mnts.m00;
            double m10 = mnts.m10;
            double m01 = mnts.m01;

            unsigned int mx = (m10 / m00);
            unsigned int my = (m01 / m00);
            indicator(src, mx, my, 10);
            blue_contour_points.push_back(cv::Point(mx, my));
        }
    }

    for (int contour_index = 0; contour_index < red_contour.size(); contour_index++) {
        if (cv::contourArea(red_contour[contour_index], false) < 0.05 * red_max_c_area) {
            red_contour.erase(red_contour.begin() + contour_index);
            contour_index--;
        }
        else
        {
            cv::Moments mnts = cv::moments(red_contour[contour_index]);
            double m00 = mnts.m00;
            double m10 = mnts.m10;
            double m01 = mnts.m01;

            unsigned int mx = (m10 / m00);
            unsigned int my = (m01 / m00);
            indicator(src, mx, my, 10);
            red_contour_points.push_back(cv::Point(mx, my));
        }
    }


    cv::drawContours(src, green_contour, -1, cv::Scalar(0, 255, 0), 2);
	cv::drawContours(src, blue_contour, -1, cv::Scalar(255, 0, 0), 2);
	cv::drawContours(src, red_contour, -1, cv::Scalar(0, 0, 255), 2);

    for (int contour_index = 0; contour_index < green_contour.size(); contour_index++) {
    temp = std::sqrt(
        ((std::abs(green_contour_points[contour_index].x - (int)lB_mx)) * 
        (std::abs(green_contour_points[contour_index].x - (int)lB_mx))) + 
        ((std::abs(green_contour_points[contour_index].y - (int)lB_my)) * 
        (std::abs(green_contour_points[contour_index].y - (int)lB_my)))
    );
        if (temp < green_min_distance) {
            green_min_distance = temp;
            green_md_index = contour_index;
        }
    }

    for (int contour_index = 0; contour_index < blue_contour.size(); contour_index++) {
    temp = std::sqrt(
        ((std::abs(blue_contour_points[contour_index].x - (int)lB_mx)) * 
        (std::abs(blue_contour_points[contour_index].x - (int)lB_mx))) + 
        ((std::abs(blue_contour_points[contour_index].y - (int)lB_my)) * 
        (std::abs(blue_contour_points[contour_index].y - (int)lB_my)))
    );
        if (temp < blue_min_distance) {
            blue_min_distance = temp;
            blue_md_index = contour_index;
        }
    }

    for (int contour_index = 0; contour_index < red_contour.size(); contour_index++) {
    temp = std::sqrt(
        ((std::abs(red_contour_points[contour_index].x - (int)lB_mx)) * 
        (std::abs(red_contour_points[contour_index].x - (int)lB_mx))) + 
        ((std::abs(red_contour_points[contour_index].y - (int)lB_my)) * 
        (std::abs(red_contour_points[contour_index].y - (int)lB_my)))
    );
        if (temp < red_min_distance) {
            red_min_distance = temp;
            red_md_index = contour_index;
        }
    }

    if (debug) {
        std::cout << "lightBuld zone: " << lightBulb_max_c_area << std::endl;
        std::cout << "green_min_distance zone: " << green_min_distance << std::endl;
        std::cout << "green_md_index zone: " << green_md_index << std::endl;
        std::cout << "blue_min_distance zone: " << blue_min_distance << std::endl;
        std::cout << "blue_md_index zone: " << blue_md_index << std::endl;
        std::cout << "red_min_distance zone: " << red_min_distance << std::endl;
        std::cout << "red_md_index zone: " << red_md_index << std::endl;
    }

    cv::line(src, 
        cv::Point(green_contour_points[green_md_index].x, green_contour_points[green_md_index].y), 
        cv::Point(lB_mx, lB_my), 
        cv::Scalar(0, 255, 0),
        3
    );

    cv::line(src, 
        cv::Point(blue_contour_points[blue_md_index].x, blue_contour_points[blue_md_index].y), 
        cv::Point(lB_mx, lB_my),
        cv::Scalar(255, 0, 0),
        3
    );

    cv::line(src, 
        cv::Point(red_contour_points[red_md_index].x, red_contour_points[red_md_index].y), 
        cv::Point(lB_mx, lB_my), 
        cv::Scalar(0, 0, 255),
        3
    );

    cv::putText(
        src, 
        std::to_string(green_min_distance), 
        cv::Point((green_contour_points[green_md_index].x + lB_mx) / 2 - 50, (green_contour_points[green_md_index].y + lB_my) / 2 - 50), 
        cv::FONT_HERSHEY_COMPLEX, 0.6, 
        cv::Scalar(0, 255, 0), 
        1.5);

    cv::putText(
        src, 
        std::to_string(blue_min_distance), 
        cv::Point((blue_contour_points[blue_md_index].x + lB_mx) / 2 - 150, (blue_contour_points[blue_md_index].y + lB_my) / 2 + 50), 
        cv::FONT_HERSHEY_COMPLEX, 0.6, 
        cv::Scalar(255, 0, 0), 
        1.5);

    cv::putText(
        src, 
        std::to_string(red_min_distance), 
        cv::Point((red_contour_points[red_md_index].x + lB_mx) / 2 + 50, (red_contour_points[red_md_index].y + lB_my) / 2 + 50), 
        cv::FONT_HERSHEY_COMPLEX, 0.6, 
        cv::Scalar(0, 0, 255), 
        1.5);

    if (debug) {
        cv::imshow("green", green);
        cv::imshow("blue", blue);
        cv::imshow("red", red); 
    }
    // cv::waitKey(-1);
    //cv::imshow("src", src);
}

void wrenchless(cv::Mat& src, cv::Mat& etalon, bool debug = true) {
    cv::Mat res = src.clone();
    cv::Mat etalon_res = etalon.clone();

    cv::cvtColor(etalon, etalon_res, cv::COLOR_BGR2GRAY);
    cv::morphologyEx(etalon_res, etalon_res, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    cv::threshold(etalon_res, etalon_res, 240, 255, cv::THRESH_BINARY_INV);

    std::vector <std::vector <cv::Point>> etalon_wrench_contour;
    cv::findContours(etalon_res, etalon_wrench_contour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::drawContours(etalon_res, etalon_wrench_contour, -1, cv::Scalar(0, 255, 0), 2);

    cv::Moments mnts = cv::moments(etalon_wrench_contour[0]);
    double m00 = mnts.m00;
    double m10 = mnts.m10;
    double m01 = mnts.m01;

    unsigned int mx = (m10 / m00);
    unsigned int my = (m01 / m00);
    indicator(etalon, mx, my, 10);
    double etalon_area = cv::contourArea(etalon_wrench_contour[0]);
    double etalon_perimeter = m00;

    if (debug) {
        std::cout << "etalon_area: " << etalon_area << std::endl;
        std::cout << "etalon_perimetr: " << etalon_perimeter << std::endl;
    }

    cv::cvtColor(src, res, cv::COLOR_BGR2GRAY);
    cv::threshold(res, res, 230, 255, cv::THRESH_BINARY_INV);
    std::vector <std::vector <cv::Point>> wrenchs_contour;
    cv::findContours(res, wrenchs_contour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    double match;

    for (int contour_index = 0; contour_index < wrenchs_contour.size(); contour_index++) {
        cv::Moments mnts = cv::moments(wrenchs_contour[contour_index]);
        double temp_area = cv::contourArea(wrenchs_contour[contour_index]);
        double match = std::abs((etalon_area / (etalon_perimeter + 10)) - (temp_area / (mnts.m00 + 10)));

        
        cv::putText(
            src, 
            std::to_string(contour_index), 
            cv::Point(mnts.m10 / mnts.m00 + 30 , mnts.m01 / mnts.m00 + 60), 
            cv::FONT_HERSHEY_COMPLEX, 1.5, 
            cv::Scalar(0, 0, 0), 
            2.0
        );
            
        if (debug) {
            std::cout << contour_index << "match: " << match << std::endl;
        }

        if (match > 0.0005 && match < 0.0006) {  
            cv::polylines(src, wrenchs_contour[contour_index], true, cv::Scalar(0, 255, 0), 5, 8);
            cv::putText(
                src, 
                "YES", 
                cv::Point(mnts.m10 / mnts.m00 - 40 , mnts.m01 / mnts.m00 - 40), 
                cv::FONT_HERSHEY_COMPLEX, 0.6, 
                cv::Scalar(0, 255, 0), 
                1.0
            );
        }
        else
        {
            cv::polylines(src, wrenchs_contour[contour_index], true, cv::Scalar(0, 0, 255), 5, 8);
            cv::putText(
                src, 
                "NO", 
                cv::Point(mnts.m10 / mnts.m00 - 40 , mnts.m01 / mnts.m00 - 50), 
                cv::FONT_HERSHEY_COMPLEX, 0.6, 
                cv::Scalar(0, 0, 255), 
                1.0
            );
        }
    }

    cv::imshow("src", src);
    cv::imshow("res", res);
    cv::imshow("etalon", etalon);
    cv::imshow("etalon_res", etalon_res);
    cv::waitKey(-1);
}



int main() {
    // task 1
    cv::Mat img1 = cv::imread("/home/gerzeg/Polytech/TV/lb3/img_zadan/allababah/ig_0.jpg");
    cv::Mat img2 = cv::imread("/home/gerzeg/Polytech/TV/lb3/img_zadan/allababah/ig_1.jpg");
    cv::Mat img3 = cv::imread("/home/gerzeg/Polytech/TV/lb3/img_zadan/allababah/ig_2.jpg");
    cv::Mat temp;

    /*
    cv::imshow("original img1", img1);
    cv::imshow("original img2", img2);
    cv::imshow("original img3", img3);
    alababah(img1, temp, 227);
    alababah(img2, temp, 227);
    alababah(img3, temp, 227);
    */

    // task2
    cv::Mat img4 = cv::imread("/home/gerzeg/Polytech/TV/lb3/img_zadan/teplovizor/21331.res.jpg");
    cv::Mat img5 = cv::imread("/home/gerzeg/Polytech/TV/lb3/img_zadan/teplovizor/445923main_STS128_16FrAvg_color1.jpg");
    cv::Mat img6 = cv::imread("/home/gerzeg/Polytech/TV/lb3/img_zadan/teplovizor/MW-AW129-measured.jpg");
    cv::Mat img7 = cv::imread("/home/gerzeg/Polytech/TV/lb3/img_zadan/teplovizor/ntcs_quest_measurement.png");
    cv::Mat img8 = cv::imread("/home/gerzeg/Polytech/TV/lb3/img_zadan/teplovizor/size0-army.mil-2008-08-28-082221.jpg");

    /*
    cv::imshow("original img4", img4);
    cv::imshow("original img5", img5);
    cv::imshow("original img6", img6);
    cv::imshow("original img7", img7);
    cv::imshow("original img8", img8);  
    democratization(img4);
    democratization(img5);
    democratization(img6);
    democratization(img7);
    democratization(img8);
    */
    

    // task3
    cv::Mat img9 = cv::imread("/home/gerzeg/Polytech/TV/lb3/img_zadan/roboti/roi_robotov.jpg");
    cv::Mat img10 = cv::imread("/home/gerzeg/Polytech/TV/lb3/img_zadan/roboti/roi_robotov_1.jpg");

    /*
    cv::imshow("original img9", img9);
    cv::imshow("original img10", img10);
    robots(img9);
    robots(img10);
    */

    // task4
    cv::Mat img11 = cv::imread("/home/gerzeg/Polytech/TV/lb3/img_zadan/gk/gk.jpg");
    cv::Mat img12 = cv::imread("/home/gerzeg/Polytech/TV/lb3/img_zadan/gk/gk_tmplt.jpg");

    /*
    cv::imshow("original img11", img11);
    cv::imshow("original img12", img12);
    wrenchless(img11, img12);
    */

    // cv::VideoCapture cap("/home/gerzeg/Polytech/TV/lb3/src_videous/Robot Swarm - University of Sheffield (HD).mp4");
    cv::VideoCapture cap("/home/gerzeg/Polytech/TV/lb3/src_videous/2.mp4");
    
    if (!cap.isOpened()) {
        std::cout << "error video" << std::endl;
    }

    int f_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int f_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::VideoWriter video("dst.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        10, cv::Size(f_width, f_height));

    int count = 0;
    while(true) {
        cv::Mat frame;
        cap >> frame;
        std::cout << "frame " + std::to_string(count) + " processed" << std::endl;
        if (!frame.empty()) {
            robots(frame);
            video.write(frame);
            cv::imshow("Frame", frame);
            count ++;
        }
        char c = (char)cv::waitKey(50);
        if (count == 466)
            break;
        
    }
    cap.release();
    video.release();
    cv::destroyAllWindows();
}