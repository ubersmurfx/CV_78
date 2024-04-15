#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>


int main() {
    double amplitude = 100;
    double freq = 0.03;
    double phase = 0;
    double vert = 200;

    cv::Mat background = cv::imread("/home/gerzeg/Polytech/TV/lb1/main_task/images/background_1.jpg");
    cv::Mat car = cv::imread("/home/gerzeg/Polytech/TV/lb1/main_task/images/model_car_1.jpg");
    cv::Mat resized_car;
    cv::Mat background_animation;
    background.copyTo(background_animation);

    // main car
    cv::resize(car, resized_car, cv::Size(20, 20));

    cv::Mat animation = background_animation(cv::Rect(0,0, 660, 440));

    int x = 0;
    int y = 200;

    while(true){
        x++;
        y = amplitude * std::cos(freq * x + phase) + vert;
        cv::circle(animation, cv::Point(x, y), 1, (255, 0, 0), 1);

        cv::Mat roi = background(cv::Rect(x ,y, 20, 20));
        cv::Mat roi2 = animation(cv::Rect(x,y, 20, 20));

        resized_car.copyTo(roi2);

        cv::imshow("Window", animation);
        if (x == 330) cv::imwrite("middle.jpg", animation);
        if (x>640) break;

        roi.copyTo(animation(cv::Rect(x,y, 20, 20)));

        cv::waitKey(10);
    }
}