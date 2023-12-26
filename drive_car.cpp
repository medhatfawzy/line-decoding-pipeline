#include "vehicles/car/api/CarApiBase.hpp"
#include "common/Common.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <filesystem>

using namespace msr::airlib;

// Function prototypes
void process_img(const cv::Mat &img_gray, float &throttle, float &steer);

int main() {
    // Initialize the AirSim client
    msr::airlib::CarApiBase::CarControls car_controls;
    msr::airlib::CarApiBase::CarState car_state;
    msr::airlib::CarApiBase client;

    // Connect to the AirSim simulator
    client.confirmConnection();
    client.enableApiControl(true);
    std::cout << "API Control enabled: " << client.isApiControlEnabled() << std::endl;

    // Get state of the car
    car_state = client.getCarState();
    std::cout << "Speed: " << car_state.speed << ", Gear: " << car_state.gear << std::endl;

    // Setup temporary directory for saving images
    std::string tmp_dir = std::filesystem::temp_directory_path().string() + "/airsim_car";
    std::filesystem::create_directory(tmp_dir);

    // Drive logic
    car_controls.throttle = 0.5;
    car_controls.steering = 0;
    client.setCarControls(car_controls);
    std::this_thread::sleep_for(std::chrono::seconds(1)); // Let car drive a bit
    std::cout << "Go Forward" << std::endl;

    int idx = 0;
    while (client.getCarState().speed > 0) {
        // Get camera images from the car
        std::vector<ImageCaptureBase::ImageRequest> request = {
            ImageCaptureBase::ImageRequest("front_center", ImageCaptureBase::ImageType::Scene, false, false)
        };
        const std::vector<ImageCaptureBase::ImageResponse>& response = client.simGetImages(request);

        // Convert to OpenCV format
        cv::Mat img_rgb = cv::Mat(response.at(0).height, response.at(0).width, CV_8UC3, response.at(0).image_data_uint8.data());
        cv::Mat img_gray;
        cv::cvtColor(img_rgb, img_gray, cv::COLOR_BGR2GRAY);

        float throttle, steer;
        process_img(img_gray, throttle, steer);

        std::string filename = tmp_dir + "/curved_" + std::to_string(idx) + ".png";
        cv::imwrite(filename, img_rgb);
        std::cout << "Throttle: " << throttle << ", Steer: " << steer << std::endl;

        if (throttle < 0) {
            car_controls.brake = 1.0;
            car_controls.throttle = 0;
            car_controls.steering = 0;
            client.setCarControls(car_controls);
            break;
        } else {
            car_controls.throttle = throttle;
            car_controls.steering = steer;
            client.setCarControls(car_controls);
        }

        idx++;
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Let car drive a bit
    }

    // Disable API control before exiting
    client.enableApiControl(false);

    return 0;
}

// Placeholder for image processing function
void process_img(const cv::Mat &img_gray, float &throttle, float &steer) {
    // Implement your image processing and control logic here
    // This is just a placeholder function
    throttle = 0.5; // Example value
    steer = 0;      // Example value
}
