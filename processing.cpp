#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

cv::Mat warpImg(const cv::Mat &img)
{
    int imgH = img.rows;
    int imgW = img.cols;

    int newW = 200;
    int newH = 300;
    cv::Point2f src[4] = {cv::Point2f(150, 260), cv::Point2f(450, 260),
                          cv::Point2f(imgW, imgH), cv::Point2f(0, imgH)};
    cv::Point2f dst[4] = {cv::Point2f(0, 0), cv::Point2f(newW, 0),
                          cv::Point2f(newW, newH), cv::Point2f(0, newH)};

    cv::Mat M = cv::getPerspectiveTransform(src, dst);
    cv::Mat warped;
    cv::warpPerspective(img, warped, M, cv::Size(newW, newH));

    return warped;
}

cv::Mat applyGaussianBlur(const cv::Mat &img, cv::Size kernelSize = cv::Size(9, 9))
{
    cv::Mat blurred;
    cv::GaussianBlur(img, blurred, kernelSize, 0);
    return blurred;
}

cv::Mat applyCannyEdge(const cv::Mat &img, double lowThreshold = 100, double highThreshold = 200)
{
    cv::Mat edges;
    cv::Canny(img, edges, lowThreshold, highThreshold);
    return edges;
}

cv::Mat applyDilation(const cv::Mat &img, cv::Size kernelSize = cv::Size(3, 3))
{
    cv::Mat dilated;
    cv::dilate(img, dilated, cv::getStructuringElement(cv::MORPH_RECT, kernelSize));
    return dilated;
}

std::vector<cv::Vec4i> detectHoughLines(const cv::Mat &img)
{
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(img, lines, 1, CV_PI / 180, 20, 2, 5);
    return lines;
}

cv::Mat drawHoughLines(const cv::Mat &img, const std::vector<cv::Vec4i> &lines)
{
    cv::Mat lineImg = cv::Mat::zeros(img.size(), img.type());
    for (size_t i = 0; i < lines.size(); i++)
    {
        cv::line(lineImg, cv::Point(lines[i][0], lines[i][1]),
                 cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255), 5);
    }
    return lineImg;
}

cv::RotatedRect detectCorrectMark(const cv::Mat &img)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::RotatedRect> rects;
    for (size_t i = 0; i < contours.size(); i++)
    {
        rects.push_back(cv::minAreaRect(contours[i]));
    }

    cv::Point centerBottom(img.cols / 2, img.rows / 1.5);
    double minDist = std::numeric_limits<double>::max();
    cv::RotatedRect centerRect;

    for (const auto &rect : rects)
    {
        double dist = cv::norm(cv::Point2f(rect.center) - cv::Point2f(centerBottom));
        if (dist < minDist)
        {
            minDist = dist;
            centerRect = rect;
        }
    }

    return centerRect;
}

std::pair<float, float> mapValues(const cv::RotatedRect &rect, const cv::Mat &img, float carSpeed, float carSteer)
{
    cv::Point2f center = rect.center;
    cv::Point2f imgCenter(img.cols / 2, img.rows / 1.5);

    float offset = (center.x - imgCenter.x) / imgCenter.x;

    float d1 = rect.size.width;
    float d2 = rect.size.height;
    float width = std::min(rect.size.width, rect.size.height);
    float height = std::max(rect.size.width, rect.size.height);
    float angle = rect.angle;

    width = static_cast<int>(5 * std::round(width / 5));
    angle = static_cast<int>(5 * std::round(angle / 5));

    if (angle == 0 || angle == 90 || angle == -0 || angle == -90)
    {
        angle = 0;
    }
    else if (d1 < d2)
    {
        angle = 90 - angle;
    }
    else
    {
        angle = -angle;
    }

    float throttle = std::max(width / (120 + carSteer), 0.4f);
    float steer = angle / (90 + throttle * 100 + carSpeed) + offset;

    return {throttle, steer};
}

std::pair<float, float> processImg(const cv::Mat &img, float carSpeed, float carSteer)
{
    cv::Mat warped = warpImg(img);
    cv::Mat blurred = applyGaussianBlur(warped);
    cv::Mat edges = applyCannyEdge(blurred);
    cv::Mat dilated = applyDilation(edges);

    std::vector<cv::Vec4i> lines = detectHoughLines(dilated);
    if (lines.empty())
    {
        return {-1, 0};
    }

    cv::Mat lineImg = drawHoughLines(cv::Mat::zeros(dilated.size(), dilated.type()), lines);
    cv::RotatedRect centerRect = detectCorrectMark(lineImg);

    auto [throttle, steer] = mapValues(centerRect, lineImg, carSpeed, carSteer);
    return {throttle, steer};
}
