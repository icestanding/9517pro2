#include <iostream>
#include <vector>
#include <cv.h>
#include <highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/stitcher.hpp>
#include <opencv2/nonfree/features2d.hpp> //Thanks to Alessandro




using namespace cv;


int main(int argc, char* argv[])
{
    auto size = cv::Size(400, 400);
    auto img = cv::imread("/Users/chenyu/simple/123.jpg",  0);
    cv::resize(img, img, size);

    // create detector && descriptor
    cv::SiftFeatureDetector Detector;
    cv::SiftDescriptorExtractor Descriptor;

    std::vector<cv::KeyPoint> feature;
    cv::Mat feature_descriptor;

    Detector.detect(img, feature);
    Descriptor.compute(img, feature, feature_descriptor);


    // second image
    auto img1 = cv::imread("/Users/chenyu/simple/456.jpg",  0);
    cv::resize(img1, img1, size);

    std::vector<cv::KeyPoint> feature1;
    cv::Mat feature_descriptor1;

    Detector.detect(img1, feature1);
    Descriptor.compute(img1, feature1, feature_descriptor1);




    std::vector<std::vector<cv::DMatch>> matches;


    // cross check
    auto matcher =  cv::BFMatcher();
    // cross checking
    matcher.knnMatch(feature_descriptor, feature_descriptor1, matches, 1);
//    matcher.knnMatch(feature_descriptor, feature_descriptor1, matches, 1);
    std::vector<cv::DMatch> good_matches;

    // compare first close element with second element
    for (unsigned int i = 0; i < matches.size(); ++i)
    {

        const float ratio = 0.45; // As in Lowe's paper; can be tuned
//        compare first matches with second
        if (matches[i][0].distance < ratio * matches[i][1].distance)
        {
            good_matches.push_back(matches[i][0]);
        }

    }







    //-- Draw only "good" matches
    cv::Mat img_matches;
    drawMatches( img, feature, img1, feature1,
                 good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                 std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    imshow( "Good Matches", img_matches);

    waitKey(0);
    return 0;
}