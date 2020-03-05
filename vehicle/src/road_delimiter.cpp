#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include "vehicle/VanishingPoint.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>



void callback(const sensor_msgs::ImageConstPtr &image, const vehicle::VanishingPoint::ConstPtr &vp, const ros::Publisher *pub){
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(image);
    int x = vp->col;
    int y = vp->row;
    cv::Rect road(0, y-50, cv_ptr->image.cols, cv_ptr->image.rows -y+50);
    cv::Mat grey_image;
    cv::cvtColor(cv_ptr->image, grey_image, CV_BGR2GRAY);
    grey_image = grey_image(road);
    
    
    //cv::imshow("pub", grey_image);
    //cv::waitKey(3);
    cv_bridge::CvImage img_pub(std_msgs::Header(), "mono8", grey_image);
    pub->publish(img_pub.toImageMsg());

}

int main(int argc, char** argv){
    ros::init(argc, argv, "road_delimiter");
    ros::NodeHandle nh;

    message_filters::Subscriber<sensor_msgs::Image> image_sub (nh, "/multisense/left/image_rect_color", 1);
    message_filters::Subscriber<vehicle::VanishingPoint> vp_sub (nh, "/vanishing_point_detector/vanishing_point", 1);
    ros::Publisher road_delimiter = nh.advertise<sensor_msgs::Image>("road_delimiter/road_estimation", 1);
    message_filters::TimeSynchronizer<sensor_msgs::Image, vehicle::VanishingPoint> sync (image_sub, vp_sub, 5);
    sync.registerCallback(boost::bind(&callback, _1, _2, &road_delimiter));
    ros::spin();
}