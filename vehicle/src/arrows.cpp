#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include "vehicle/VanishingPoint.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>


void callback(const sensor_msgs::ImageConstPtr& image, const vehicle::VanishingPoint::ConstPtr& vp, const ros::Publisher *pub)
{
	cv_bridge::CvImagePtr cv_ptr;
	cv_ptr = cv_bridge::toCvCopy(image);
	int x = vp->col;
	int y = vp->row;
	cv::arrowedLine(cv_ptr->image, cv::Point(512, 544), cv::Point(x, y), CV_RGB(255, 0, 0), 10, 8, 0.1);
	pub->publish(cv_ptr->toImageMsg());

}
int main(int argc, char** argv)
{
	ros::init(argc, argv, "arrow_drawer");
	ros::NodeHandle nh;
	message_filters::Subscriber<sensor_msgs::Image> image_sub (nh, "/multisense/left/image_rect_color", 1);
	message_filters::Subscriber<vehicle::VanishingPoint> vp_sub(nh, "/vanishing_point_detector/vanishing_point", 1);
	ros::Publisher arrow_pub = nh.advertise<sensor_msgs::Image>("arrow_drawer/vp_arrow", 1);
	message_filters::TimeSynchronizer<sensor_msgs::Image, vehicle::VanishingPoint> sync(image_sub, vp_sub, 5);
	sync.registerCallback(boost::bind(&callback, _1, _2, &arrow_pub));
	ros::spin();
}

