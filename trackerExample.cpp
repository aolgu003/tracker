#include "opencv2/opencv.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include <iostream>
#include "klttracker.h"
#include "string"

using namespace cv;
using namespace std;

int main()
{
  cout << "OpenCV Version" << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << endl;
  VideoCapture cap;
  if(!cap.open(0))
      return 0;

  int frameIndex = 0;
  Mat frame;
  cap >> frame;
  cout << "Starting tracker" << endl;
  kltTracker tracker(frame);
  vector<vector<pointHistory> > lostTrackBuffer;

  while (1)
  {
    cap >> frame;
    cv::Mat color_frame;
    Mat featureImage = tracker.update(frame);
    cout << "Getting track buffer" << endl;
    lostTrackBuffer = tracker.getAndClearLostTrackBuffer();
    auto current_tracks = tracker.GetCurrentTracks();
    std::cout << current_tracks.size() << std::endl;
    for (size_t i = 0; i < current_tracks.size(); i++) {
      auto feature_tracks = current_tracks[i];
      std::cout << feature_tracks.size() << std::endl;
      for (size_t j = 0; j < feature_tracks.size(); j++) {
        auto pt = feature_tracks[j].position;
        cv::circle(featureImage, pt, 1, cv::Scalar(255,0,0));
      }
    }
    imshow("Features image", featureImage);
    cout << "Track buffer: " << lostTrackBuffer.size() << endl;
    if( waitKey(10) == 27 ) return 0; // stop capturing by pressing ESC
  }
}
