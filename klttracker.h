#ifndef KLTTRACKER_H
#define KLTTRACKER_H
#include "opencv2/opencv.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
using namespace cv;
using namespace std;

struct pointHistory{
  int frameIndex;
  Point2f position;
};

enum featureStatus {
  failedTrack = 0,
  successfulTrack = 1
};

class kltTracker
{
public:
  kltTracker(cv::Mat initialImage);

  Mat update(cv::Mat image);
  vector<vector<pointHistory> > GetCurrentTracks() const;
  vector<vector<pointHistory> > getAndClearLostTrackBuffer();

private:
  void processTrackerResults( vector<Point2f> newPoints,
                             vector<uchar> trackStatus);
  void findNewFeatures( Mat grayImage );

  void eraseMissingFeature( vector<vector<pointHistory> >::iterator featuresHistoryIterator,
      vector<Point2f>::iterator oldPointsIterator
      );

  vector<pointHistory> addFeatureToHistory( vector<pointHistory> history,
                            pointHistory extractedFeature );

  void printTrackSizes(int numNewPoints);
  void printTrackSizes();

  void setVerbosePrintOut();
  void unsetVerbosePrintOut();

private:
  bool verbose_printouts = false;
  Mat featureImage, oldFrame;

  Ptr<GFTTDetector> featureDetector;
  vector<Point2f> oldPoints;

  vector<vector<pointHistory> > featuresHistory;
  vector<vector<pointHistory> > lostTrackBuffer;

  int frameIndex = 0;
  TermCriteria termcrit;
  Size subPixWinSize;

};

#endif // KLTTRACKER_H
