#include "opencv2/opencv.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include <iostream>
#include "string"

using namespace cv;
using namespace std;

struct pointHistory{
  int frameIndex;
  Point2f position;
};

int main()
{
  cout << "OpenCV Version" << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << endl;
  VideoCapture cap;
  if(!cap.open(0))
      return 0;
  Mat frameNew, frameOld, grayImage;

  Ptr<GFTTDetector> featureDetector = GFTTDetector::create(200, .01, 2);
  //tracker->setFlags();

  vector<KeyPoint> initialFeatures, detectedFeatures, currentFeatures;
  vector<Point2f> pointsOld, pointsNew;
  vector<Point2f> points[2];
  vector<Point2f>::iterator pointsOldIterator, pointsNewIterator;
  vector<uchar> status;

  vector<vector<pointHistory>> featuresHistory;
  vector<vector<pointHistory>>::iterator featuresHistoryIterator;

  int frameIndex = 0;
  cout << "------------ Detecting Initial features -----------------" << endl;

  cap >> frameOld;
  cvtColor(frameOld, grayImage, CV_BGR2GRAY);
  //frameOld = grayImage;
  featureDetector->setMaxFeatures(200);
  featureDetector->detect( grayImage, initialFeatures );

  vector<Point2f> initialPoints;
  KeyPoint::convert(initialFeatures, initialPoints);
  //KeyPoint::convert(initialFeatures, points[0]);

  TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
  Size subPixWinSize(10,10), winSize(31,31);
  cornerSubPix(grayImage, initialPoints, subPixWinSize, Size(-1,-1), termcrit);
  grayImage.copyTo(frameOld);
  pointsOld.insert(pointsOld.end(), initialPoints.begin(), initialPoints.end() );
  vector<pointHistory> initialHistory;

  for (int i = 0; i < initialPoints.size(); i++)
  {
    pointHistory extractedFeature;
    extractedFeature.frameIndex = frameIndex;
    extractedFeature.position = pointsOld[i];

    initialHistory.push_back(extractedFeature);
    featuresHistory.push_back(initialHistory);
    initialHistory.clear();
  }
  frameIndex++;
  while (1) {
    cap >> frameNew;
    frameIndex++;

    if( frameNew.empty() ) {
      cout << "empty frame" << endl;
      //break; // end of video stream
      return 0;
    }

    imshow("Camera frame", frameNew);
    Mat featureImage;
    frameNew.copyTo(featureImage);

    cvtColor(frameNew, grayImage, CV_BGR2GRAY);
    cout << "---------------- Tracking features ------------------" << endl;
    cout << "Old feature vector size: " << pointsOld.size() << endl;
    cout << "New feature vector size: " << pointsNew.size() << endl;
    cout << "Feature History size: " << featuresHistory.size() << endl;
    pointsNew.clear();

    vector<float> err;
    pointsNew.clear();
    calcOpticalFlowPyrLK(frameOld, grayImage, pointsOld, pointsNew, status, err);

    vector<pointHistory> history;
    int i = 0;

    pointsNewIterator = pointsNew.begin();
    featuresHistoryIterator = featuresHistory.begin();
    for (pointsOldIterator = pointsOld.begin(); pointsOldIterator != pointsOld.end(); )
    {
      pointHistory extractedFeature;
      extractedFeature.frameIndex = frameIndex;
      extractedFeature.position = *pointsNewIterator;
      if (status[i] == 0)
      {
        //cout << (int) status[i] << endl;

        history = *featuresHistoryIterator;
        cout << "Track length: " << history.size() << endl;
        featuresHistory.erase(featuresHistoryIterator);
        pointsOld.erase(pointsOldIterator);
      }
      else
      {
        circle( featureImage, *pointsNewIterator, 3, Scalar(0,255,0), -1, 8);

        //cout << (int) status[i] << endl;
        history = *featuresHistoryIterator;

        history.push_back(extractedFeature);
        *featuresHistoryIterator = history;
        history.clear();

        *pointsOldIterator = *pointsNewIterator;

        pointsOldIterator++;
        featuresHistoryIterator++;
      }
      pointsNewIterator++;
      i++;
    }

    imshow("Features frame", featureImage);

    cout << "Old feature vector size: " << pointsOld.size() << endl;
    cout << "New feature vector size: " << pointsNew.size() << endl;
    cout << "Feature History size: " << featuresHistory.size() << endl;
    status.clear();

    cout << "------------ Detecting features -----------------" << endl;

    if (pointsOld.size() < 200)
    {
      cout << "Old feature vector size: " << pointsOld.size() << endl;
      cout << "New feature vector size: " << pointsNew.size() << endl;
      cout << "Feature History size: " << featuresHistory.size() << endl;


      featureDetector->setMaxFeatures(200-pointsOld.size());
      featureDetector->detect( grayImage, detectedFeatures );

      vector<Point2f> detectedPoints;
      KeyPoint::convert(detectedFeatures, detectedPoints);
      pointsOld.insert(pointsOld.end(), detectedPoints.begin(), detectedPoints.end() );

      vector<pointHistory> history;
      for (int i = 0; i < detectedFeatures.size(); i++)
      {
        pointHistory extractedFeature;
        extractedFeature.frameIndex = frameIndex;
        extractedFeature.position = detectedPoints[i];

        history.push_back(extractedFeature);
        featuresHistory.push_back(history);
        history.clear();

      }
      detectedFeatures.clear();

      cout << "Old feature vector size: " << pointsOld.size() << endl;
      cout << "New feature vector size: " << pointsNew.size() << endl;
      cout << "Feature History size: " << featuresHistory.size() << endl;
    }
    grayImage.copyTo( frameOld );

    //waitKey(0);
    if( waitKey(10) == 27 ) return 0; // stop capturing by pressing ESC
  }
}
