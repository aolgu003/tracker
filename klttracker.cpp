#include "klttracker.h"

kltTracker::kltTracker(cv::Mat initialImage):
  termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03),
  subPixWinSize(10,10)
{
  vector<KeyPoint> initialFeatures;
  vector<Point2f> initialPoints;
  vector<pointHistory> initialHistory;

  featureDetector->setMaxFeatures(200);

  cvtColor(initialImage, grayImage, CV_BGR2GRAY);
  featureDetector->detect( grayImage, initialFeatures );
  KeyPoint::convert(initialFeatures, initialPoints);
  cornerSubPix(grayImage, initialPoints, subPixWinSize, Size(-1,-1), termcrit);

  for (int i = 0; i < initialPoints.size(); i++)
  {
    pointHistory extractedFeature;
    extractedFeature.frameIndex = frameIndex;
    extractedFeature.position = oldPoints[i];

    initialHistory.push_back(extractedFeature);
    featuresHistory.push_back(initialHistory);
    initialHistory.clear();
  }
  frameIndex++;
}

Mat kltTracker::update(Mat image)
{
  Mat grayImage;
  image.copyTo(featureImage);

  cvtColor(image, grayImage, CV_BGR2GRAY);
  cout << "---------------- Tracking features ------------------" << endl;
  vector<float> err;
  vector<Point2f> pointsNew;
  vector<uchar> status;

  calcOpticalFlowPyrLK(oldFrame, grayImage, oldPoints, pointsNew, status, err);
  processTrackerResults( pointsNew, status );
  printTrackSizes( (int) pointsNew.size() );
  status.clear();

  cout << "------------ Detecting features -----------------" << endl;
  findNewFeatures();
  grayImage.copyTo( oldFrame );
  frameIndex++;

  return featureImage;
}

vector<vector<pointHistory> > kltTracker::getAndClearLostTrackBuffer()
{
  vector<vector<pointHistory>> lostTrackBufferTemp = lostTrackBuffer;
  lostTrackBuffer.clear();
  return lostTrackBufferTemp;
}

void kltTracker::processTrackerResults( vector<Point2f> newPoints,
                                        vector<uchar> trackStatus)
{
  int i;
  vector<Point2f>::iterator oldPointsIterator, newPointIterator;
  vector<vector<pointHistory>>::iterator featuresHistoryIterator;
  for ( // loop initialization
        i = 0,
        oldPointsIterator = oldPoints.begin(),
        newPointIterator = newPoints.begin(),
        featuresHistoryIterator = featuresHistory.begin();
        // loop termination condition
        oldPointsIterator != oldPoints.end();
        // Iterate
        i++,
        newPointIterator++ )
  {
    pointHistory extractedFeature;
    extractedFeature.frameIndex = frameIndex;
    extractedFeature.position = *newPointIterator;
    if (trackStatus[i] == featureStatus::failedTrack)
    {
      eraseMissingFeature(featuresHistoryIterator, oldPointsIterator);
    }
    else
    {
      circle( featureImage, *newPointIterator, 3, Scalar(0,255,0), -1, 8);

      *oldPointsIterator = *newPointIterator;
      *featuresHistoryIterator = addFeatureToHistory(*featuresHistoryIterator,
                                                     extractedFeature);
      oldPointsIterator++;
      featuresHistoryIterator++;
    }
  }
}

void kltTracker::findNewFeatures()
{
  if (oldPoints.size() < 200)
  {
    vector<KeyPoint> detectedFeatures;
    featureDetector->setMaxFeatures(200-oldPoints.size());
    featureDetector->detect( grayImage, detectedFeatures );

    vector<Point2f> detectedPoints;
    KeyPoint::convert(detectedFeatures, detectedPoints);
    oldPoints.insert(oldPoints.end(), detectedPoints.begin(), detectedPoints.end() );

    vector<pointHistory> history;
    for (int i = 0; i < detectedPoints.size(); i++)
    {
      pointHistory extractedFeature;
      extractedFeature.frameIndex = frameIndex;
      extractedFeature.position = detectedPoints[i];

      history.push_back(extractedFeature);
      featuresHistory.push_back(history);
      history.clear();
    }
    printTrackSizes();
  }
}

void kltTracker::eraseMissingFeature(
    vector<vector<pointHistory>>::iterator featuresHistoryIterator,
    vector<Point2f>::iterator oldPointsIterator
    )
{
  lostTrackBuffer.push_back(*featuresHistoryIterator);
  featuresHistory.erase(featuresHistoryIterator);
  oldPoints.erase(oldPointsIterator);
}

vector<pointHistory> kltTracker::addFeatureToHistory(
    vector<pointHistory> history,
    pointHistory extractedFeature)
{
  history.push_back(extractedFeature);
  return history;
}

void kltTracker::printTrackSizes(int numNewPoints)
{
  cout << "Old feature vector size: " << oldPoints.size() << endl;
  cout << "New feature vector size: " << numNewPoints << endl;
  cout << "Feature History size: " << featuresHistory.size() << endl;
}
void kltTracker::printTrackSizes()
{
  cout << "Old feature vector size: " << oldPoints.size() << endl;
  cout << "Feature History size: " << featuresHistory.size() << endl;
}
