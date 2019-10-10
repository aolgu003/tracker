#include "klttracker.h"

kltTracker::kltTracker(cv::Mat initialImage):
  termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03),
  subPixWinSize(10,10)
{
  Mat grayImage;
  cvtColor(initialImage, grayImage, CV_BGR2GRAY);

  cout << "Initializing Tracker" << endl;
  featureDetector = GFTTDetector::create(200, .01, 2);

  vector<KeyPoint> initialFeatures;
  featureDetector->detect( grayImage, initialFeatures );

  vector<Point2f> initialPoints;
  KeyPoint::convert(initialFeatures, oldPoints);
  cornerSubPix(grayImage, oldPoints, subPixWinSize, Size(-1,-1), termcrit);

  cout << "Detected Features" << endl;
  vector<pointHistory> initialHistory;
  for (int i = 0; i < oldPoints.size(); i++)
  {
    pointHistory extractedFeature;
    extractedFeature.frameIndex = frameIndex;
    extractedFeature.position = oldPoints[i];

    initialHistory.push_back(extractedFeature);
    featuresHistory.push_back(initialHistory);
    initialHistory.clear();
  }

  grayImage.copyTo( oldFrame );
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
  if (verbose_printouts) {
    printTrackSizes( (int) pointsNew.size() );
  }
  status.clear();

  cout << "------------ Detecting features -----------------" << endl;
  findNewFeatures(grayImage);
  grayImage.copyTo( oldFrame );
  frameIndex++;

  return featureImage;
}

vector<vector<pointHistory> > kltTracker::getAndClearLostTrackBuffer()
{
  vector<vector<pointHistory> > lostTrackBufferTemp = lostTrackBuffer;
  lostTrackBuffer.clear();
  return lostTrackBufferTemp;
}

vector<vector<pointHistory> > kltTracker::GetCurrentTracks() const {
  return featuresHistory;
}

void kltTracker::processTrackerResults( vector<Point2f> newPoints,
                                        vector<uchar> trackStatus)
{
  int i;
  vector<Point2f>::iterator oldPointsIterator, newPointIterator;
  vector<vector<pointHistory> >::iterator featuresHistoryIterator;
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
    if (trackStatus[i] == failedTrack)
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

void kltTracker::findNewFeatures(Mat grayImage)
{
  if (oldPoints.size() < 200)
  {
    vector<KeyPoint> detectedFeatures;
    featureDetector->setMaxFeatures(200-oldPoints.size());
    featureDetector->detect( grayImage, detectedFeatures );

    vector<Point2f> detectedPoints;
    KeyPoint::convert(detectedFeatures, detectedPoints);
    //cornerSubPix(grayImage, detectedPoints, subPixWinSize, Size(-1,-1), termcrit);

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
    if (verbose_printouts) {
      printTrackSizes();
    }
  }
}

void kltTracker::eraseMissingFeature(
    vector<vector<pointHistory> >::iterator featuresHistoryIterator,
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

void kltTracker::setVerbosePrintOut()
{
  verbose_printouts = true;
}

void kltTracker::unsetVerbosePrintOut()
{
  verbose_printouts = false;
}

