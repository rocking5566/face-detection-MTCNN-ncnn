#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "Mtcnn.h"
#include "HeadPosePredictor.h"

using namespace std;
using namespace cv;


void PlotDetectionResult(const Mat& frame, 
                        const std::vector<SMtcnnFace>& bbox_vec, 
                        const std::vector<SHeadPoseInfo>& headPoseVec)
{
    for (int i = 0; i < bbox_vec.size(); ++i)
    {
        // Plot bounding box
        rectangle(frame, Point(bbox_vec[i].boundingBox[0], bbox_vec[i].boundingBox[1]), 
            Point(bbox_vec[i].boundingBox[2], bbox_vec[i].boundingBox[3]), Scalar(0, 0, 255), 2, 8, 0);

        // Plot facial landmark
        for (int num = 0; num < 5; num++)
        {
            circle(frame, Point(bbox_vec[i].landmark[num], bbox_vec[i].landmark[num + 5]), 3, Scalar(0, 255, 255), -1);
        }

        // Plot head pose
        cv::putText(frame, "Yaw:" + to_string(headPoseVec[i].facialUnitNormalVector[0]), 
                        cv::Point2f(bbox_vec[i].boundingBox[0], 
                        max(bbox_vec[i].boundingBox[1] - 90, 0)), CV_FONT_HERSHEY_PLAIN, 1.5, CV_RGB(0, 0, 255), 2);
        cv::putText(frame, "Pitch:" + to_string(headPoseVec[i].facialUnitNormalVector[1]), 
                        cv::Point2f(bbox_vec[i].boundingBox[0], 
                        max(bbox_vec[i].boundingBox[1] - 60, 0)), CV_FONT_HERSHEY_PLAIN, 1.5, CV_RGB(0, 0, 255), 2);
        cv::putText(frame, "Roll:" + to_string(headPoseVec[i].facialUnitNormalVector[2]), 
                        cv::Point2f(bbox_vec[i].boundingBox[0], 
                        max(bbox_vec[i].boundingBox[1] - 30, 0)), CV_FONT_HERSHEY_PLAIN, 1.5, CV_RGB(0, 0, 255), 2);

        // Plot facial normal.
        cv::Point noseTip = cv::Point(bbox_vec[i].landmark[2], bbox_vec[i].landmark[7]);
        cv::line(frame, noseTip, cv::Point(noseTip.x + 100 * headPoseVec[i].facialUnitNormalVector[0],
            noseTip.y + 100 * headPoseVec[i].facialUnitNormalVector[1]), cv::Scalar(0, 255, 0), 2);
    }
}

int main(int argc, char** argv)
{
    VideoCapture cap(0);

    if (!cap.isOpened())
    {
        cout << "video is not open" << endl;
        return -1;
    }

    Mat frame;
    CMtcnn mtcnn;
    CHeadPosePredictor headPosePredictor;
    bool bSetParamToMtcnn = false;
    mtcnn.LoadModel("det1.param", "det1.bin", "det2.param", "det2.bin", "det3.param", "det3.bin");

    double sumMs = 0;
    int count = 0;

    while (1)
    {
        cap >> frame;
        std::vector<SMtcnnFace> finalBbox_vec;
        std::vector<SHeadPoseInfo> headPose_vec;

        if (!bSetParamToMtcnn && frame.cols > 0)
        {
            SImageFormat format(frame.cols, frame.rows, eBGR888);
            const float faceScoreThreshold[3] = { 0.6f, 0.6f, 0.6f };
            mtcnn.SetParam(format, 90, 0.709, 4, faceScoreThreshold);
            bSetParamToMtcnn = true;
        }

        double t1 = (double)getTickCount();
        mtcnn.Detect(frame.data, finalBbox_vec);
        headPose_vec.resize(finalBbox_vec.size());
        for (int i = 0; i < finalBbox_vec.size(); ++i) {
            headPose_vec[i] = headPosePredictor.Predict(finalBbox_vec[i].landmark);
        }

        double t2 = (double)getTickCount();
        double t = 1000 * double(t2 - t1) / getTickFrequency();
        sumMs += t;
        ++count;
        cout << "time = " << t << " ms, FPS = " << 1000 / t << ", Average time = " << sumMs / count <<endl;

        PlotDetectionResult(frame, finalBbox_vec, headPose_vec);

        imshow("frame", frame);

        if (waitKey(1) == 'q')
            break;
    }

    return 0;
}
