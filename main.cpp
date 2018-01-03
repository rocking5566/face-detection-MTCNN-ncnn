#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "Mtcnn.h"

using namespace std;
using namespace cv;

void PlotDetectionResult(const Mat& frame, const std::vector<SBoundingBox>& bbox)
{
    for (vector<SBoundingBox>::const_iterator it = bbox.begin(); it != bbox.end(); it++)
    {
        if ((*it).bExist)
        {
            rectangle(frame, Point((*it).x1, (*it).y1), Point((*it).x2, (*it).y2), Scalar(0, 0, 255), 2, 8, 0);
            for (int num = 0; num < 5; num++)
            {
                circle(frame, Point((int)*(it->ppoint + num), (int)*(it->ppoint + num + 5)), 3, Scalar(0, 255, 255), -1);
            }
        }
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
    bool bSetParamToMtcnn = false;
    mtcnn.LoadModel("det1.param", "det1.bin", "det2.param", "det2.bin", "det3.param", "det3.bin");

    double sumMs = 0;
    int count = 0;

    while (1)
    {
        cap >> frame;
        std::vector<SBoundingBox> finalBbox;

        if (!bSetParamToMtcnn && frame.cols > 0)
        {
            mtcnn.SetParam(frame.cols, frame.rows, eBGR, 90, 0.709);
            bSetParamToMtcnn = true;
        }

        double t1 = (double)getTickCount();
        mtcnn.Detect(frame.data, finalBbox);
        double t2 = (double)getTickCount();
        double t = 1000 * double(t2 - t1) / getTickFrequency();
        sumMs += t;
        ++count;
        cout << "time = " << t << " ms, FPS = " << 1000 / t << ", Average time = " << sumMs / count <<endl;

        PlotDetectionResult(frame, finalBbox);

        imshow("frame", frame);

        if (waitKey(1) == 'q')
            break;
    }

    return 0;
}
