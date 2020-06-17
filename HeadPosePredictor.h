#ifndef HeadPosePredictor_h__
#define HeadPosePredictor_h__

#include <vector>
#include "opencv2/core/core.hpp"

struct SHeadPoseInfo
{
    float yaw;
    float pitch;
    float roll;

    // Facial normal means head direction.
    float facialUnitNormalVector[3];    // 0: x-axis, 1: y-axis, 2: z-axis
};


// Implementation of Determining the gaze of faces in images
class CHeadPosePredictor
{
public:
    CHeadPosePredictor();
    virtual ~CHeadPosePredictor();

    // x1, x2, x3, x4, x5, y1, y2, y3, y4, y5
    // (x1, y1) = Left eye center
    // (x2, y2) = Right eye center
    // (x3, y3) = Nose tip
    // (x4, y4) = Left Mouth corner
    // (x5, y5) = Right mouth corner
    SHeadPoseInfo Predict(const int* pFacial5points);

private:
    // Camera-centered coordinate system
    // x & y axis aligned along the horizontal and vertical directions in the image.
    // z axis along the normal to the image plain.
    CvPoint3D32f Predict3DFacialNormal(const cv::Point& noseTip,
        const cv::Point& noseBase,
        const cv::Point& midEye,
        const cv::Point& midMouth);

    float CalDistance(const cv::Point &p1, const cv::Point &p2);
    float CalAngle(const cv::Point &pt1, const cv::Point &pt2);
    float CalSlant(int ln, int lf, const float Rn, float theta);
};
#endif // HeadPosePredictor_h__