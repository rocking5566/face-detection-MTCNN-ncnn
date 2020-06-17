#include "HeadPosePredictor.h"
#include <cmath>
#define PI 3.14159265358979f

template<typename T>
T Saturate(const T& val, const T& minVal, const T& maxVal)
{
    return std::min(std::max(val, minVal), maxVal);
}

CHeadPosePredictor::CHeadPosePredictor()
{

}

CHeadPosePredictor::~CHeadPosePredictor()
{

}
#include <iostream>
    using namespace std;
// x1, x2, x3, x4, x5, y1, y2, y3, y4, y5
// (x1, y1) = Left eye center
// (x2, y2) = Right eye center
// (x3, y3) = Nose tip
// (x4, y4) = Left Mouth corner
// (x5, y5) = Right mouth corner
SHeadPoseInfo CHeadPosePredictor::Predict(const int* pFacial5points)
{
    cv::Point leye = cv::Point(pFacial5points[0], pFacial5points[5]);
    cv::Point reye = cv::Point(pFacial5points[1], pFacial5points[6]);
    cv::Point lmouth = cv::Point(pFacial5points[3], pFacial5points[8]);
    cv::Point rmouth = cv::Point(pFacial5points[4], pFacial5points[9]);
    cv::Point noseTip = cv::Point(pFacial5points[2], pFacial5points[7]);

    cv::Point midEye =
        cv::Point((leye.x + reye.x) * 0.5, (leye.y + reye.y) * 0.5);
    cv::Point midMouth = cv::Point((lmouth.x + rmouth.x) * 0.5,
        (lmouth.y + rmouth.y) * 0.5);
    cv::Point noseBase =
        cv::Point((midMouth.x + midEye.x) * 0.5, (midMouth.y + midEye.y) * 0.5);

    CvPoint3D32f normal = Predict3DFacialNormal(noseTip, noseBase, midEye, midMouth);
    SHeadPoseInfo ret;
    ret.facialUnitNormalVector[0] = normal.x;
    ret.facialUnitNormalVector[1] = normal.y;
    ret.facialUnitNormalVector[2] = normal.z;

    ret.yaw = acos((std::abs(normal.z)) / (std::sqrt(normal.x * normal.x + normal.z * normal.z)));
    if (noseTip.x < noseBase.x)
        ret.yaw = -ret.yaw;
    ret.yaw = Saturate(ret.yaw, -1.f, 1.f);

    ret.pitch = acos(std::sqrt((normal.x * normal.x + normal.z * normal.z) / 
        (normal.x * normal.x + normal.y * normal.y + normal.z * normal.z)));
    if (noseTip.y > noseBase.y)
        ret.pitch = -ret.pitch;
    ret.pitch = Saturate(ret.pitch, -1.f, 1.f);

    ret.roll = CalAngle(leye, reye);
    if (ret.roll > 180)
        ret.roll = ret.roll - 360;
    ret.roll /= 90;
    ret.roll = Saturate(ret.roll, -1.f, 1.f);

    return ret;
}

CvPoint3D32f CHeadPosePredictor::Predict3DFacialNormal(const cv::Point& noseTip,
                                                       const cv::Point& noseBase,
                                                       const cv::Point& midEye,
                                                       const cv::Point& midMouth)
{
    float noseBase_noseTip_distance = CalDistance(noseTip, noseBase);
    float midEye_midMouth_distance = CalDistance(midEye, midMouth);

    // Angle facial middle (symmetric) line.
    float symm = CalAngle(noseBase, midEye);

    // Angle between 2D image facial normal & x-axis.
    float tilt = CalAngle(noseBase, noseTip);

    // Angle between 2D image facial normal & facial middle (symmetric) line.
    float theta = (std::abs(tilt - symm)) * (PI / 180.0);

    // Angle between 3D image facial normal & image plain normal (optical axis).
    float slant = CalSlant(noseBase_noseTip_distance,
        midEye_midMouth_distance, 0.5, theta);

    // Define a 3D vector for the facial normal
    CvPoint3D32f ret;
    ret.x = sin(slant) * (cos((360 - tilt) * (PI / 180.0)));
    ret.y = sin(slant) * (sin((360 - tilt) * (PI / 180.0)));
    ret.z = -cos(slant);

    return ret;
}

float CHeadPosePredictor::CalDistance(const cv::Point &p1, const cv::Point &p2)
{
    float x = p1.x - p2.x;
    float y = p1.y - p2.y;
    return sqrtf(x * x + y * y);
}

float CHeadPosePredictor::CalAngle(const cv::Point &pt1, const cv::Point &pt2)
{
    return 360 - cvFastArctan(pt2.y - pt1.y, pt2.x - pt1.x);
}

float CHeadPosePredictor::CalSlant(int ln, int lf, const float Rn, float theta)
{
    float dz = 0;
    float slant = 0;
    const float m1 = ((float)ln * ln) / ((float)lf * lf);
    const float m2 = (cos(theta)) * (cos(theta));
    const float Rn_sq = Rn * Rn;

    if (m2 == 1) {
        dz = sqrt(Rn_sq / (m1 + Rn_sq));
    }
    if (m2 >= 0 && m2 < 1) {
        dz = sqrt((Rn_sq - m1 - 2 * m2 * Rn_sq +
            sqrt(((m1 - Rn_sq) * (m1 - Rn_sq)) + 4 * m1 * m2 * Rn_sq)) /
            (2 * (1 - m2) * Rn_sq));
    }
    slant = acos(dz);
    return slant;
}

