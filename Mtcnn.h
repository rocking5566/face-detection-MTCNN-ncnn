#ifndef Mtcnn_h__
#define Mtcnn_h__

#include <algorithm>
#include <vector>

#include "net.h"

struct SBoundingBox
{
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    bool bExist;
    float ppoint[10];
    float regreCoord[4];
};

struct SOrderScore
{
    float score;
    int oriOrder;
};

enum imageType
{
    eBGR
};

class CMtcnn
{
public:
    CMtcnn();
    void LoadModel(const char* pNetStructPath, const char* pNetWeightPath
                 , const char* rNetStructPath, const char* rNetWeightPath
                 , const char* oNetStructPath, const char* oNetWeightPath);

    // Can be called in any time
    void SetParam(unsigned int width, unsigned int height, imageType type = eBGR, int iMinSize = 90, float fPyramidFactor = 0.709);
    void Detect(const unsigned char* img, std::vector<SBoundingBox>& result);

private:
    void GenerateBbox(ncnn::Mat score, ncnn::Mat location, std::vector<SBoundingBox>& boundingBox_, std::vector<SOrderScore>& bboxScore_, float scale);
    void Nms(std::vector<SBoundingBox> &boundingBox_, std::vector<SOrderScore> &bboxScore_, const float overlap_threshold, std::string modelname = "Union");
    void RefineAndSquareBbox(std::vector<SBoundingBox> &vecBbox, const int &height, const int &width);

    std::vector<float> GetPyramidScale(unsigned int width, unsigned int height, int iMinSize, float fPyramidFactor);
    std::vector<SBoundingBox> PNetWithPyramid(const ncnn::Mat& img, const std::vector<float> pyramidScale);
    std::vector<SBoundingBox> RNet(const ncnn::Mat& img, const std::vector<SBoundingBox> PNetResult);
    std::vector<SBoundingBox> ONet(const ncnn::Mat& img, const std::vector<SBoundingBox> RNetResult);


private:
    ncnn::Net m_Pnet;
    ncnn::Net m_Rnet;
    ncnn::Net m_Onet;

    int m_ImgWidth;
    int m_ImgHeight;
    imageType m_ImgType;
    std::vector<float> m_pyramidScale;
};
#endif // Mtcnn_h__
