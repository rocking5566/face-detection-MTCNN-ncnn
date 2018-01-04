#ifndef Mtcnn_h__
#define Mtcnn_h__

#include <algorithm>
#include <vector>
#include "net.h"

struct SMtcnnFace
{
    float score;
    int boundingBox[4];    // x1, y1, x2, y2
    int landmark[10];    // x1, x2, x3, x4, x5, y1, y2, y3, y4, y5
};

struct SFaceProposal
{
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    bool bExist;
    float ppoint[10];    // x1, x2, x3, x4, x5, y1, y2, y3, y4, y5
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
    void Detect(const unsigned char* img, std::vector<SMtcnnFace>& result);

private:
    void ResizeFaceFromScale(ncnn::Mat score, ncnn::Mat location, std::vector<SFaceProposal>& boundingBox_, std::vector<SOrderScore>& bboxScore_, float scale);
    void Nms(std::vector<SFaceProposal> &boundingBox_, std::vector<SOrderScore> &bboxScore_, const float overlap_threshold, std::string modelname = "Union");
    void RefineAndSquareBbox(std::vector<SFaceProposal> &vecBbox, const int &height, const int &width);
    void ConvertToSMtcnnFace(const std::vector<SFaceProposal>& src, std::vector<SMtcnnFace>& dst);

    std::vector<float> GetPyramidScale(unsigned int width, unsigned int height, int iMinSize, float fPyramidFactor);
    std::vector<SFaceProposal> PNetWithPyramid(const ncnn::Mat& img, const std::vector<float> pyramidScale);
    std::vector<SFaceProposal> RNet(const ncnn::Mat& img, const std::vector<SFaceProposal> PNetResult);
    std::vector<SFaceProposal> ONet(const ncnn::Mat& img, const std::vector<SFaceProposal> RNetResult);


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
