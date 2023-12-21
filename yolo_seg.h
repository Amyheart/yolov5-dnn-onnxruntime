#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>


struct OutputSeg {
	int id;             //������id
	float confidence;   //������Ŷ�
	cv::Rect box;       //���ο�
	cv::Mat boxMask;       //���ο���mask����ʡ�ڴ�ռ�ͼӿ��ٶ�
};

class YoloSeg {
public:
	YoloSeg() {
	}
	~YoloSeg() {}

	bool ReadModel(cv::dnn::Net& net, std::string& netPath, bool isCuda);
	bool Detect(cv::Mat& srcImg, const wchar_t* model_path, std::vector<OutputSeg>& output);
	void DrawPred(cv::Mat& img, std::vector<OutputSeg> result, std::vector<cv::Scalar> color);
	void LetterBox(const cv::Mat& image, cv::Mat& outImage,
		cv::Vec4d& params, //[ratio_x,ratio_y,dw,dh]
		const cv::Size& newShape = cv::Size(640, 640),
		bool autoShape = false,
		bool scaleFill = false,
		bool scaleUp = true,
		int stride = 32,
		const cv::Scalar& color = cv::Scalar(114, 114, 114));
private:
	void GetMask(const int* const _seg_params, const cv::Mat& maskProposals, const cv::Mat& mask_protos, const cv::Vec4d& params, const cv::Size& srcImgShape, std::vector<OutputSeg>& output);

	const int _netWidth = 960;   //ONNXͼƬ������
	const int _netHeight = 640;  //ONNXͼƬ����߶�

	float _classThreshold = 0.5;
	float _nmsThreshold = 0.45;
	float _maskThreshold = 0.5;

	//��������Լ���ģ����Ҫ�޸Ĵ���
	std::vector<std::string> _className = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		"hair drier", "toothbrush" };
};

