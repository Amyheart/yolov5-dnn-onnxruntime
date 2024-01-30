#pragma once
#ifdef _WIN32
#include <Windows.h>
#include <direct.h>
#include <io.h>
#endif
#include <onnxruntime_cxx_api.h>
#include <cuda_fp16.h>
#include <fstream>
#include "utils.h"


class Yolov5_Ort {
public:
	Yolov5_Ort() {
	}
	~Yolov5_Ort() { delete session; }

	void LoadModel(std::string& model_path);
	void LetterBox(const cv::Mat& image, cv::Mat& outImage,
		cv::Vec4d& params, //[ratio_x,ratio_y,dw,dh]
		const cv::Size& newShape = cv::Size(640, 640),
		bool autoShape = false,
		bool scaleFill = false,
		bool scaleUp = true,
		int stride = 32,
		const cv::Scalar& color = cv::Scalar(114, 114, 114));
	template<typename N> void BlobFromImage(cv::Mat& iImg, N& iBlob);
	bool Detect(cv::Mat& srcImg, std::vector<OutputSeg>& output);
	template<typename N> void RunSession(cv::Mat& SrcImg, cv::Mat& netInputImg, cv::Vec4d& params, N* blob,  std::vector<OutputSeg>& output);
	void GetMask(const int* const _seg_params, const cv::Mat& maskProposals, const cv::Mat& mask_protos, const cv::Vec4d& params, const cv::Size& srcImgShape, std::vector<OutputSeg>& output);
	void DrawPred(cv::Mat& img, std::vector<OutputSeg> result, std::vector<cv::Scalar> color);

private:
	Ort::Session* session;
	Ort::Env env;
	std::vector<const char*> input_node_names;
	std::vector<const char*> output_node_names;
	bool RunSegmentation = false;
	bool RunFP16 = false;

	int _netWidth = 640;   //ONNX图片输入宽度
	int _netHeight = 640;  //ONNX图片输入高度
	int _clsNum = 80;

	float _classThreshold = 0.5;
	float _nmsThreshold = 0.45;
	float _maskThreshold = 0.5;

	//类别名，自己的模型需要修改此项
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

