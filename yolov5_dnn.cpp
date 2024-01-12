#include"yolov5_dnn.h"
using namespace std;
using namespace cv;
using namespace cv::dnn;

bool Yolov5_Dnn::ReadModel(Net& net, string& netPath, bool isCuda = false) {
	try {
		net = readNetFromONNX(netPath);
	}
	catch (const std::exception&) {
		return false;
	}
	//cuda
	if (isCuda) {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
		//net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		//net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	}
	//cpu
	else {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	return true;
}

void Yolov5_Dnn::LetterBox(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape,
	bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar& color)
{
	if (false) {
		int maxLen = MAX(image.rows, image.cols);
		outImage = Mat::zeros(Size(maxLen, maxLen), CV_8UC3);
		image.copyTo(outImage(Rect(0, 0, image.cols, image.rows)));
		params[0] = 1;
		params[1] = 1;
		params[3] = 0;
		params[2] = 0;
	}

	cv::Size shape = image.size();
	float r = std::min((float)newShape.height / (float)shape.height,
		(float)newShape.width / (float)shape.width);
	if (!scaleUp)
		r = std::min(r, 1.0f);

	float ratio[2]{ r, r };
	int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

	auto dw = (float)(newShape.width - new_un_pad[0]);
	auto dh = (float)(newShape.height - new_un_pad[1]);

	if (autoShape)
	{
		dw = (float)((int)dw % stride);
		dh = (float)((int)dh % stride);
	}
	else if (scaleFill)
	{
		dw = 0.0f;
		dh = 0.0f;
		new_un_pad[0] = newShape.width;
		new_un_pad[1] = newShape.height;
		ratio[0] = (float)newShape.width / (float)shape.width;
		ratio[1] = (float)newShape.height / (float)shape.height;
	}

	dw /= 2.0f;
	dh /= 2.0f;

	if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
	{
		cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
	}
	else {
		outImage = image.clone();
	}

	int top = int(std::round(dh - 0.1f));
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(dw - 0.1f));
	int right = int(std::round(dw + 0.1f));
	params[0] = ratio[0];
	params[1] = ratio[1];
	params[2] = left;
	params[3] = top;
	cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}


bool Yolov5_Dnn::Detect(Mat& SrcImg, Net& net, vector<OutputSeg>& output) {
	Mat blob;
	output.clear();
	
	Mat netInputImg;
	Vec4d params;
	LetterBox(SrcImg, netInputImg, params, cv::Size(_netWidth, _netHeight));
	blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(0, 0, 0), true, false);
	//如果在其他设置没有问题的情况下但是结果偏差很大，可以尝试下用下面两句语句
	//blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(104, 117, 123), true, false);
	//blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(114, 114,114), true, false);
	net.setInput(blob);
	std::vector<cv::Mat> netOutputImg;
	net.forward(netOutputImg, net.getUnconnectedOutLayersNames()); //获取output的输出
	if (netOutputImg.size() == 2) RunSegmentation = true;
	this->_clsNum = netOutputImg[0].size[2] - 5;
	if (RunSegmentation) _clsNum=_clsNum - 32;

	std::vector<int> classIds;//结果id数组
	std::vector<float> confidences;//结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;//每个id矩形框
	std::vector<vector<float>> picked_proposals;  //存储output0[:,:, 5 + _clsNum:net_width]用以后续计算mask
	int net_height = netOutputImg[0].size[1];
	int net_width = netOutputImg[0].size[2];  
	
	float* pdata = (float*)netOutputImg[0].data;
	for (int r = 0; r < net_height; ++r) {
	float box_score = pdata[4]; ;//获取每一行的box框中含有某个物体的概率
	if (box_score >= _classThreshold) {
		cv::Mat scores(1, _clsNum, CV_32FC1, pdata + 5);
		Point classIdPoint;
		double max_class_socre;
		minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
		max_class_socre = max_class_socre * box_score;
		if (max_class_socre >= _classThreshold) {
			vector<float> temp_proto(pdata + 5 + _clsNum, pdata + net_width);
			picked_proposals.push_back(temp_proto);
			//rect [x,y,w,h]
			float x = (pdata[0] - params[2]) / params[0];
			float y = (pdata[1] - params[3]) / params[1];
			float w = pdata[2] / params[0];
			float h = pdata[3] / params[1];
			int left = MAX(round(x - 0.5 * w + 0.5), 0);
			int top = MAX(round(y - 0.5 * h + 0.5), 0);
			classIds.push_back(classIdPoint.x);
			confidences.push_back(max_class_socre);
			boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
			}
		}
	pdata += net_width;//下一行
	}
	//执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
	vector<int> nms_result;
	NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result);
	std::vector<vector<float>> temp_mask_proposals;
	Rect holeImgRect(0, 0, SrcImg.cols, SrcImg.rows);
	for (int i = 0; i < nms_result.size(); ++i) {
		int idx = nms_result[i];
		OutputSeg result;
		result.id = classIds[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx] & holeImgRect;
		temp_mask_proposals.push_back(picked_proposals[idx]);
		if (result.box.width != 0 && result.box.height != 0) output.push_back(result);
	}
	if (RunSegmentation) {
		Mat mask_proposals;
		for (int i = 0; i < temp_mask_proposals.size(); ++i)
			mask_proposals.push_back(Mat(temp_mask_proposals[i]).t());
		GetMask(mask_proposals, netOutputImg[1], params, SrcImg.size(), output);
	}
	if (output.size())
		return true;
	else
		return false;
}

void Yolov5_Dnn::GetMask(const Mat& maskProposals, const Mat& mask_protos, const cv::Vec4d& params, const cv::Size& srcImgShape, vector<OutputSeg>& output) {
	int _segChannels = mask_protos.size[1];
	int _segHeight = mask_protos.size[2];
	int _segWidth = mask_protos.size[3];
	Mat protos = mask_protos.reshape(0, { _segChannels,_segWidth * _segHeight });
	Mat matmulRes = (maskProposals * protos).t();
	Mat masks = matmulRes.reshape(output.size(), { _segHeight, _segWidth });
	vector<Mat> maskChannels;
	split(masks, maskChannels);
	for (int i = 0; i < output.size(); ++i) {
		Mat dest, mask;
		//sigmoid
		cv::exp(-maskChannels[i], dest);
		dest = 1.0 / (1.0 + dest);

		Rect roi(int(params[2] / _netWidth * _segWidth), int(params[3] / _netHeight * _segHeight), int(_segWidth - params[2] / 2), int(_segHeight - params[3] / 2));
		dest = dest(roi);
		resize(dest, mask, srcImgShape, INTER_NEAREST);

		//crop
		Rect temp_rect = output[i].box;
		mask = mask(temp_rect) > _maskThreshold;
		output[i].boxMask = mask;
	}
}

void Yolov5_Dnn::DrawPred(Mat& img, vector<OutputSeg> result, vector<Scalar> color) {
	Mat mask = img.clone();
	for (int i = 0; i < result.size(); i++) {
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;
		int color_num = i;
		rectangle(img, result[i].box, color[result[i].id], 2, 8);
		mask(result[i].box).setTo(color[result[i].id], result[i].boxMask);
		string label = _className[result[i].id] + ":" + to_string(result[i].confidence).substr(0, 4);
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height)-10;
		putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
	}
	if (RunSegmentation) {
		addWeighted(img, 0.5, mask, 0.5, 0, img); //将mask加在原图上面
	}
	imshow("1", img);
	imwrite("out.bmp", img);
	waitKey();
	destroyWindow("1");
}
