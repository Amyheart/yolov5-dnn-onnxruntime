#include"yolo_seg.h"
#include <onnxruntime_cxx_api.h>
#define segment  //define segment when performing inference of segmentation  
using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace Ort;


bool YoloSeg::ReadModel(Net& net, string& netPath, bool isCuda = false) {
	try {
		net = readNet(netPath);
	}
	catch (const std::exception&) {
		return false;
	}
	//cuda
	if (isCuda) {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
	}
	//cpu
	else {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	return true;
}
void YoloSeg::LetterBox(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape,
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
	int newUnpad[2]{ (int)std::round((float)shape.width * r),
		(int)std::round((float)shape.height * r) };

	auto dw = (float)(newShape.width - newUnpad[0]);
	auto dh = (float)(newShape.height - newUnpad[1]);

	if (autoShape)
	{
		dw = (float)((int)dw % stride);
		dh = (float)((int)dh % stride);
	}
	else if (scaleFill)
	{
		dw = 0.0f;
		dh = 0.0f;
		newUnpad[0] = newShape.width;
		newUnpad[1] = newShape.height;
		ratio[0] = (float)newShape.width / (float)shape.width;
		ratio[1] = (float)newShape.height / (float)shape.height;
	}

	dw /= 2.0f;
	dh /= 2.0f;

	if (shape.width != newUnpad[0] && shape.height != newUnpad[1])
	{
		cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
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


bool YoloSeg::Detect(Mat& SrcImg, const wchar_t* model_path, vector<OutputSeg>& output) {
	Mat blob;
	output.clear();
	int col = SrcImg.cols;
	int row = SrcImg.rows;
	int maxLen = MAX(col, row);
	Mat netInputImg;
	Vec4d params;
	LetterBox(SrcImg, netInputImg, params, cv::Size(_netWidth, _netHeight));
	//blobFromImage��ͼ�����Ԥ������������ֵ���������ţ��ü�������ͨ����
	blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(0, 0, 0), true, false);
	//blobFromImage������dnn��ͼƬ��Ԥ�����������������û�����������µ��ǽ��ƫ��ܴ󣬿��Գ�����������������䣻Ҳ����������дԤ��������
	//blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(104, 117, 123), true, false);
	//blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(114, 114,114), true, false);
	//environment ������ΪVERBOSE��ORT_LOGGING_LEVEL_VERBOSE��ʱ���������̨���ʱ������ʹ����cpu����gpuִ�У�
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "OnnxModel");
	Ort::SessionOptions session_options;
	// ʹ��1���߳�ִ��op,���������ٶȣ������߳���
	session_options.SetIntraOpNumThreads(1);
	//CUDA���ٿ���(����onnxruntime�İ汾̫�ߣ���cuda_provider_factory.h��ͷ�ļ������ٿ���ʹ��onnxruntime V1.8�İ汾)
	//OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
	// ORT_ENABLE_ALL: �������п��ܵ��Ż�
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	Ort::Session session(env, model_path, session_options);

	// print model input layer (node names, types, shape etc.)
	Ort::AllocatorWithDefaultOptions allocator;
	//model info
	// ���ģ���ֶ��ٸ�����������һ����ָ��Ӧ��������Ŀ
	// һ������ֻ��ͼ��Ļ�input_nodesΪ1
	size_t num_input_nodes = session.GetInputCount();
	// ����Ƕ�������磬�ͻ��Ƕ�Ӧ�������Ŀ
	size_t num_output_nodes = session.GetOutputCount();
	printf("Number of inputs = %zu\n", num_input_nodes);
	printf("Number of output = %zu\n", num_output_nodes);
	//��ȡ����names
	std::vector<const char*> input_node_names;
	for (int i = 0; i < num_input_nodes; i++) {
		AllocatedStringPtr input_name = session.GetInputNameAllocated(i, allocator);
		input_node_names.push_back(input_name.get());
		std::cout << "input_name:" << input_name.get() << std::endl;
	}
	//��ȡ���names
	std::vector<const char*> output_node_names;
	for (int i = 0; i < num_output_nodes; i++) {
		AllocatedStringPtr output_name = session.GetOutputNameAllocated(i, allocator);
		output_node_names.push_back(output_name.get());
		std::cout << "output_name: " << output_name.get() << std::endl;
	}
	// �Զ���ȡά������
	auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	auto output_dims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	std::cout << "input_dims:" << input_dims.size() << std::endl;
	std::cout << "output_dims:" << output_dims.size() << std::endl;

	//�Զ���ȡģ��names
	auto meta_names = session.GetModelMetadata().LookupCustomMetadataMapAllocated("names", allocator);
	cout <<"names:"<< meta_names.get() << endl;

	// ��onnxģ���Ƕ�̬����ʱ�����������ά��Ĭ��ֵ��-1����Ҫ����������ֵ
	input_dims = {1, 3, _netHeight,_netWidth };
	
	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor = Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), blob.total(), input_dims.data(), input_dims.size());
	//����(score model & input tensor, get back output tensor)
	vector<Value> ort_outputs = session.Run(RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), output_node_names.size());

	std::vector<int> classIds;//���id����
	std::vector<float> confidences;//���ÿ��id��Ӧ���Ŷ�����
	std::vector<cv::Rect> boxes;//ÿ��id���ο�
	std::vector<vector<float>> picked_proposals;  //�洢output0[:,:, 5 + _className.size():net_width]���Ժ�������mask

	float ratio_h = (float)netInputImg.rows / _netHeight;
	float ratio_w = (float)netInputImg.cols / _netWidth;
	float* pdata = ort_outputs[0].GetTensorMutableData<float>();

	std::vector<int64_t> _outputTensorShape;
	_outputTensorShape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	int net_width = _outputTensorShape[2];
	int net_height = _outputTensorShape[1];
	for (int r = 0; r < net_height; ++r) {
		float box_score = pdata[4]; ;//��ȡÿһ�е�box���к���ĳ������ĸ���
		if (box_score >= _classThreshold) {
			cv::Mat scores(1, _className.size(), CV_32FC1, pdata + 5);
			Point classIdPoint;
			double max_class_socre;
			minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
			max_class_socre = max_class_socre * box_score;
			if (max_class_socre >= _classThreshold) {
				vector<float> temp_proto(pdata + 5 + _className.size(), pdata + net_width);
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
		pdata += net_width;//��һ��
	}

	//ִ�з�����������������нϵ����Ŷȵ������ص���NMS��
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
		output.push_back(result);
	}
	// ����mask
#ifdef segment
	Mat mask_proposals;
	for (int i = 0; i < temp_mask_proposals.size(); ++i)
		mask_proposals.push_back(Mat(temp_mask_proposals[i]).t());
	
	std::vector<int64_t> _outputMaskTensorShape;
	_outputMaskTensorShape = ort_outputs[1].GetTensorTypeAndShapeInfo().GetShape();
	int _segChannels = _outputMaskTensorShape[1];
	int _segWidth = _outputMaskTensorShape[2];
	int _segHeight = _outputMaskTensorShape[3];
	float* pdata1 = ort_outputs[1].GetTensorMutableData<float>();
	std::vector<float> mask(pdata1, pdata1 + _segChannels * _segWidth * _segHeight);

	int _seg_params[3] = {_segChannels, _segWidth, _segHeight};
	Mat mask_protos = Mat(mask);
	GetMask(_seg_params, mask_proposals, mask_protos, params, SrcImg.size(), output);
#endif
	if (output.size())
		return true;
	else
		return false;
}

void YoloSeg::GetMask(const int* const _seg_params, const Mat& maskProposals, const Mat& mask_protos, const cv::Vec4d& params, const cv::Size& srcImgShape, vector<OutputSeg>& output) {
	int _segChannels = * _seg_params;
	int _segHeight = *(_seg_params+1);
	int _segWidth = *(_seg_params + 2);
	Mat protos = mask_protos.reshape(0, { _segChannels,_segWidth * _segHeight });
	Mat matmulRes = (maskProposals * protos).t();
	Mat masks = matmulRes.reshape(output.size(), { _segHeight,_segWidth });
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

void YoloSeg::DrawPred(Mat& img, vector<OutputSeg> result, vector<Scalar> color) {
	Mat mask = img.clone();
	for (int i = 0; i < result.size(); i++) {
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;
		int color_num = i;
		rectangle(img, result[i].box, color[result[i].id], 2, 8);
		mask(result[i].box).setTo(color[result[i].id], result[i].boxMask);
		string label = _className[result[i].id] + ":" + to_string(result[i].confidence).substr(0,4);
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height)-10;
		putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.8, color[result[i].id], 2);
	}
#ifdef segment
	addWeighted(img, 0.5, mask, 0.5, 0, img); //��mask����ԭͼ����
#endif
	imshow("1", img);
	imwrite("out.bmp", img);
	waitKey();
	destroyWindow("1");

}
