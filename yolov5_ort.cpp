#define _CRT_SECURE_NO_WARNINGS
#include"yolov5_ort.h"


using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace Ort;

void Yolov5_Ort::LoadModel(string& model_path) {
#ifdef _WIN32
	int ModelPathSize = MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), static_cast<int>(model_path.length()), nullptr, 0);
	wchar_t* wide_cstr = new wchar_t[ModelPathSize + 1];
	MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), static_cast<int>(model_path.length()), wide_cstr, ModelPathSize);
	wide_cstr[ModelPathSize] = L'\0';
	const wchar_t* modelPath = wide_cstr;
#else
	const char* modelPath = iParams.ModelPath.c_str();
#endif // _WIN32

	//environment （设置为VERBOSE（ORT_LOGGING_LEVEL_VERBOSE）时，方便控制台输出时看到是使用了cpu还是gpu执行）
	env=Ort::Env(ORT_LOGGING_LEVEL_WARNING, "OnnxModel");
	Ort::SessionOptions session_options;
	// 使用1个线程执行op,若想提升速度，增加线程数
	session_options.SetIntraOpNumThreads(1);
	//CUDA加速开启(由于onnxruntime的版本太高，无cuda_provider_factory.h的头文件，加速可以使用onnxruntime V1.8的版本)
	//OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
	//ORT_ENABLE_ALL: 启用所有可能的优化
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	session_options.SetLogSeverityLevel(3);
	this->session =new Ort::Session(env, modelPath, session_options);

	// print model input layer (node names, types, shape etc.)
	Ort::AllocatorWithDefaultOptions allocator;
	//model info
	// 获得模型又多少个输入和输出，一般是指对应网络层的数目
	// 一般输入只有图像的话input_nodes为1
	size_t num_input_nodes = session->GetInputCount();
	// 如果是多输出网络，就会是对应输出的数目
	size_t num_output_nodes = session->GetOutputCount();
	printf("Number of inputs = %zu\n", num_input_nodes);
	printf("Number of output = %zu\n", num_output_nodes);
	if (num_output_nodes == 2) RunSegmentation = true;
	//获取输入节点names
	for (int i = 0; i < num_input_nodes; i++) {
		AllocatedStringPtr input_name = session->GetInputNameAllocated(i, allocator);
		char* temp_buf = new char[20];
		strcpy(temp_buf, input_name.get());
		input_node_names.push_back(temp_buf);
		std::cout << "input_name:" << temp_buf << std::endl;
	}
	//获取输出节点names
	for (int i = 0; i < num_output_nodes; i++) {
		AllocatedStringPtr output_name = session->GetOutputNameAllocated(i, allocator);
		char* temp_buf = new char[20];
		strcpy(temp_buf, output_name.get());
		output_node_names.push_back(temp_buf);
		std::cout << "output_name: " << temp_buf << std::endl;
	}
	// 自动获取维度数量
	auto input_dims = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	auto output_dims = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	this->_netHeight = input_dims[2];
	this->_netWidth = input_dims[3];
	this->_clsNum = output_dims[2] - 5;
	if (RunSegmentation) _clsNum = _clsNum - 32;
	std::cout << "input_dims:";
	for (int i = 0; i < input_dims.size(); i++) std::cout << input_dims[i] << " ";
	std::cout << std::endl;
	std::cout << "output_dims:";
	for (int i = 0; i < output_dims.size(); i++) std::cout << output_dims[i] << " ";
	std::cout << std::endl;

	//自动获取模型names
	auto meta_names = session->GetModelMetadata().LookupCustomMetadataMapAllocated("names", allocator);
	string names_str = meta_names.get();
	cout << "names:" << names_str << endl;
	std::ofstream fout("model_labels.txt");
	fout << names_str;
	fout.close();
}

void Yolov5_Ort::LetterBox(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape,
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


bool Yolov5_Ort::Detect(Mat& SrcImg, vector<OutputSeg>& output) {
	output.clear();
	Mat blob;
	Mat netInputImg;
	Vec4d params;
	LetterBox(SrcImg, netInputImg, params, cv::Size(_netWidth, _netHeight));
	//blobFromImage对图像进行预处理，包括减均值，比例缩放，裁剪，交换通道等
	blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(0, 0, 0), true, false);
	//blobFromImage是利用dnn对图片做预处理，如果在其他设置没有问题的情况下但是结果偏差很大，可以尝试下用下面两句语句；也可以另外重写预处理函数；
	//blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(104, 117, 123), true, false);
	//blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(114, 114,114), true, false);
	
	// 当onnx模型是动态输入时，input_dims输入变量的维度默认值是-1，需要给出具体数值
	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	std::vector<int64_t> input_dims={1, 3, _netHeight, _netWidth };
	Value input_tensor = Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), blob.total(), input_dims.data(), input_dims.size());
	//推理(score model & input tensor, get back output tensor)
	vector<Value> ort_outputs = session->Run(RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), output_node_names.size());

	std::vector<int> classIds;//结果id数组
	std::vector<float> confidences;//结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;//每个id矩形框
	std::vector<vector<float>> picked_proposals;  //存储output0[:,:, 5 + _clsNum:net_width]用以后续计算mask

	float ratio_h = (float)netInputImg.rows / _netHeight;
	float ratio_w = (float)netInputImg.cols / _netWidth;
	float* pdata = ort_outputs[0].GetTensorMutableData<float>();

	std::vector<int64_t> _outputTensorShape;
	_outputTensorShape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	int net_width = _outputTensorShape[2];
	int net_height = _outputTensorShape[1];
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
	// 计算mask
	if (RunSegmentation) {
		Mat mask_proposals;
		for (int i = 0; i < temp_mask_proposals.size(); ++i)
			mask_proposals.push_back(Mat(temp_mask_proposals[i]).t());

		std::vector<int64_t> _outputMaskTensorShape;
		_outputMaskTensorShape = ort_outputs[1].GetTensorTypeAndShapeInfo().GetShape();
		int _segChannels = _outputMaskTensorShape[1];
		int _segHeight = _outputMaskTensorShape[2];
		int _segWidth = _outputMaskTensorShape[3];
		float* pdata1 = ort_outputs[1].GetTensorMutableData<float>();
		std::vector<float> mask(pdata1, pdata1 + _segChannels * _segWidth * _segHeight);

		int _seg_params[3] = { _segChannels,  _segHeight,_segWidth };
		Mat mask_protos = Mat(mask);
		GetMask(_seg_params, mask_proposals, mask_protos, params, SrcImg.size(), output);
	}
	
	if (output.size())
		return true;
	else
		return false;
}

void Yolov5_Ort::GetMask(const int* const _seg_params, const Mat& maskProposals, const Mat& mask_protos, const cv::Vec4d& params, const cv::Size& srcImgShape, vector<OutputSeg>& output) {
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

void Yolov5_Ort::DrawPred(Mat& img, vector<OutputSeg> result, vector<Scalar> color) {
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
	if (RunSegmentation) {
		addWeighted(img, 0.5, mask, 0.5, 0, img); //将mask加在原图上面
	}
	imshow("1", img);
	imwrite("out.bmp", img);
	waitKey();
	destroyWindow("1");
	
}
