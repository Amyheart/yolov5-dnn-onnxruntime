#include "yolov5_dnn.h"
#include "yolov5_ort.h"

using namespace std;
using namespace cv;
using namespace dnn;

void main()
{
	string img_path = "../img_test/zidane.jpg";
	string model_path = "../weight_v5/yolov5s-seg_960.onnx";
	string test_cls = "dnn";
	//生成随机颜色
	vector<Scalar> color;
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(Scalar(b, g, r));
	}

	vector<OutputSeg> result;
	Mat img = imread(img_path);
	if (test_cls == "ort") {
		Yolov5_Ort test;
		test.LoadModel(model_path);
		if (test.Detect(img, result)) {
			test.DrawPred(img, result, color);
		}
		else {
			cout << "Detect Nothing!" << endl;
		}
	}
	
	if (test_cls == "dnn") {
		Yolov5_Dnn test;
		Net net;
		if (test.ReadModel(net, model_path, true)) {
			cout << "read net ok!" << endl;
		}
		else {
			cout << "read net failed!" << endl;
		}

		if (test.Detect(img, net, result)) {
			test.DrawPred(img, result, color);
		}
		else {
			cout << "Detect Nothing!" << endl;
		}
	}
	

	system("pause");
}