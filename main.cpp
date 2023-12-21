#include <iostream>
#include<opencv2//opencv.hpp>
#include<math.h>
#include "yolo_seg.h"
#include<time.h>
//#include"yolov5.h"

using namespace std;
using namespace cv;
using namespace dnn;

int yolov5_seg()
{
	string img_path = "./images/zidane.jpg";
	const wchar_t* model_path = L"yolov5s-seg_d.onnx";
	YoloSeg test;
	//生成随机颜色
	vector<Scalar> color;
	srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(Scalar(b, g, r));
	}
	vector<OutputSeg> result;
	Mat img = imread(img_path);
	clock_t t1, t2;
	if (test.Detect(img, model_path, result)) {
		test.DrawPred(img, result, color);
	}
	else {
		cout << "Detect Failed!" << endl;
	}
	system("pause");
	return 0;
}


int main() {
	//system("chcp 65001");  //终端显示中文
	yolov5_seg();
	return 0;
}


