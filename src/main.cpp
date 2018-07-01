#include <Windows.h>
#include <vector>
#include <chrono>
#include <iostream>

#include "ImageLayer.h"
#include "ConvLayer.h"
#include "ConcatLayer.h"
#include "MAXPoolLayer.h"
#include "AVEPoolLayer.h"
#include "Net.h"

#ifndef NDEBUG
	const int N_FORWARD_LOOPS = 1;
#else 
	const int N_FORWARD_LOOPS = 1;     // release
#endif // DEBUG

int main(int argc, char* argv[]) {

	Net net;
	
	LOG_DUMP("============= Creating layers ===========");

	// image layer
	ImageLayer imlayer("image_layer", 1, 3, 227, 227);
	// conv1 layer
	ConvLayer conv1("conv1", 64, 3, 3, 2, 0);
	MAXPoolLayer pool1("pool1", 3, 3, 2, 0);
	// fire2 module layers
	ConvLayer fire2_sq("fire2/squeeze1x1", 16, 1, 1, 1, 0);
	ConvLayer fire2_ex1("fire2/expand1x1", 64, 1, 1, 1, 0);
	ConvLayer fire2_ex3("fire2/expand3x3", 64, 3, 3, 1, 1);
	ConcatLayer fire2_con("fire2/concat");
	// fire3 module layers
	ConvLayer fire3_sq("fire3/squeeze1x1", 16, 1, 1, 1);
	ConvLayer fire3_ex1("fire3/expand1x1", 64, 1, 1, 1, 0);
	ConvLayer fire3_ex3("fire3/expand3x3", 64, 3, 3, 1, 1);
	ConcatLayer fire3_con("fire3/concat");
	// pool3 layer
	MAXPoolLayer pool3("pool3", 3, 3, 2, 0);
	// fire4 module layers
	ConvLayer fire4_sq("fire4/squeeze1x1", 32, 1, 1, 1);
	ConvLayer fire4_ex1("fire4/expand1x1", 128, 1, 1, 1, 0);
	ConvLayer fire4_ex3("fire4/expand3x3", 128, 3, 3, 1, 1);
	ConcatLayer fire4_con("fire4/concat");
	// fire5 module layers
	ConvLayer fire5_sq("fire5/squeeze1x1", 32, 1, 1, 1);
	ConvLayer fire5_ex1("fire5/expand1x1", 128, 1, 1, 1, 0);
	ConvLayer fire5_ex3("fire5/expand3x3", 128, 3, 3, 1, 1);
	ConcatLayer fire5_con("fire5/concat");
	// pool5 layer
	MAXPoolLayer pool5("pool5", 3, 3, 2, 0);
	// fire6 module layers
	ConvLayer fire6_sq("fire6/squeeze1x1", 48, 1, 1, 1);
	ConvLayer fire6_ex1("fire6/expand1x1", 192, 1, 1, 1, 0);
	ConvLayer fire6_ex3("fire6/expand3x3", 192, 3, 3, 1, 1);
	ConcatLayer fire6_con("fire6/concat");
	// fire7 module layers
	ConvLayer fire7_sq("fire7/squeeze1x1", 48, 1, 1, 1);
	ConvLayer fire7_ex1("fire7/expand1x1", 192, 1, 1, 1, 0);
	ConvLayer fire7_ex3("fire7/expand3x3", 192, 3, 3, 1, 1);
	ConcatLayer fire7_con("fire7/concat");
	// fire8 module layers
	ConvLayer fire8_sq("fire8/squeeze1x1", 64, 1, 1, 1);
	ConvLayer fire8_ex1("fire8/expand1x1", 256, 1, 1, 1, 0);
	ConvLayer fire8_ex3("fire8/expand3x3", 256, 3, 3, 1, 1);
	ConcatLayer fire8_con("fire8/concat");
	// fire9 module layers
	ConvLayer fire9_sq("fire9/squeeze1x1", 64, 1, 1, 1);
	ConvLayer fire9_ex1("fire9/expand1x1", 256, 1, 1, 1, 0);
	ConvLayer fire9_ex3("fire9/expand3x3", 256, 3, 3, 1, 1);
	ConcatLayer fire9_con("fire9/concat");
	// conv10 layer
	ConvLayer conv10("conv10", 1000, 1, 1, 1, 0);
	AVEPoolLayer pool10("pool10", 0, 0, 2, 0);

	LOG_DUMP("============= Connecting layers =============");

	// image layer
	Layer_I::connect(imlayer, conv1);
	// conv1 layer
	Layer_I::connect(conv1, pool1);
	// fire2 module layers
	Layer_I::connect(pool1, fire2_sq);
	Layer_I::connect(fire2_sq, fire2_ex1);
	Layer_I::connect(fire2_sq, fire2_ex3);
	ConcatLayer::connect2(fire2_ex1, fire2_ex3, fire2_con);
	// fire3 module layers
	Layer_I::connect(fire2_con, fire3_sq);
	Layer_I::connect(fire3_sq, fire3_ex1);
	Layer_I::connect(fire3_sq, fire3_ex3);
	ConcatLayer::connect2(fire3_ex1, fire3_ex3, fire3_con);
	// pool3 layer
	Layer_I::connect(fire3_con, pool3);
	// fire4 module layers
	Layer_I::connect(pool3, fire4_sq);
	Layer_I::connect(fire4_sq, fire4_ex1);
	Layer_I::connect(fire4_sq, fire4_ex3);
	ConcatLayer::connect2(fire4_ex1, fire4_ex3, fire4_con);
	// fire5 module layers
	Layer_I::connect(fire4_con, fire5_sq);
	Layer_I::connect(fire5_sq, fire5_ex1);
	Layer_I::connect(fire5_sq, fire5_ex3);
	ConcatLayer::connect2(fire5_ex1, fire5_ex3, fire5_con);
	// pool5 layer
	Layer_I::connect(fire5_con, pool5);
	// fire6 module layers
	Layer_I::connect(pool5, fire6_sq);
	Layer_I::connect(fire6_sq, fire6_ex1);
	Layer_I::connect(fire6_sq, fire6_ex3);
	ConcatLayer::connect2(fire6_ex1, fire6_ex3, fire6_con);
	// fire7 module layers
	Layer_I::connect(fire6_con, fire7_sq);
	Layer_I::connect(fire7_sq, fire7_ex1);
	Layer_I::connect(fire7_sq, fire7_ex3);
	ConcatLayer::connect2(fire7_ex1, fire7_ex3, fire7_con);
	// fire8 module layers
	Layer_I::connect(fire7_con, fire8_sq);
	Layer_I::connect(fire8_sq, fire8_ex1);
	Layer_I::connect(fire8_sq, fire8_ex3);
	ConcatLayer::connect2(fire8_ex1, fire8_ex3, fire8_con);
	// fire9 module layers
	Layer_I::connect(fire8_con, fire9_sq);
	//Layer_I::connect(fire8_ex1, fire9_sq);
	Layer_I::connect(fire9_sq, fire9_ex1);
	Layer_I::connect(fire9_sq, fire9_ex3);
	ConcatLayer::connect2(fire9_ex1, fire9_ex3, fire9_con);
	// conv10 and pool10 - output layers
	Layer_I::connect(fire9_con, conv10);
	Layer_I::connect(conv10, pool10);

	// --------------------------------------------------------------

	net.addLayer(imlayer);
	net.addLayer(conv1);
	net.addLayer(pool1);
	net.addLayer(fire2_sq);
	net.addLayer(fire2_ex1);
	net.addLayer(fire2_ex3);
	net.addLayer(fire2_con);
	net.addLayer(fire3_sq);
	net.addLayer(fire3_ex1);
	net.addLayer(fire3_ex3);
	net.addLayer(fire3_con);
	net.addLayer(pool3);
	net.addLayer(fire4_sq);
	net.addLayer(fire4_ex1);
	net.addLayer(fire4_ex3);
	net.addLayer(fire4_con);
	net.addLayer(fire5_sq);
	net.addLayer(fire5_ex1);
	net.addLayer(fire5_ex3);
	net.addLayer(fire5_con);
	net.addLayer(pool5);
	net.addLayer(fire6_sq);
	net.addLayer(fire6_ex1);
	net.addLayer(fire6_ex3);
	net.addLayer(fire6_con);
	net.addLayer(fire7_sq);
	net.addLayer(fire7_ex1);
	net.addLayer(fire7_ex3);
	net.addLayer(fire7_con);
	net.addLayer(fire8_sq);
	net.addLayer(fire8_ex1);
	net.addLayer(fire8_ex3);
	net.addLayer(fire8_con);
	net.addLayer(fire9_sq);
	net.addLayer(fire9_ex1);
	net.addLayer(fire9_ex3);
	net.addLayer(fire9_con);
	net.addLayer(conv10);
	net.addLayer(pool10);

	LOG_DUMP("============= Initializing layers =============");

	net.init();

	LOG_DUMP("============= Loading parameters =============");


	std::string execFilepath = argv[0];
	size_t lastSepIdx = execFilepath.find_last_of("/\\");
	std::string execDirpath = execFilepath.substr(0, lastSepIdx);
	const std::string paramsH5File(execDirpath + "/squeezenet_v1.1.caffemodel.h5");

	net.copyTrainedLayersFromHDF5(paramsH5File);

	LOG_DUMP("============= Compute first and second moment on params =============");

	net.printParamMoments();

	LOG_DUMP("============= Computing layer sizes =============");

	size_t byteSize = net.calcByteSize();
	LOG_DUMP("Overall byte size: %d (%.2f MB)", byteSize, static_cast<float>(byteSize) / (1024 * 1024));

	LOG_DUMP("============= Activating layers =============");

	auto start_time = std::chrono::high_resolution_clock::now();

	for (size_t i = 0; i < N_FORWARD_LOOPS; i++)
	{
		net.forward();
	}

	auto end_time = std::chrono::high_resolution_clock::now();

	auto time = end_time - start_time;
	float ellapsed = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(time).count());
	std::cout << "Forward mean exec time: " << std::to_string(ellapsed / N_FORWARD_LOOPS / imlayer.image_n) << "ms" << std::endl;
	
	system("pause");
	return 0;
}