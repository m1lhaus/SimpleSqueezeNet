#pragma once

#include "Layer_I.h"


struct ConcatLayer : Layer_I
{
	Layer_I * bottomLayer2;

	ConcatLayer(std::string _name) : Layer_I(_name, LayerType::CONCAT), bottomLayer2(nullptr)
	{
		LOG_DUMP("Created layer '%s'", name.c_str());
	}

	 //initialize layer blobs for outputBlob data and params
	void init() {
		assert(bottomLayer != nullptr);
		assert(bottomLayer2 != nullptr);

		// bottomLayer blobs will be redirected to this layer blob (result of concat will be continious array)
		assert(bottomLayer->outputBlob == nullptr);
		assert(bottomLayer2->outputBlob == nullptr);

		if ((topLayer != nullptr) && (topLayer->padding > 0))
		{
			outputBlobPaddingOffset = topLayer->padding;
		}

		// compute shape of output blob (can be bigger because of topLayer padding)
		BlobShape outputShape = computeOutputShape();
		outputBlob = new Blob(name + "_blob", outputShape.ndims, outputShape.channels, outputShape.height, outputShape.width);
		ownsOuputBlob = true;

		// point concatenated sublayers blobs to this layer, since those two blobs are stored as continuous array here
		bottomLayer->outputBlob = outputBlob;
		bottomLayer2->outputBlob = outputBlob;
		
		LOG_DUMP("Initialized CONCAT layer '%s' for layers (%s, %s), output blob '%s' (%d, %d, %d, %d)", 
			name.c_str(), bottomLayer->name.c_str(), bottomLayer2->name.c_str(), 
			outputBlob->name.c_str(), outputBlob->ndims, outputBlob->channels, outputBlob->height, outputBlob->width);
	}

	void activate() 
	{
		// pass
	}

	// computes shape of outputBlob depending on shape of input blob and kernel parameters
	BlobShape computeOutputShape()
	{
		assert(bottomLayer != nullptr);
		assert(bottomLayer2 != nullptr);

		BlobShape shape = { 0 };

		BlobShape layer1_shape = bottomLayer->computeOutputShape();
		BlobShape layer2_shape = bottomLayer2->computeOutputShape();

		assert(layer1_shape.ndims == layer2_shape.ndims);
		assert(layer1_shape.channels == layer2_shape.channels);
		assert(layer1_shape.height == layer2_shape.height);
		assert(layer1_shape.width == layer2_shape.width);

		shape.ndims = layer1_shape.ndims;
		shape.channels = layer1_shape.channels + layer2_shape.channels;
		shape.height = layer1_shape.height + (2 * outputBlobPaddingOffset);
		shape.width = layer1_shape.width + (2 * outputBlobPaddingOffset);

		return shape;
	}

	// computes byse size of data stored on heap (blobs) for layer
	size_t byteSize()
	{
		size_t outputByteSize = 0;
		if ((outputBlob != nullptr) && (ownsOuputBlob == true))
		{
			outputByteSize += outputBlob->ndims * outputBlob->channels * outputBlob->height * outputBlob->width * sizeof(float);
		}
		
		LOG_DUMP("Layer '%s' byte size: %d (%.4f MB)", name.c_str(), outputByteSize, static_cast<float>(outputByteSize) / (1024 * 1024));

		return outputByteSize;
	}

	void loadHDF5Params(const std::string & filepath)
	{
		// pass
	}

	void printParamMoments()
	{
		// pass
	}

	~ConcatLayer() 
	{
		// layers which are concatenated are not holding any output data, but writes into concat layer storage directly
		if (ownsOuputBlob == true)
		{
			assert(outputBlob != nullptr);
			delete outputBlob;
		}
		outputBlob = nullptr;
	}

	static void connect2(Layer_I & _bottomLayer1, Layer_I & _bottomLayer2, ConcatLayer & _topLayer)
	{
		_bottomLayer1.topLayer = &_topLayer;
		_bottomLayer2.topLayer = &_topLayer;

		_topLayer.bottomLayer = &_bottomLayer1;
		_topLayer.bottomLayer2 = &_bottomLayer2;

		LOG_DUMP("Connected layers: ['%s', '%s'] --> '%s'", _bottomLayer1.name.c_str(), _bottomLayer2.name.c_str(), _topLayer.name.c_str());
	}
};
