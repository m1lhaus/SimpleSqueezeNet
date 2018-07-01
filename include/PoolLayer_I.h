#pragma once

#include "Blob.h"
#include "Layer_I.h"

struct PoolLayer_I : Layer_I
{
	int kernel_w;		// width of pooling window
	int kernel_h;		// height of pooling window
	int stride;			// stride of pooling window when doing sliding window


	PoolLayer_I(std::string _name, int _kwidth, int _kheight, int _kstride=1, int _kpad=0) :
		Layer_I(_name, LayerType::POOL, _kpad), kernel_w(_kwidth), kernel_h(_kheight), stride(_kstride)
	{
		//assert(kernel_w > 0);		// could be zero => global pooling
		//assert(kernel_h > 0);		// could be zero => global pooling
		assert(stride > 0);

		LOG_DUMP("Created layer '%s' (%d x %d) - stride: %d, pad: %d", name.c_str(),  kernel_h, kernel_w, stride, padding);
	}

	// kernel and bias blob will be created and initialized to zero 
	virtual void init() 
	{
		assert(bottomLayer != nullptr);
		assert(bottomLayer->outputBlob != nullptr);

		Blob * inputBlob = bottomLayer->outputBlob;
		
		// top layer is using padding on input blob (our outputBlob) 
		// => blob is actually bigger than neccessary
		// => we need to adjust indexes by offset to adress only non-padded area on this layer
		if ((topLayer != nullptr) && (topLayer->padding > 0))
		{
			outputBlobPaddingOffset = topLayer->padding;
		}

		// bottom layer has increased outputBlob size, because one if top layer is using padding
		// if this is not the layer which is using padding, we need to adjust blob size accordingly
		if ((bottomLayer->outputBlobPaddingOffset != 0) && (padding == 0))
		{
			inputBlobPaddingOffset = bottomLayer->outputBlobPaddingOffset;
		}

		// global pooling (over whole HxW input map)
		if (kernel_w == 0)
			kernel_w = inputBlob->width;
		if (kernel_h == 0)
			kernel_h = inputBlob->height;

		BlobShape outputShape = computeOutputShape();
		std::string outputBlobName = name + "_blob";

		if ((topLayer == nullptr) || (topLayer->type != LayerType::CONCAT))
		{
			// compute shape of output blob (can be bigger because of topLayer padding)
			outputBlob = new Blob(outputBlobName, outputShape.ndims, outputShape.channels, outputShape.height, outputShape.width);
			ownsOuputBlob = true;
		}

		LOG_DUMP("Initialized POOL layer '%s' (%d, %d, %d, %d), output blob '%s' (%d, %d, %d, %d)",
			name.c_str(), inputBlob->ndims, inputBlob->channels, inputBlob->height, inputBlob->width,
			outputBlobName.c_str(), outputShape.ndims, outputShape.channels, outputShape.height - (2 * outputBlobPaddingOffset), outputShape.width - (2 * outputBlobPaddingOffset));
	}

	// computes shape of outputBlob depending on shape of input blob and kernel parameters
	BlobShape computeOutputShape()
	{
		assert(bottomLayer != nullptr);
		assert(bottomLayer->outputBlob != nullptr);

		BlobShape shape = { 0 };
		Blob * inputBlob = bottomLayer->outputBlob;

		// pooling layer shape in Caffe is computed using ceil() method with contrast to conv layer
		int h_o = static_cast<int>(ceil(static_cast<float>(inputBlob->height - (2 * inputBlobPaddingOffset) - kernel_h) / stride) + 1 + (2 * outputBlobPaddingOffset));
		int w_o = static_cast<int>(ceil(static_cast<float>(inputBlob->width - (2 * inputBlobPaddingOffset) - kernel_w) / stride) + 1 + (2 * outputBlobPaddingOffset));
		assert((h_o > 0) && (w_o > 0));		// just check if given shape params makes sense

		shape.ndims = inputBlob->ndims;
		shape.channels = inputBlob->channels;
		shape.height = h_o;
		shape.width = w_o;

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

	~PoolLayer_I()
	{
		// layers which are concatenated are not holding any output data, but writes into concat layer storage directly
		if (ownsOuputBlob == true)
		{
			assert(outputBlob != nullptr);
			delete outputBlob;
		}
		outputBlob = nullptr;
	}

};
