#pragma once

#include "Blob.h"
#include "Layer_I.h"

struct ImageLayer : Layer_I
{
	int image_n;			// number of input images
	int image_ch;		    // number of image channels (i.e. colored image has 3 channels)
	int image_w;			// image width
	int image_h;			// image height

	ImageLayer(std::string _name, int _image_n, int _image_ch, int _image_h, int _image_w) :
		Layer_I(_name, LayerType::IMAGE), image_n(_image_n), image_ch(_image_ch), image_h(_image_h), image_w(_image_w) 
	{
		assert(image_n > 0);
		assert(image_ch > 0);
		assert(image_w > 0);
		assert(image_h > 0);

		LOG_DUMP("Created layer '%s' (%d x %d x %d x %d)", name.c_str(), image_n, image_ch, image_h, image_w);
	}

	void init() 
	{
		assert(bottomLayer == nullptr);
		assert(topLayer != nullptr);
		
		// top layer is using padding on input blob (our outputBlob) 
		// => blob is actually bigger than neccessary
		// => we need to adjust indexes by offset to adress only non-padded area on this layer
		if ((topLayer != nullptr) && (topLayer->padding > 0))
		{
			outputBlobPaddingOffset = topLayer->padding;
		}

		BlobShape outputShape = computeOutputShape();
		std::string outputBlobName = name + "_blob";

		outputBlob = new Blob(outputBlobName, outputShape.ndims, outputShape.channels, outputShape.height, outputShape.width);
		ownsOuputBlob = true;
		
		LOG_DUMP("Init IMAGE layer '%s' (%d, %d, %d, %d), output blob '%s' (%d, %d, %d, %d)", 
			name.c_str(), outputBlob->ndims, outputBlob->channels, outputBlob->height, outputBlob->width, 
			outputBlobName.c_str(), outputShape.ndims, outputShape.channels, outputShape.height - (2 * outputBlobPaddingOffset), outputShape.width - (2 * outputBlobPaddingOffset));
	}

	void activate() 
	{
		// TODO image preprocessing (mean subbtraction, etc.)
	}

	// computes outputBlob shape
	BlobShape computeOutputShape()
	{
		BlobShape shape = { 0 };
		
		shape.ndims = image_n;
		shape.channels = image_ch;
		shape.height = image_h + (2 * outputBlobPaddingOffset) - (2 * inputBlobPaddingOffset);
		shape.width = image_w + (2 * outputBlobPaddingOffset) - (2 * inputBlobPaddingOffset);
		
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

	~ImageLayer() 
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
