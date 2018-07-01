#pragma once

#include <string>

#include "Blob_I.h"

#include "hdf5.h"
#include "hdf5_hl.h"



enum LayerType
{
	CONV,
	POOL,
	CONCAT,
	IMAGE
};


struct Layer_I
{
	Blob * outputBlob;				// output data blob (actual data storage)
	bool ownsOuputBlob;

	Layer_I * bottomLayer;			// reference to layer connected to bottom
	Layer_I * topLayer;				// reference to layer connected to top 

	std::string name;				// name of the layer (identifier)
	LayerType type;                 // type of the layer

	int padding;					// padding on every side of input blob in pixels (i.e. padding of 2 pixels on 10x10 2D array results in 14x14 array)
	int channelsOffset;				// when layer writes into "concat" blob (top layer), this offset is set to point into correct "channels" memory segment
	int outputBlobPaddingOffset;	// when top layer has non-zero padding, output blob size is increased and this offset is used to point this layer to correct (non-padded) data 
	int inputBlobPaddingOffset;		// when bottom layer has non-zero outputBlobPaddingOffset, but this layer has no padding (multiple outputs from bottom layer), adjust outputBlob size properly

	Layer_I(std::string _name, LayerType _type, int _padding=0) : name(_name), type(_type), padding(_padding), channelsOffset(0), outputBlobPaddingOffset(0), inputBlobPaddingOffset(0), outputBlob(nullptr), ownsOuputBlob(false), bottomLayer(nullptr), topLayer(nullptr)
	{
		assert(!name.empty());
		assert(padding >= 0);

	}

	// initialize layer blobs and params
	virtual void init() = 0;
	
	// compute response (activation value) from inputBlob values
	virtual void activate() = 0;

	virtual BlobShape computeOutputShape() = 0;

	virtual size_t byteSize() = 0;

	virtual void loadHDF5Params(const std::string & filepath) = 0;

	virtual void printParamMoments() = 0;

	// make a connection between bottom layer and top layer
	static void connect(Layer_I & bottomLayer, Layer_I & topLayer) 
	{
		bottomLayer.topLayer = &topLayer;
		topLayer.bottomLayer = &bottomLayer;

		LOG_DUMP("Connected layers: '%s' --> '%s'", bottomLayer.name.c_str(), topLayer.name.c_str());
	}

	~Layer_I()
	{

	}

};
