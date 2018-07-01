#pragma once

#include <algorithm>

#include "Blob.h"
#include "Layer_I.h"


struct ConvLayer : Layer_I
{
	int kernel_n;		    // number of kernels (feature maps)
	int kernel_w;		    // height of kernel
	int kernel_h;		    // width of kernel
	int stride;			    // pixel stride (step) when sliding the kernel windows

	Blob * kernelsBlob;		// storage of kernels (feature maps)
	Blob * biasBlob;		// storage of kernels (feature maps)


	ConvLayer(std::string _name, int _nkernels, int _kwidth, int _kheight, int _kstride = 1, int _kpad = 0) :
		Layer_I(_name, LayerType::CONV, _kpad), kernel_n(_nkernels), kernel_w(_kwidth), kernel_h(_kheight), stride(_kstride), kernelsBlob(nullptr)
	{
		assert(kernel_n > 0);
		assert(kernel_w > 0);
		assert(kernel_h > 0);
		assert(stride > 0);

		LOG_DUMP("Created layer '%s' (%d x %d x %d) - stride: %d, pad: %d", name.c_str(), kernel_n, kernel_h, kernel_w, stride, padding);
	}

	// kernel and bias blob will be created and initialized to zero 
	void init()
	{
		assert(bottomLayer != nullptr);
		assert(bottomLayer->outputBlob != nullptr);

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
		
		Blob * inputBlob = bottomLayer->outputBlob;
		kernelsBlob = new Blob(name + "_kernels", kernel_n, inputBlob->channels, kernel_h, kernel_w);
		biasBlob = new Blob(name + "_bias", kernel_n, 1, 1, 1);


		// compute shape of output blob (can be bigger because of topLayer padding)
		BlobShape outputShape = computeOutputShape();
		std::string outputBlobName = name + "_blob";

		if ((topLayer == nullptr) || (topLayer->type != LayerType::CONCAT))
		{
			outputBlob = new Blob(outputBlobName, outputShape.ndims, outputShape.channels, outputShape.height, outputShape.width);
			ownsOuputBlob = true;
		}
		
		LOG_DUMP("Initialized CONV layer '%s' (%d, %d, %d, %d), output blob '%s' (%d, %d, %d, %d)", 
			name.c_str(), inputBlob->ndims, inputBlob->channels, kernel_h, kernel_w,
			outputBlobName.c_str(), outputShape.ndims, outputShape.channels, outputShape.height - (2 * outputBlobPaddingOffset), outputShape.width - (2 * outputBlobPaddingOffset));

	}

	void activate()
	{
		// inputBlob is defined N x CH x H x W
		//	N - number of images/inputBlob maps
		//	CH - number of image channels / kernels in prev layer
		//	H - height of image / inputBlob map
		//	W - width of image / inputBlob map

		Blob * inputBlob = bottomLayer->outputBlob;

		float sum, kval, ival, bias;
		int x, y, n, k_n, o_y, o_x, k_z, k_y, k_x;
		int ival_index, kval_index, oval_index;
		
		// store members to local variables for slighly faster access
		// precompute offsets sequentially to reduce number of multiplications in nested loop
		const int kernelsBlob_width = kernelsBlob->width;
		const int kernelsBlob_height = kernelsBlob->height;
		const int kernelsBlob_channels = kernelsBlob->channels;
		const int kernelsBlob_patch_size_offset = kernelsBlob_height * kernelsBlob_width;
		int kernelsBlob_n_dim_offset;
		int kernelsBlob_ch_dim_offset;

		const int inputBlob_width = inputBlob->width;
		const int inputBlob_height = inputBlob->height;
		const int inputBlob_channels = inputBlob->channels;
		const int inputBlob_patch_size_offset = inputBlob_height * inputBlob_width;
		int inputBlob_n_dim_offset;
		int inputBlob_ch_dim_offset;
		
		const int outputBlob_width = outputBlob->width;
		const int outputBlob_height = outputBlob->height;
		const int outputBlob_channels = outputBlob->channels;
		const int outputBlob_patch_size_offset = outputBlob_height * outputBlob_width;
		int outputBlob_n_dim_offset;
		int outputBlob_ch_dim_offset;

		const float* kernelsBlob_data = kernelsBlob->data;
		const float* inputBlob_data = inputBlob->data;
		float* outputBlob_data = outputBlob->data;

		// for every image / inputBlob map
		for (n = 0; n < inputBlob->ndims; n++)
		{
			inputBlob_n_dim_offset = n * inputBlob_channels * inputBlob_patch_size_offset;
			outputBlob_n_dim_offset = n * outputBlob_channels * outputBlob_patch_size_offset;

			// for every kernel feature map
			for (k_n = 0; k_n < kernel_n; k_n++)
			{
				bias = biasBlob->data[k_n];
				kernelsBlob_n_dim_offset = k_n * kernelsBlob_channels * kernelsBlob_patch_size_offset;
				outputBlob_ch_dim_offset = (k_n + channelsOffset) * outputBlob_patch_size_offset;		// if on top there is a concat layer, shift memory index by channels offset

				// for every row in outputBlob map
				// take only valid output pixels for this layer since output blob could be padded for topLayer
				for (o_y = outputBlobPaddingOffset; o_y < (outputBlob_height - outputBlobPaddingOffset); o_y++)
				{
					// y mapped to inputBlob image and aligned depending on bottom layer padding
					y = (o_y * stride) - outputBlobPaddingOffset + inputBlobPaddingOffset;		

					// for every column in outputBlob map
					// take only valid output pixels for this layer since output blob could be padded for topLayer
					for (o_x = outputBlobPaddingOffset; o_x < (outputBlob_width - outputBlobPaddingOffset); o_x++)
					{
						// x mapped to inputBlob image and aligned depending on bottom layer padding
						x = (o_x * stride) - outputBlobPaddingOffset + inputBlobPaddingOffset;		

						sum = 0;		// compute convolution for given pixel [o_x, o_y]

						// for every input channel
						for (k_z = 0; k_z < inputBlob_channels; k_z++)
						{
							inputBlob_ch_dim_offset = k_z * inputBlob_patch_size_offset;
							kernelsBlob_ch_dim_offset = k_z * kernelsBlob_patch_size_offset;
							
							// for every row in kernel layer (sample patch)
							for (k_y = 0; k_y < kernel_h; k_y++)
							{

								// for every column in kernel layer (sample patch)
								for (k_x = 0; k_x < kernel_w; k_x++)
								{
									assert(((x + k_x) >= inputBlobPaddingOffset) && ((x + k_x) < (inputBlob_width - inputBlobPaddingOffset)));
									assert(((y + k_y) >= inputBlobPaddingOffset) && ((y + k_y) < (inputBlob_height - inputBlobPaddingOffset)));
									
									kval_index = kernelsBlob_n_dim_offset + kernelsBlob_ch_dim_offset + k_y*(kernelsBlob_width) + k_x;
									ival_index = inputBlob_n_dim_offset + inputBlob_ch_dim_offset + (y + k_y)*(inputBlob_width) + (x + k_x);

									assert((ival_index >= 0) && (ival_index < inputBlob->size()));
									assert((kval_index >= 0) && (kval_index < kernelsBlob->size()));

									kval = kernelsBlob_data[kval_index];
									ival = inputBlob_data[ival_index];

									sum += kval * ival;
								}
							}
						}
						oval_index = outputBlob_n_dim_offset + outputBlob_ch_dim_offset + o_y*(outputBlob_width) + o_x;
						assert((oval_index >= 0) && (oval_index < outputBlob->size()));
						
						sum += bias;                                    // apply bias
						outputBlob_data[oval_index] = max(sum, 0);      // appy relu
					}
				}
			}
		}

	}

	// computes shape of outputBlob depending on shape of input blob and kernel parameters
	BlobShape computeOutputShape()
	{
		assert(bottomLayer != nullptr);
		assert(bottomLayer->outputBlob != nullptr);
		
		BlobShape shape = { 0 };
		Blob * inputBlob = bottomLayer->outputBlob;

		// pooling layer shape in Caffe is computed using float->int (floor) casting method with contrast to pool layer
		int h_o = static_cast<int>(floor(static_cast<float>(inputBlob->height - (2 * inputBlobPaddingOffset) - kernel_h) / stride) + 1 + (2 * outputBlobPaddingOffset));
		int w_o = static_cast<int>(floor(static_cast<float>(inputBlob->width - (2 * inputBlobPaddingOffset) - kernel_w) / stride) + 1 + (2 * outputBlobPaddingOffset));
		assert((h_o > 0) && (w_o > 0));		// just check if given shape params makes sense

		shape.ndims = inputBlob->ndims;
		shape.channels = kernel_n;
		shape.height = h_o;
		shape.width = w_o;

		return shape;
	}

	// computes byse size of data stored on heap (blobs) for layer
	size_t byteSize()
	{
		size_t kernelByteSize = 0;
		if (kernelsBlob != nullptr)
		{
			kernelByteSize += kernelsBlob->ndims * kernelsBlob->channels * kernelsBlob->height * kernelsBlob->width * sizeof(float);
		}

		size_t outputByteSize = 0;
		if ((outputBlob != nullptr) && (ownsOuputBlob == true))
		{
			outputByteSize += outputBlob->ndims * outputBlob->channels * outputBlob->height * outputBlob->width * sizeof(float);
		}

		LOG_DUMP("Layer '%s' byte size: %d (%.4f MB)", name.c_str(), kernelByteSize + outputByteSize, static_cast<float>(kernelByteSize + outputByteSize) / (1024 * 1024));
	
		return kernelByteSize + outputByteSize;
	}

	void loadHDF5Params(const std::string & filepath)
	{
		LOG_DUMP("Loading params for layer '%s'", this->name.c_str());
		
		hid_t file_hid = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		assert(file_hid >= 0);

		hid_t data_hid = H5Gopen2(file_hid, "data", H5P_DEFAULT);
		assert(data_hid >= 0);

		std::string layerName(this->name);
		std::replace(layerName.begin(), layerName.end(), '/', '_');         // replace all '/' to '_' since hdf5 has speacial behaviour for slashes

		hid_t layer_hid = H5Gopen2(data_hid, layerName.c_str(), H5P_DEFAULT);
		assert(layer_hid >= 0);

		int num_source_params = hdf5_get_num_links(layer_hid);
		assert(num_source_params == 2);         // kernel params and bias

		herr_t readStatus = 0;

		// --------------- read kernel params -------------------

		const char * kernelParamsDstName = "0";     // first are stored kernel params with ID "0"
		assert(H5Lexists(layer_hid, kernelParamsDstName, H5P_DEFAULT));      
		
		hid_t kernelParamsDst = H5Dopen2(layer_hid, kernelParamsDstName, H5P_DEFAULT);
		assert(kernelParamsDst >= 0);       // 0 is kernel params

		// read info about stored data
		hid_t kernelDSpace = H5Dget_space(kernelParamsDst);       // load info about data space (data storage)
		assert(kernelDSpace >= 0);

		const int kernel_ndims = H5Sget_simple_extent_ndims(kernelDSpace);
		assert(kernel_ndims == 4);
		
		hsize_t * kernelParamsDstShape = new hsize_t[kernel_ndims];
		H5Sget_simple_extent_dims(kernelDSpace, kernelParamsDstShape, NULL);

		assert(kernelsBlob->ndims == kernelParamsDstShape[0]);
		assert(kernelsBlob->channels == kernelParamsDstShape[1]);
		assert(kernelsBlob->height == kernelParamsDstShape[2]);
		assert(kernelsBlob->width == kernelParamsDstShape[3]);

		H5Sclose(kernelDSpace);
		H5Dclose(kernelParamsDst);

		readStatus = H5LTread_dataset_float(layer_hid, kernelParamsDstName, kernelsBlob->data);
		assert(readStatus >= 0);

		// cleanup
		delete[] kernelParamsDstShape;

		// --------------- read bias params -----------------------

		const char * biasParamsDstName = "1";     // first are stored kernel params with ID "1"
		assert(H5Lexists(layer_hid, biasParamsDstName, H5P_DEFAULT));       // 1 is kernel params

		hid_t biasParamsDst = H5Dopen2(layer_hid, biasParamsDstName, H5P_DEFAULT);
		assert(biasParamsDst >= 0);       

		// read info about stored data
		hid_t biasDSpace = H5Dget_space(biasParamsDst);       // load info about data space (data storage)
		assert(biasDSpace >= 0);

		const int bias_ndims = H5Sget_simple_extent_ndims(biasDSpace);
		assert(bias_ndims == 1);

		hsize_t * biasParamsDstShape = new hsize_t[bias_ndims];
		H5Sget_simple_extent_dims(biasDSpace, biasParamsDstShape, NULL);
		assert(biasBlob->ndims == biasParamsDstShape[0]);

		H5Sclose(biasDSpace);
		H5Dclose(biasParamsDst);

		readStatus = H5LTread_dataset_float(layer_hid, biasParamsDstName, biasBlob->data);
		assert(readStatus >= 0);

		// cleanup
		delete[] biasParamsDstShape;

		// -----------------------------------------------------------

		H5Gclose(layer_hid);
		H5Gclose(data_hid);
		H5Fclose(file_hid);

	}

	void printParamMoments()
	{
		kernelsBlob->printMoments();
		biasBlob->printMoments();
	}

	~ConvLayer()
	{
		if (kernelsBlob != nullptr)
		{
			delete kernelsBlob;
			kernelsBlob = nullptr;
		}
		if (biasBlob != nullptr)
		{
			delete biasBlob;
			biasBlob = nullptr;
		}

		// layers which are concatenated are not holding any output data, but writes into concat layer storage directly
		if (ownsOuputBlob == true)
		{
			assert(outputBlob != nullptr);
			delete outputBlob;
		}
		outputBlob = nullptr;
	}
};
