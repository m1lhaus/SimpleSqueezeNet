#pragma once

#include "PoolLayer_I.h"

struct MAXPoolLayer : PoolLayer_I
{

	using PoolLayer_I::PoolLayer_I;

	void activate()
	{
		// inputBlob is defined N x CH x H x W
		//	N - number of images/inputBlob maps
		//	CH - number of image channels / kernels in prev layer
		//	H - height of image / inputBlob map
		//	W - width of image / inputBlob map

		Blob * inputBlob = bottomLayer->outputBlob;

		float max, ival;
		int x_start, x_end, y_start, y_end;
		int n, ch_n, o_y, o_x, w_y, w_x;;
		int ival_index, oval_index;

		// store members to local variables for slighly faster access
		// precompute offsets sequentially to reduce number of multiplications in nested loop
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

		const float* inputBlob_data = inputBlob->data;
		float* outputBlob_data = outputBlob->data;

		// for every image / inputBlob map
		for (n = 0; n < inputBlob->ndims; n++)
		{
			inputBlob_n_dim_offset = n * inputBlob_channels * inputBlob_patch_size_offset;
			outputBlob_n_dim_offset = n * outputBlob_channels * outputBlob_patch_size_offset;

			 //for every blob channel
			for (ch_n = 0; ch_n < inputBlob_channels; ch_n++)
			{
				inputBlob_ch_dim_offset = ch_n * inputBlob_patch_size_offset;
				outputBlob_ch_dim_offset = (ch_n + channelsOffset) * outputBlob_patch_size_offset;

				// for every row in outputBlob map
				// take only valid output pixels for this layer since output blob could be padded for topLayer
				for (o_y = outputBlobPaddingOffset; o_y < (outputBlob_height - outputBlobPaddingOffset); o_y++)
				{
					y_start = (o_y * stride) - outputBlobPaddingOffset + inputBlobPaddingOffset;							// y mapped to inputBlob image - TODO padding
					y_end = min(y_start + kernel_h, inputBlob_height - inputBlobPaddingOffset);

					
					// for every column in outputBlob map
					// take only valid output pixels for this layer since output blob could be padded for topLayer
					for (o_x = outputBlobPaddingOffset; o_x < (outputBlob_width - outputBlobPaddingOffset); o_x++)
					{
						x_start = (o_x * stride) - outputBlobPaddingOffset + inputBlobPaddingOffset;						// x mapped to inputBlob image - TODO padding
						x_end = min(x_start + kernel_w, inputBlob_width - inputBlobPaddingOffset);
						
						// get max inputBlob response for given pixel [o_x, o_y]
						max = -99999;
						
						// for every row in window
						for (w_y = y_start; w_y < y_end; w_y++)
						{
							
							// for every column in window
							for (w_x = x_start; w_x < x_end; w_x++)
							{
								ival_index = inputBlob_n_dim_offset + inputBlob_ch_dim_offset + (w_y*inputBlob_width) + w_x;
								assert((ival_index >= 0) && (ival_index < inputBlob->size()));

								ival = inputBlob_data[ival_index];
								if (ival > max)
									max = ival;
							}
						}
						oval_index = outputBlob_n_dim_offset + outputBlob_ch_dim_offset + (o_y*outputBlob_width) + o_x;
						assert((oval_index >= 0) && (oval_index < outputBlob->size()));

						outputBlob_data[oval_index] = max;
					}
				}
			}
		}

	}

};