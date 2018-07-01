#pragma once

#include <cstdint>
#include <cstring>
#include <cassert>
#include <cstdio>

#include "Blob_I.h"

struct BlobShape
{
	int ndims;
	int channels;
	int height;
	int width;
};


struct Blob : public Blob_I {

	float* data;


	Blob(std::string _name, int _ndims, int _channels, int _height, int _width, float* _data=nullptr, bool copy=true) : data(nullptr), Blob_I(_name, _ndims, _channels, _height, _width)
	{
		size_t size = _ndims * _channels * _height * _width;
		if (_data != nullptr)
		{
			if (copy)
			{
				this->data = new float[size];
				std::memcpy(this->data, _data, size * sizeof(float));
			}
			else
			{
				this->data = _data;
			}
		} 
		else
		{
			this->data = new float[size];
			std::memset(this->data, 0, size * sizeof(float));
		}
	}

	int size()
	{
		return ndims*channels*width*height;
	}

	float& operator()(int n, int ch, int y, int x)
	{
		assert(n < ndims && ch < channels && y < height && x < width);
		return data[n*(channels * width * height) + ch*(width * height) + y*(width) + x];
	}

	float mean()
	{
		size_t size = this->size();
		float * dataPtr = this->data;

		long double mean = 0;
		for (int i = 0; i < size; ++i)
		{
			mean += *dataPtr;
			dataPtr++;
		}
		mean /= size;

		return static_cast<float>(mean);
	}

	float var()
	{
		size_t size = this->size();
		float mean = this->mean();
		float * dataPtr = this->data;

		long double var = 0;
		for (int i = 0; i < size; ++i)
		{
			var += (*dataPtr - mean) * (*dataPtr - mean);
			dataPtr++;
		}
		var /= size;

		return static_cast<float>(var);
	}

	float var(float mean)
	{
		size_t size = this->size();
		float * dataPtr = this->data;

		long double var = 0;
		for (int i = 0; i < size; ++i)
		{
			var += (*dataPtr - mean) * (*dataPtr - mean);
			dataPtr++;
		}
		var /= size;

		return static_cast<float>(var);
	}

	void printBlob() {
		printf("BLOB: %s\n", name.c_str());
		
		float value;
		for (int n = 0; n < ndims; n++)
		{
			printf("[Dim: %d]\n", n);
			printf("=========\n");

			for (int ch = 0; ch < channels; ch++)
			{
				printf("[Channel: %d]\n", ch);
			
				for (int y = 0; y < height; y++)
				{
					for (int x = 0; x < width; x++)
					{
						value = data[n*(channels * width * height) + ch*(width * height) + y*(width)+x];
						printf("%12.8f ", value);
					}
					printf("\n");
				}
			}
		}
	}

	void printMoments()
	{
		float mean = this->mean();
		float var = this->var(mean);
		printf("-----------------------------\n");
		printf("BLOB: %s\n", name.c_str());
		printf("-----------------------------\n");
		printf("Mean: %.8f\n", mean);
		printf("Var: %.8f\n", var);
	}


	~Blob()
	{
		if (data != nullptr)
		{
			delete[] data;
			data = nullptr;
		}
	}
};
