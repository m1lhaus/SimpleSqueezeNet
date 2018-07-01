#pragma once

#include <cassert>

#include "helper_defines.h"


struct Blob_I {

	std::string name;
	int ndims, channels, height, width;

	Blob_I(std::string _name, int _ndim, int _channels, int _height, int _width) :
		name(_name), ndims(_ndim), channels(_channels), height(_height), width(_width)
	{
		assert(ndims >= 0);
		assert(channels >= 0);
		assert(height >= 0);
		assert(width >= 0);
		assert((ndims * channels * height * width) > 0);
	}

	//Blob_I(std::string _name) : name(_name), ndims(0), channels(0), height(0), width(0)
	//{

	//}

	//Blob_I() : name(std::string()), ndims(0), channels(0), height(0), width(0)
	//{

	//}

	//virtual float& operator()(int n, int ch, int y, int x) = 0;

	//virtual inline float * get_data(size_t n, size_t ch, size_t y, size_t x) = 0;

	//~Blob_I()
	//{

	//}

};
