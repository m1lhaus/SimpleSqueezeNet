#pragma once

#include <vector>

#include "Layer_I.h"

struct Net
{
    std::vector<Layer_I *> layers;

    void addLayer(Layer_I & layer)
    {
        layers.push_back(&layer);
    }

    void init()
    {
        LOG_DUMP("Initializing %d layers...", layers.size());

        for (Layer_I * layer : layers)
        {
            layer->init();
        }
    }

    void forward()
    {
        forward(static_cast<int>(layers.size()));
    }

    void forward(const int upToLayer)
    {
        for (int i = 0; i < upToLayer; ++i)
        {
            Layer_I * layer = layers[i];
            layer->activate();
        }
    }

    void copyTrainedLayersFromHDF5(const std::string & filepath)
    {
        for (Layer_I * layer : layers)
        {
            layer->loadHDF5Params(filepath);
        }
    }

    size_t calcByteSize()
    {
        size_t byteSize = 0;

        for (Layer_I * layer : layers)
        {
            byteSize += layer->byteSize();
        }

        return byteSize;
    }

    void printParamMoments()
    {
        for (Layer_I * layer : layers)
        {
            layer->printParamMoments();
        }
    }
};