#pragma  once

#include <cstdio>

#include "hdf5.h"

#define LOG(errtype, format_str, ...) printf((std::string(errtype) + std::string(": ") + std::string(format_str) + std::string("\n")).c_str(), __VA_ARGS__)

#define LOG_ERROR(format_str,...)		LOG("ERROR"  , format_str , __VA_ARGS__)
#define LOG_WARNING(format_str,...)	LOG("WARNING", format_str , __VA_ARGS__)
#define LOG_INFO(format_str,...)		LOG("INFO"   , format_str , __VA_ARGS__)
#define LOG_DUMP(format_str,...)		LOG("DEBUG"   , format_str , __VA_ARGS__)

int hdf5_get_num_links(hid_t loc_id) {
    H5G_info_t info;
    herr_t status = H5Gget_info(loc_id, &info);
    assert(status >= 0);

    return static_cast<int>(info.nlinks);
}