/**
 * @Author:  xiezhongzhao
 * @Email:   2234309583@qq.com
 * @Data:    2023/7/7 10:28
 * @Version: 1.0
**/

#ifndef SEGMENT_INFER_COMMON_H
#define SEGMENT_INFER_COMMON_H

#pragma once

#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include <iostream>
#include <filesystem>

#include <MNN/ImageProcess.hpp>
#define MNN_OPEN_TIME_TRACE
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>

//#define STB_IMAGE_IMPLEMENTATION
//#include <imageHelper/stb_image.h>
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include <imageHelper/stb_image_write.h>
//#include <imageHelper/stb_image_resize.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace MNN;
using namespace MNN::CV;
using namespace MNN::Express;

#endif //SEGMENT_INFER_COMMON_H
