/**
 * @Author:  xiezhongzhao
 * @Email:   2234309583@qq.com
 * @Data:    2023/7/5 17:57
 * @Version: 1.0
**/
#ifndef SEGMENT_INFER_INFERENCE_H
#define SEGMENT_INFER_INFERENCE_H

#include "common.h"
#include <modnet/mnn_handler.h>

class MODNET : public BasicMNNHandler{
public:
    explicit MODNET(const std::string &mnn_path, unsigned int num_threads);
    ~MODNET() override = default;

private:
    void initialzie_pretreat();
    void transform(const cv::Mat &mat) override; // resize & normalize
    cv::Mat generate_matting(const std::map<std::string, MNN::Tensor *> &output_tensors,
                          const cv::Mat &mat);

private:
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.f/127.5f, 1.f/127.5, 1.f/127.5f};

public:
    cv::Mat detect(const cv::Mat& mat);
};

#endif //SEGMENT_INFER_INFERENCE_H
