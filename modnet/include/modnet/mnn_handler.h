/**
 * @Author:  xiezhongzhao
 * @Email:   2234309583@qq.com
 * @Data:    2023/7/7 10:16
 * @Version: 1.0
**/

#ifndef SEGMENT_INFER_MNN_HANDLER_H
#define SEGMENT_INFER_MNN_HANDLER_H

#include "common.h"

class BasicMNNHandler{
protected:
    std::shared_ptr<MNN::Interpreter> mnn_interpreter;
    MNN::Session *mnn_session = nullptr;
    MNN::Tensor *input_tensor = nullptr;
    MNN::ScheduleConfig schedule_config;
    std::shared_ptr<MNN::CV::ImageProcess> pretreat;
    const char *log_id = nullptr;
    const char *mnn_path = nullptr;

protected:
    unsigned int num_threads; //
    int input_batch;
    int input_channel;
    int input_height;
    int input_width;
    int dimension_type;
    int num_outputs = 1;

protected:
    explicit BasicMNNHandler(const std::string &mnn_path, unsigned int num_thread);
    virtual ~BasicMNNHandler();
// un-copyable
protected:
    BasicMNNHandler(const BasicMNNHandler &) = delete;
    BasicMNNHandler(BasicMNNHandler &&); //
    BasicMNNHandler &operator=(const BasicMNNHandler &) = delete;
    BasicMNNHandler &operator=(BasicMNNHandler &&) = delete;

protected:
    virtual void transform(const cv::Mat &mat) = 0; // needed ?

private:
    void initialize_handler();
    void print_debug_string();
};

#endif //SEGMENT_INFER_MNN_HANDLER_H
