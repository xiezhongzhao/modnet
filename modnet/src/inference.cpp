/**
 * @Author:  xiezhongzhao
 * @Email:   2234309583@qq.com
 * @Data:    2023/7/5 17:56
 * @Version: 1.0
**/

#include <modnet/inference.h>

MODNET::MODNET(const std::string &mnn_path, unsigned int num_threads)
    : BasicMNNHandler(mnn_path, num_threads){
    initialzie_pretreat();
}

inline void MODNET::initialzie_pretreat() {
    pretreat = std::shared_ptr<MNN::CV::ImageProcess>(
            MNN::CV::ImageProcess::create(
                    MNN::CV::BGR,
                    MNN::CV::RGB,
                    mean_vals, 3,
                    norm_vals, 3
                    )
            );
}

void MODNET::transform(const cv::Mat &mat) {
    cv::Mat canvas;

    cv::resize(mat, canvas,
               cv::Size(input_width, input_height));
    pretreat->convert(canvas.data, input_width, input_height,
                      canvas.step[0], input_tensor);
}

cv::Mat MODNET::detect(const cv::Mat &mat){
    if(mat.empty()) return mat;
    // 1. make input tensor
    this->transform(mat);
    // 2. inference
    mnn_interpreter->runSession(mnn_session);
    auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);
    // 3. generate matting
    cv::Mat alpha = this->generate_matting(output_tensors, mat);
    return alpha;
}

cv::Mat MODNET::generate_matting(const std::map<std::string, MNN::Tensor *> &output_tensors,
                              const cv::Mat &mat){
    auto device_output_ptr = output_tensors.at("pred_matte"); // alpha node
    MNN::Tensor host_output_tensor(device_output_ptr,
                                   device_output_ptr->getDimensionType());
    device_output_ptr->copyToHostTensor(&host_output_tensor);
    const unsigned int h = mat.rows;
    const unsigned int w = mat.cols;

    auto output_dims = host_output_tensor.shape();
    const unsigned int out_h = output_dims.at(2);
    const unsigned int out_w = output_dims.at(3);

    float *output_ptr = host_output_tensor.host<float>();
    cv::Mat alpha_pred(out_h, out_w, CV_32FC1, output_ptr);
//    cv::imshow("alpha", alpha_pred); // debug
//    cv::waitKey(0); // debug
//    cv::destroyAllWindows(); // debug

    alpha_pred = alpha_pred * 255;
    cv::resize(alpha_pred, alpha_pred, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);
//    cv::imwrite("alpha.png", alpha_pred); // debug
    return alpha_pred;
}








