/**
 * @Author:  xiezhongzhao
 * @Email:   2234309583@qq.com
 * @Data:    2023/7/5 18:00
 * @Version: 1.0
**/
#include <modnet/inference.h>

int main(){

    std::string root_dir = "/mnt/e/modnet/";
    std::string img_path = root_dir + "data/sky/1.jpg";
    const std::string mnn_path= root_dir + "model/sky/sky_modnet_sim_512_672_opset16.mnn";

    // 1. model initialization
    MODNET *modnet = new MODNET(mnn_path, 2); // 1 thread

    // 2. load the image
    cv::Mat raw_img = cv::imread(img_path);

    // 3. model inference
    cv::Mat alpha = modnet->detect(raw_img); // 返回所需mask结果

    cv::imwrite(root_dir + "alpha.png", alpha);
    std::cout << "inference finished!!!" << std::endl;

    return 0;
}








