/**
 * @Author:  xiezhongzhao
 * @Email:   2234309583@qq.com
 * @Data:    2023/7/7 10:16
 * @Version: 1.0
**/

#include <modnet/mnn_handler.h>

BasicMNNHandler::BasicMNNHandler(
        const std::string &mnn_path, unsigned int num_threads):
        log_id(mnn_path.data()), mnn_path(mnn_path.data()),
        num_threads(num_threads){
    initialize_handler();
}

void BasicMNNHandler::initialize_handler() {
    // 1. init interpreter
    mnn_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path));
    // 2. init schedule_config
    schedule_config.numThread = (int) num_threads;
    MNN::BackendConfig backend_config;
    backend_config.precision = MNN::BackendConfig::Precision_High; // default Precision_High
    schedule_config.backendConfig = &backend_config;
    // 3. create session
    mnn_session = mnn_interpreter->createSession(schedule_config);
    // 4. init input tensor
    input_tensor = mnn_interpreter->getSessionInput(mnn_session, nullptr);
    // 5. init input dims
    input_batch = input_tensor->batch();
    input_channel = input_tensor->channel();
    input_height = input_tensor->height();
    input_width = input_tensor->width();

    dimension_type = input_tensor->getDimensionType();

    // output count
    num_outputs = mnn_interpreter->getSessionOutputAll(mnn_session).size();
    this->print_debug_string();
}

BasicMNNHandler::~BasicMNNHandler() {
    mnn_interpreter->releaseModel();
    if(mnn_session)
        mnn_interpreter->releaseSession(mnn_session);
}

void BasicMNNHandler::print_debug_string() {
    std::cout << "========================= Input-Dims =======================\n";
    if(input_tensor)
        input_tensor->printShape();
    std::cout << "========================= Output-Dims ======================\n";
    auto tmp_output_map = mnn_interpreter->getSessionOutputAll(mnn_session);
    std::cout << "getSessionOutputAll done !\n" ;
    for(auto it=tmp_output_map.cbegin(); it!=tmp_output_map.cend(); ++it){
        std::cout << "Output: " << it->first << ": ";
        it->second->printShape();
    }
    std::cout << "============================================================\n";
}



