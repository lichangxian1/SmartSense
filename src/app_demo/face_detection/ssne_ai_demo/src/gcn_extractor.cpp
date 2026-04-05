#include "../include/common.hpp"
#include <iostream>
#include <cmath>

namespace utils {
    // 这里可以保留官方的一些辅助函数，如果没有就不写
}

void GCNV2_EXTRACTOR::Initialize(std::string& model_path, std::array<int, 2>* in_det_shape) {
    det_shape = *in_det_shape; // {640, 480}
    
    // 获取硬件预处理管道 (极低延迟)
    pipe_offline = GetAIPreprocessPipe();

    // 1. 加载 m1model 到 NPU
    char* model_path_char = const_cast<char*>(model_path.c_str());
    model_id = ssne_loadmodel(model_path_char, SSNE_STATIC_ALLOC);

    // 2. 申请输入显存 (严格使用官方的 SSNE_BUF_AI 宏)
    uint32_t det_width = static_cast<uint32_t>(det_shape[0]);
    uint32_t det_height = static_cast<uint32_t>(det_shape[1]);
    inputs[0] = create_tensor(det_width, det_height, SSNE_Y_8, SSNE_BUF_AI);
    std::cout << "[INFO] GCNv2 Extractor Initialized. Shape: " << det_width << "x" << det_height << std::endl;
}

void GCNV2_EXTRACTOR::Predict(ssne_tensor_t* img_in, GcnResult* result, float conf_threshold) {
    result->Clear();

    // 1. 硬件级预处理 (调用 A1 内置图像加速器完成缩放/格式转换)
    int ret = RunAiPreprocessPipe(pipe_offline, *img_in, inputs[0]);
    if (ret != 0) {
        std::cerr << "[ERROR] AI Preprocess Pipe failed!" << std::endl;
        return;
    }

    // 2. 触发 NPU 矩阵计算
    if (ssne_inference(model_id, 1, inputs)) {
        std::cerr << "[ERROR] SSNE Inference failed!" << std::endl;
        return;
    }

    // 3. 拿到 2 个输出 (Outputs[0]=prob, Outputs[1]=desc)
    ssne_getoutput(model_id, 2, outputs);
    Postprocess(result, conf_threshold);
}

void GCNV2_EXTRACTOR::Postprocess(GcnResult* result, float conf_threshold) {
    // 假设导出时 prob 是第 0 个输出
    float* prob_data = (float*)get_data(outputs[0]); 
    // float* desc_data = (float*)get_data(outputs[1]); // 描述子后续供 SLAM 匹配用，暂不解析

    // 根据 640x480 的输入，总降采样率为 8
    const int GRID_H = det_shape[1] / 8; // 480 / 8 = 60
    const int GRID_W = det_shape[0] / 8; // 640 / 8 = 80
    const int CHANNELS = 65;

    for (int h = 0; h < GRID_H; ++h) {
        for (int w = 0; w < GRID_W; ++w) {
            int max_c = 0;
            float max_val = -1e9;

            // 找 65 通道中的最大值
            for (int c = 0; c < CHANNELS; ++c) {
                // 【替换为这行 A1 硬件专属的 NHWC 公式】
                int idx = h * (GRID_W * CHANNELS) + w * CHANNELS + c;
                if (prob_data[idx] > max_val) {
                    max_val = prob_data[idx];
                    max_c = c;
                }
            }

            // 如果不是第 65 个（垃圾桶通道），且置信度够高
            if (max_c < 64) {
                float score = exp(max_val); // 简易 Softmax
                if (score > conf_threshold) {
                    float pt_x = w * 8 + (max_c % 8);
                    float pt_y = h * 8 + (max_c / 8);
                    result->keypoints.push_back({pt_x, pt_y});
                    result->scores.push_back(score);
                }
            }
        }
    }
}

void GCNV2_EXTRACTOR::Release() {
    release_tensor(inputs[0]);
    release_tensor(outputs[0]);
    release_tensor(outputs[1]);
    ReleaseAIPreprocessPipe(pipe_offline);
    std::cout << "[INFO] GCNv2 Extractor Released." << std::endl;
}