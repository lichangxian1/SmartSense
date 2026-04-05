#include <iostream>
#include <thread>
#include <unistd.h>
#include <algorithm> // 用于 std::max 和 std::min
#include "include/utils.hpp"
#include "include/common.hpp"

using namespace std;

// 官方预留的线程安全退出标志
bool g_exit_flag = false;

int main() {
    // 1. 恢复官方物理屏幕的真实分辨率 (720x1280 竖屏)
    array<int, 2> screen_shape = {720, 1280}; 
    // GCN 模型的实际输入尺寸
    array<int, 2> model_shape = {640, 480}; 
    string model_path = "/app_demo/app_assets/models/937a1331-cad7-43b2-b597-5d96111550ee_gcnv2_a1_640.m1model";
    
    ssne_initial();
    
    // 初始化时，流获取和屏幕投射使用 720x1280
    IMAGEPROCESSOR processor;
    processor.Initialize(&screen_shape);
    VISUALIZER visualizer;
    visualizer.Initialize(screen_shape);
    
    // 模型推理器依然使用 640x480
    GCNV2_EXTRACTOR extractor;
    extractor.Initialize(model_path, &model_shape);
    GcnResult* gcn_result = new GcnResult;
    
    ssne_tensor_t img_sensor;
    
    // 计算坐标放大的比例系数
    float scale_x = 720.0f / 640.0f;
    float scale_y = 1280.0f / 480.0f;

    int frame_cnt = 0;
    while (true) { 
        processor.GetImage(&img_sensor);
        
        // 推理 (0.6f 阈值)
        extractor.Predict(&img_sensor, gcn_result, 0.6f); 
        
        std::vector<std::array<float, 4>> osd_points;
        for (size_t i = 0; i < gcn_result->keypoints.size(); i++) {
            // 💥 将模型输出的坐标，等比例放大到真实屏幕上
            float x = gcn_result->keypoints[i][0] * scale_x;
            float y = gcn_result->keypoints[i][1] * scale_y;
            
            // 依然画 4x4 的小准星
            float x1 = std::max(0.0f, x - 2.0f);
            float y1 = std::max(0.0f, y - 2.0f);
            float x2 = std::min(719.0f, x + 2.0f); // 屏幕最大宽度 720
            float y2 = std::min(1279.0f, y + 2.0f); // 屏幕最大高度 1280
            
            osd_points.push_back({x1, y1, x2, y2});
        }
        
        visualizer.Draw(osd_points);
        // ==========================================
        
        // 为了防止终端刷屏太快，每 30 帧打印一次状态
        if (frame_cnt % 30 == 0) {
            cout << "[Frame " << frame_cnt << "] OSD绘制点数: " << osd_points.size() << endl;
        }
        
        frame_cnt++;
    }

    // 6. 清理现场 (因为改成了 while(true)，这段代码通常通过强杀进程退出时由 OS 回收)
    delete gcn_result;
    extractor.Release();
    processor.Release();
    visualizer.Release(); // 释放屏幕资源
    ssne_release();
    
    return 0;
}