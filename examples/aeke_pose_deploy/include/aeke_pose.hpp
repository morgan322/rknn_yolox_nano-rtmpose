#ifndef aekePose_H
#define aekePose_H

#include "rknn_api.h"

#include "opencv2/core/core.hpp"
#include "postprocess.h"
#include "preprocess.h"


static void dump_tensor_attr(rknn_tensor_attr *attr);
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static int saveFloat(const char *file_name, float *output, int element_size);

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    rknn_input inputs[1];

    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
} rknn_app_context_t;

class aekePose
{
private:
    int ret;
    std::mutex mtx;
    std::string det_model_path;
    std::string pose_model_path;

    

    rknn_app_context_t det_ctx;
    rknn_app_context_t pose_ctx;

    int img_width, img_height;

    float nms_threshold, box_conf_threshold;
    
public:
    aekePose(const std::string &det_model_path,const std::string &pose_model_path);
    int init();
    rknn_context *get_pctx();
    FrameInfo infer(FrameInfo &Info);
    ~aekePose();
};


#endif