#define EIGEN_USE_THREADS

#include <functional>
#include <string>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("CropAndResize3DGradBoxes")
    .Input("grads: float")
    .Input("image: T")
    .Input("boxes: float")
    .Input("box_ind: int32")
    .Output("output: float")
    .Attr("T: {uint8, uint16, int8, int16, int32, int64, half, float, double}")
    .Attr("method_name: {'trilinear'} = 'trilinear'")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(2));
      return Status::OK();
    });

static inline Status ParseAndCheckBoxSizes(const Tensor& boxes,
                                           const Tensor& box_index,
                                           int* num_boxes) {
  if (boxes.NumElements() == 0 && box_index.NumElements() == 0) {
    *num_boxes = 0;
    return Status::OK();
  }
  // The shape of 'boxes' is [num_boxes, 6].
  if (boxes.dims() != 2) {
    return errors::InvalidArgument("boxes must be 2-D",
                                   boxes.shape().DebugString());
  }
  *num_boxes = boxes.dim_size(0);
  if (boxes.dim_size(1) != 6) {
    return errors::InvalidArgument("boxes must have 6 columns");
  }
  // The shape of 'box_index' is [num_boxes].
  if (box_index.dims() != 1) {
    return errors::InvalidArgument("box_index must be 1-D",
                                   box_index.shape().DebugString());
  }
  if (box_index.dim_size(0) != *num_boxes) {
    return errors::InvalidArgument("box_index has incompatible shape");
  }
  return Status::OK();
}

class CropAndResize3DGradBoxesOp : public OpKernel {
public:
  explicit CropAndResize3DGradBoxesOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("method_name", &method_name_));
    OP_REQUIRES(context, method_name_ == "trilinear" || method_name_ == "nearest",
                errors::InvalidArgument(
                    "method must be 'trilinear' or 'nearest'", method_name_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& grads = context-> input(0);
    const Tensor& image = context-> input(1);
    const Tensor& boxes = context-> input(2);
    const Tensor& box_index = context-> input(3);

    OP_REQUIRES(context, grads.dims() == 5,
                      errors::InvalidArgument("grads image must be 5-D",
                                              image.shape().DebugString()));

    const int crop_height = grads.dim_size(1);
    const int crop_width = grads.dim_size(2);
    const int crop_depth = grads.dim_size(3);
    const int depth = grads.dim_size(4);
    OP_REQUIRES(
        context, crop_height > 0 && crop_width > 0 && crop_depth > 0,
        errors::InvalidArgument("grads dimensions must be positive"));
    OP_REQUIRES(context, image.dims() == 5,
        errors::InvalidArgument("input image must be 5-D",
                                image.shape().DebugString()));

    const int batch_size = image.dim_size(0);
    const int image_height = image.dim_size(1);
    const int image_width = image.dim_size(2);
    const int image_depth = image.dim_size(3);

    OP_REQUIRES(
        context, image_height > 0 && image_width > 0 && image_depth > 0,
        errors::InvalidArgument("image dimensions must be positive"));
    OP_REQUIRES(
        context, image.dim_size(4) == depth,
        errors::InvalidArgument("image and grads depths are incompatible"));

    int num_boxes = 0;
    OP_REQUIRES_OK(context, ParseAndCheckBoxSizes(boxes, box_index, &num_boxes));

    OP_REQUIRES(
        context, grads.dim_size(0) == num_boxes,
        errors::InvalidArgument("boxes and grads have incompatible shape"));

    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({num_boxes, 6}), &output));

    auto gradsT = grads.tensor<float, 5>();
    auto imageT = image.tensor<float, 5>();
    auto boxesT = boxes.tensor<float, 2>();
    auto box_indexT = box_index.tensor<int32, 1>();
    auto grads_boxes = output->tensor<float, 2>();

    grads_boxes.setZero();

    for (int b = 0; b < num_boxes; ++b) {
      const float y1 = boxesT(b, 0);
      const float x1 = boxesT(b, 1);
      const float z1 = boxesT(b, 2);
      const float y2 = boxesT(b, 3);
      const float x2 = boxesT(b, 4);
      const float z2 = boxesT(b, 5);

      const int32 b_in = box_indexT(b);

      const float height_ratio =
          (crop_height > 1) ? static_cast<float>(image_height - 1) / (crop_height - 1)
              : 0;
      const float width_ratio =
          (crop_width > 1) ? static_cast<float>(image_width - 1) / (crop_width - 1)
                           : 0;
      const float depth_ratio =
          (crop_depth > 1) ? static_cast<float>(image_depth - 1) / (crop_depth - 1)
                       : 0;

      const float height_scale = (crop_height > 1) ? (y2 - y1) * height_ratio : 0;
      const float width_scale = (crop_width > 1) ? (x2 - x1) * width_ratio : 0;
      const float depth_scale = (crop_depth > 1) ? (z2 - y1) * height_ratio : 0;

      for (int y = 0; y < crop_height; ++y) {
        const float in_y = (crop_height > 1)
                               ? y1 * (image_height - 1) + y * height_scale
                               : 0.5 * (y1 + y2) * (image_height - 1);
        if (in_y < 0 || in_y > image_height - 1) {
          continue;
        }
        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;

        for (int x = 0; x < crop_width; ++x) {
          const float in_x = (crop_width > 1)
                                 ? x1 * (image_width - 1) + x * width_scale
                                 : 0.5 * (x1 + x2) * (image_width - 1);
          if (in_x < 0 || in_x > image_width - 1) {
            continue;
          }
          const int left_x_index = floorf(in_x);
          const int right_x_index = ceilf(in_x);
          const float x_lerp = in_x - left_x_index;

          for (int z = 0; z < crop_depth; ++z) {
            const float in_z = (crop_depth > 1)
                                   ? z1 * (image_depth - 1) + z * depth_scale
                                   : 0.5 * (z1 + z2) * (image_depth - 1);
            if (in_z < 0 || in_z > image_depth - 1) {
              continue;
            }
            const int forward_z_index = floorf(in_z);
            const int backward_z_index = ceilf(in_z);
            const float z_lerp = in_z - forward_z_index;

            for (int d = 0; d < depth; ++d) {
              const float top_left_forward(static_cast<float>(
                  imageT(b_in, top_y_index, left_x_index, forward_z_index, d)));
              const float top_left_backward(static_cast<float>(
                  imageT(b_in, top_y_index, left_x_index, backward_z_index, d)));
              const float top_right_forward(static_cast<float>(
                  imageT(b_in, top_y_index, right_x_index, forward_z_index, d)));
              const float top_right_backward(static_cast<float>(
                  imageT(b_in, top_y_index, right_x_index, backward_z_index, d)));
              const float bottom_left_forward(static_cast<float>(
                  imageT(b_in, bottom_y_index, left_x_index, forward_z_index, d)));
              const float bottom_left_backward(static_cast<float>(
                  imageT(b_in, bottom_y_index, left_x_index, backward_z_index, d)));
              const float bottom_right_forward(static_cast<float>(
                  imageT(b_in, bottom_y_index, right_x_index, forward_z_index, d)));
              const float bottom_right_backward(static_cast<float>(
                  imageT(b_in, bottom_y_index, right_x_index, backward_z_index, d)));

              float image_grad_y = (1 - z_lerp) * ( (1 - x_lerp) * (bottom_left_forward - top_left_forward) +
                                                      x_lerp * (bottom_right_forward - top_right_forward) ) +
                                        z_lerp  * ( (1 - x_lerp) * (bottom_left_backward - top_left_backward) +
                                                      x_lerp * (bottom_right_backward - top_right_backward) );
              float image_grad_x = (1 - z_lerp) * ( (1 - y_lerp) * (top_right_forward - top_left_forward) +
                                                      y_lerp * (bottom_right_forward - bottom_left_forward) ) +
                                        z_lerp  * ( (1 - y_lerp) * (top_right_backward - top_left_backward) +
                                                      y_lerp * (bottom_right_backward - bottom_left_backward) );
              float image_grad_z = (1 - x_lerp) * ( (1 - y_lerp) * (top_left_backward - top_left_forward) +
                                                      y_lerp * (bottom_left_backward - bottom_left_forward) ) +
                                        x_lerp  * ( (1 - y_lerp) * (top_right_backward - top_right_forward) +
                                                      y_lerp * (bottom_right_backward - bottom_right_forward) );

              const float top_grad = gradsT(b, y, x, z, d);
              image_grad_y *= top_grad;
              image_grad_x *= top_grad;
              image_grad_z *= top_grad;

              if (crop_height > 1) {
                grads_boxes(b, 0) += image_grad_y * (image_height - 1 - y * height_ratio);
                grads_boxes(b, 3) += image_grad_y * y * height_ratio;
              } else {
                grads_boxes(b, 0) += image_grad_y * 0.5 * (image_height - 1);
                grads_boxes(b, 3) += image_grad_y * 0.5 * (image_height - 1);
              }
              if (crop_width > 1) {
                grads_boxes(b, 1) += image_grad_x * (image_width - 1 - x * width_ratio);
                grads_boxes(b, 4) += image_grad_x * x * width_ratio;
              } else {
                grads_boxes(b, 1) += image_grad_x * 0.5 * (image_width - 1);
                grads_boxes(b, 4) += image_grad_x * 0.5 * (image_width - 1);
              }
              if (crop_depth > 1) {
                grads_boxes(b, 2) += image_grad_z * (image_depth - 1 - z * depth_ratio);
                grads_boxes(b, 5) += image_grad_z * z * depth_ratio;
              } else {
                grads_boxes(b, 2) += image_grad_z * 0.5 * (image_depth - 1);
                grads_boxes(b, 5) += image_grad_z * 0.5 * (image_depth - 1);
              }
            }
          }
        }
      }
    }
  }
private:
 string method_name_ ;
};

REGISTER_KERNEL_BUILDER(Name("CropAndResize3DGradBoxes").Device(DEVICE_CPU), CropAndResize3DGradBoxesOp);
