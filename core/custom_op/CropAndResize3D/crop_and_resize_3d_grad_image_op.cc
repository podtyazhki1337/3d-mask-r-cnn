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

REGISTER_OP("CropAndResize3DGradImage")
    .Input("grads: float")
    .Input("boxes: float")
    .Input("box_ind: int32")
    .Input("image_size: int32")
    .Output("output: T")
    .Attr("T: {float, half, double}")
    .Attr("method_name: {'trilinear', 'nearest'} = 'trilinear'")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(3, &out));
      TF_RETURN_IF_ERROR(c->WithRank(out, 5, &out));
      c->set_output(0, out);
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

class CropAndResize3DGradImageOp : public OpKernel {
public:
  explicit CropAndResize3DGradImageOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("method_name", &method_name_));
    OP_REQUIRES(context, method_name_ == "trilinear" || method_name_ == "nearest",
                errors::InvalidArgument(
                    "method must be 'trilinear' or 'nearest'", method_name_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& grads = context-> input(0);
    const Tensor& boxes = context-> input(1);
    const Tensor& box_index = context-> input(2);
    const Tensor& image_size = context-> input(3);

    OP_REQUIRES(context, grads.dims() == 5,
                      errors::InvalidArgument("grads image must be 5-D",
                                              grads.shape().DebugString()));

    const int crop_height = grads.dim_size(1);
    const int crop_width = grads.dim_size(2);
    const int crop_depth = grads.dim_size(3);
    OP_REQUIRES(
        context, crop_height > 0 && crop_width > 0 && crop_depth > 0,
        errors::InvalidArgument("grads dimensions must be positive"));
    int num_boxes = 0;
    OP_REQUIRES_OK(
        context, ParseAndCheckBoxSizes(boxes, box_index, &num_boxes));
    OP_REQUIRES(context, image_size.dims() == 1,
                      errors::InvalidArgument("image_size must be 1-D",
                                              image_size.shape().DebugString()));
    OP_REQUIRES(
        context, image_size.dim_size(0) == 5,
        errors::InvalidArgument("image_size must have five elements",
                                image_size.shape().DebugString()));

    auto image_size_vec = image_size.vec<int32>();
    const int batch_size = internal::SubtleMustCopy(image_size_vec(0));
    const int image_height = internal::SubtleMustCopy(image_size_vec(1));
    const int image_width = internal::SubtleMustCopy(image_size_vec(2));
    const int image_depth = internal::SubtleMustCopy(image_size_vec(3));
    const int depth = internal::SubtleMustCopy(image_size_vec(4));
    OP_REQUIRES(
        context, image_height > 0 && image_width > 0 && image_depth > 0,
        errors::InvalidArgument("image dimensions must be positive"));
    OP_REQUIRES(
        context, grads.dim_size(4) == depth,
        errors::InvalidArgument("image_size and grads are incompatible"));

    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({batch_size,
      image_height, image_width, image_depth, depth}), &output));

    auto gradsT = grads.tensor<float, 5>();
    auto boxesT = boxes.tensor<float, 2>();
    auto grads_image = output->tensor<float, 5>();

    grads_image.setZero();

    for (int b = 0; b < num_boxes; ++b) {
      const float y1 = boxesT(b, 0);
      const float x1 = boxesT(b, 1);
      const float z1 = boxesT(b, 2);
      const float y2 = boxesT(b, 3);
      const float x2 = boxesT(b, 4);
      const float z2 = boxesT(b, 5);

      const int32 b_in = box_index.tensor<int32, 1>()(b);

      const float height_scale =
          (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
              : 0;
      const float width_scale =
          (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1)
                           : 0;
      const float depth_scale =
          (crop_depth > 1) ? (z2 - z1) * (image_depth - 1) / (crop_depth - 1)
                       : 0;

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
            if (method_name_ == "trilinear") {
              const int forward_z_index = floorf(in_z);
              const int backward_z_index = ceilf(in_z);
              const float z_lerp = in_z - forward_z_index;

              for (int d = 0; d < depth; ++d) {
                grads_image(b_in, top_y_index, left_x_index, forward_z_index, d) +=
                    static_cast<float>((1 - y_lerp) * (1 - x_lerp) * (1 - z_lerp) * gradsT(b, y, x, z, d));
                grads_image(b_in, top_y_index, left_x_index, backward_z_index, d) +=
                    static_cast<float>((1 - y_lerp) * (1 - x_lerp) * z_lerp * gradsT(b, y, x, z, d));
                grads_image(b_in, top_y_index, right_x_index, forward_z_index, d) +=
                    static_cast<float>((1 - y_lerp) * x_lerp * (1 - z_lerp) * gradsT(b, y, x, z, d));
                grads_image(b_in, top_y_index, right_x_index, backward_z_index, d) +=
                    static_cast<float>((1 - y_lerp) * x_lerp * z_lerp * gradsT(b, y, x, z, d));
                grads_image(b_in, bottom_y_index, left_x_index, forward_z_index, d) +=
                    static_cast<float>(y_lerp * (1 - x_lerp) * (1 - z_lerp) * gradsT(b, y, x, z, d));
                grads_image(b_in, bottom_y_index, left_x_index, backward_z_index, d) +=
                    static_cast<float>(y_lerp * (1 - x_lerp) * z_lerp * gradsT(b, y, x, z, d));
                grads_image(b_in, bottom_y_index, right_x_index, forward_z_index, d) +=
                    static_cast<float>(y_lerp * x_lerp * (1 - z_lerp) * gradsT(b, y, x, z, d));
                grads_image(b_in, bottom_y_index, right_x_index, backward_z_index, d) +=
                    static_cast<float>(y_lerp * x_lerp * z_lerp * gradsT(b, y, x, z, d));
              }
            } else {  // method == "nearest"
              const int closest_x_index = roundf(in_x);
              const int closest_y_index = roundf(in_y);
              const int closest_z_index = roundf(in_z);
              for (int d = 0; d < depth; ++d) {
                grads_image(b_in, closest_y_index, closest_x_index, closest_z_index, d) +=
                    static_cast<float>(gradsT(b, y, x, z, d));
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

REGISTER_KERNEL_BUILDER(Name("CropAndResize3DGradImage").Device(DEVICE_CPU), CropAndResize3DGradImageOp);
