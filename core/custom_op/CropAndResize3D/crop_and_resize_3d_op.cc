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

namespace {

// Sets output[0] to shape [batch_dim,height,width,channel_dim], where
// height and width come from the size_tensor.
Status SetOutputToSizedImage(::tensorflow::shape_inference::InferenceContext* c,
                              ::tensorflow::shape_inference::DimensionHandle batch_dim,
                             int size_input_idx,
                             ::tensorflow::shape_inference::DimensionHandle channel_dim) {
  // Verify shape of size input.
  ::tensorflow::shape_inference::ShapeHandle size;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(size_input_idx), 1, &size));
  ::tensorflow::shape_inference::DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(size, 0), 3, &unused));

  // Get size values from the size tensor.
  const Tensor* size_tensor = c->input_tensor(size_input_idx);
  ::tensorflow::shape_inference::DimensionHandle width;
  ::tensorflow::shape_inference::DimensionHandle height;
  ::tensorflow::shape_inference::DimensionHandle depth;
  if (size_tensor == nullptr) {
    width = c->UnknownDim();
    height = c->UnknownDim();
    depth = c->UnknownDim();
  } else {
    // TODO(petewarden) - Remove once we have constant evaluation in C++ only.
    if (size_tensor->dtype() != DT_INT32) {
      return errors::InvalidArgument(
          "Bad size input type for SetOutputToSizedImage: Expected DT_INT32 "
          "but got ",
          DataTypeString(size_tensor->dtype()), " for input #", size_input_idx,
          " in ", c->DebugString());
    }
    auto vec = size_tensor->vec<int32>();
    height = c->MakeDim(vec(0));
    width = c->MakeDim(vec(1));
    depth = c->MakeDim(vec(2));
  }
  c->set_output(0, c->MakeShape({batch_dim, height, width, depth, channel_dim}));
  return Status::OK();
}

}

REGISTER_OP("CropAndResize3D")
    .Input("image: T")
    .Input("boxes: float")
    .Input("box_index: int32")
    .Input("crop_size: int32")
    .Output("crops: float")
    .Attr("T: {uint8, uint16, int8, int16, int32, int64, half, float, double}")
    .Attr("method_name: {'trilinear', 'nearest'} = 'trilinear'")
    .Attr("extrapolation_value: float = 0")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      // Get inputs and validate ranks.
      ::tensorflow::shape_inference::ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &input));
      ::tensorflow::shape_inference::ShapeHandle boxes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &boxes));
      ::tensorflow::shape_inference::ShapeHandle box_ind;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &box_ind));

      // boxes[0] and box_ind[0] are both num_boxes.
      ::tensorflow::shape_inference::DimensionHandle num_boxes_dim;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(boxes, 0), c->Dim(box_ind, 0), &num_boxes_dim));

      // boxes.dim(1) is 6.
      ::tensorflow::shape_inference::DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(boxes, 1), 6, &unused));

      return SetOutputToSizedImage(c, num_boxes_dim, 3 /* size_input_idx */,
                                   c->Dim(input, 4));
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

class CropAndResize3DOp : public OpKernel {
public:
  explicit CropAndResize3DOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("method_name", &method_name_));
    OP_REQUIRES(context, method_name_ == "trilinear" || method_name_ == "nearest",
                errors::InvalidArgument(
                    "method must be 'trilinear' or 'nearest'", method_name_));
    OP_REQUIRES_OK(context, context->GetAttr("extrapolation_value",
                                             &extrapolation_value_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& image = context-> input(0);
    const Tensor& boxes = context-> input(1);
    const Tensor& box_index = context-> input(2);
    const Tensor& crop_size = context-> input(3);

    OP_REQUIRES(context, image.dims() == 5,
                      errors::InvalidArgument("input image must be 5-D",
                                              image.shape().DebugString()));

    const int batch_size = image.dim_size(0);
    const int image_height = image.dim_size(1);
    const int image_width = image.dim_size(2);
    const int image_depth = image.dim_size(3);
    const int depth = image.dim_size(4);
    OP_REQUIRES(
        context, image_height > 0 && image_width > 0 && image_depth > 0,
        errors::InvalidArgument("image dimensions must be positive"));
    int num_boxes = 0;
    OP_REQUIRES_OK(
        context, ParseAndCheckBoxSizes(boxes, box_index, &num_boxes));
    OP_REQUIRES(context, crop_size.dims() == 1,
                      errors::InvalidArgument("crop_size must be 1-D",
                                              crop_size.shape().DebugString()));
    OP_REQUIRES(
        context, crop_size.dim_size(0) == 3,
        errors::InvalidArgument("crop_size must have three elements",
                                crop_size.shape().DebugString()));

    auto crop_size_vec = crop_size.vec<int32>();
    const int crop_height = internal::SubtleMustCopy(crop_size_vec(0));
    const int crop_width = internal::SubtleMustCopy(crop_size_vec(1));
    const int crop_depth = internal::SubtleMustCopy(crop_size_vec(2));
    OP_REQUIRES(
        context, crop_height > 0 && crop_width > 0 && crop_depth > 0,
        errors::InvalidArgument("crop dimensions must be positive"));

    Tensor* cropped = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({num_boxes,
      crop_height, crop_width, crop_depth, depth}), &cropped));

    auto boxesT = boxes.tensor<float, 2>();
    auto imageT = image.tensor<float, 5>();
    auto croppedT = cropped->tensor<float, 5>();

    for (int b = 0; b < num_boxes; ++b) {
      const float y1 = boxesT(b, 0);
      const float x1 = boxesT(b, 1);
      const float z1 = boxesT(b, 2);
      const float y2 = boxesT(b, 3);
      const float x2 = boxesT(b, 4);
      const float z2 = boxesT(b, 5);

      const int32 b_in = box_index.tensor<int32, 1>()(b);

      const float height_scale =
          (crop_height > 1)
              ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
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
          for (int x = 0; x < crop_width; ++x) {
            for (int z = 0; z < crop_depth; z++) {
              for (int d = 0; d < depth; ++d) {
                croppedT(b, y, x, z, d) = extrapolation_value_;
              }
            }
          }
          continue;
        }
        if (method_name_ == "trilinear") {
          const int top_y_index = floorf(in_y);
          const int bottom_y_index = ceilf(in_y);
          const float y_lerp = in_y - top_y_index;

          for (int x = 0; x < crop_width; ++x) {
            const float in_x = (crop_width > 1)
                                   ? x1 * (image_width - 1) + x * width_scale
                                   : 0.5 * (x1 + x2) * (image_width - 1);
            if (in_x < 0 || in_x > image_width - 1) {
              for (int z = 0; z < crop_depth; z++) {
                for (int d = 0; d < depth; ++d) {
                  croppedT(b, y, x, z, d) = extrapolation_value_;
                }
              }
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
                for (int d = 0; d < depth; ++d) {
                 croppedT(b, y, x, z, d) = extrapolation_value_;
                }
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
                const float top_left = top_left_forward + (top_left_backward - top_left_forward)*z_lerp;
                const float top_right = top_right_forward + (top_right_backward - top_right_forward)*z_lerp;
                const float bottom_left = bottom_left_forward + (bottom_left_backward - bottom_left_forward)*z_lerp;
                const float bottom_right = bottom_right_forward + (bottom_right_backward - bottom_right_forward)*z_lerp;
                const float top = top_left + (top_right - top_left) * x_lerp;
                const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
                croppedT(b, y, x, z, d) = top + (bottom - top) * y_lerp;
              }
            }
          }
        } else {  // method == "nearest"
          for (int x = 0; x < crop_width; ++x) {
            const float in_x = (crop_width > 1)
                                   ? x1 * (image_width - 1) + x * width_scale
                                   : 0.5 * (x1 + x2) * (image_width - 1);
            if (in_x < 0 || in_x > image_width - 1) {
              for (int z = 0; z < crop_depth; ++z) {
                for (int d = 0; d < depth; ++d) {
                  croppedT(b, y, x, z, d) = extrapolation_value_;
                }
              }
              continue;
            }
            for (int z = 0; z < crop_width; ++z) {
              const float in_z = (crop_depth > 1)
                                     ? z1 * (image_depth - 1) + z * depth_scale
                                     : 0.5 * (z1 + z2) * (image_depth - 1);
              if (in_z < 0 || in_z > image_depth - 1) {
                for (int d = 0; d < depth; ++d) {
                  croppedT(b, y, x, z, d) = extrapolation_value_;
                }
                continue;
              }
              const int closest_x_index = roundf(in_x);
              const int closest_y_index = roundf(in_y);
              const int closest_z_index = roundf(in_z);
              for (int d = 0; d < depth; ++d) {
                croppedT(b, y, x, z, d) = static_cast<float>(
                    imageT(b_in, closest_y_index, closest_x_index, closest_z_index, d));
              }
            }
          }
        }
      }
    }
  }
private:
  string method_name_ ;
  float extrapolation_value_ ;
};

REGISTER_KERNEL_BUILDER(Name("CropAndResize3D").Device(DEVICE_CPU), CropAndResize3DOp);
