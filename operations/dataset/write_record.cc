#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <string>

using tensorflow::OpKernel;
using tensorflow::OpKernelContext;
using tensorflow::OpKernelConstruction;
using tensorflow::Tensor;
using tensorflow::shape_inference::NoOutputs;
using tensorflow::DEVICE_CPU;
using tensorflow::io::RecordWriterOptions;
using tensorflow::StringPiece;
using tensorflow::tstring;
using u8 = tensorflow::uint8;

class WriteRecordOp : public OpKernel {
public:
  explicit WriteRecordOp(OpKernelConstruction *context) : OpKernel(context) {}
    void Compute(OpKernelContext *ctx) override {
      const std::string& filename = ctx->input(0).scalar<tstring>()();
      const StringPiece& record = ctx->input(1).scalar<tstring>()();
      const bool& append = ctx->input(2).scalar<bool>()();

      std::unique_ptr<tensorflow::WritableFile> file;
      if(append){
        tensorflow::Env::Default()->NewAppendableFile(filename, &file);
      }else{
        tensorflow::Env::Default()->NewWritableFile(filename, &file);
      }
      const RecordWriterOptions& options = RecordWriterOptions::CreateRecordWriterOptions("");
      auto writer = absl::make_unique<tensorflow::io::RecordWriter>(file.get(), options);

      if(file == nullptr && writer == nullptr){
        return;
      }

      writer->WriteRecord(StringPiece(record));
      writer->Close();
      file->Close();
    }
};

REGISTER_OP("WriteRecord")
    .Input("filename: string")
    .Input("record: string")
    .Input("append: bool")
    .SetIsStateful()
    .SetShapeFn(NoOutputs);

REGISTER_KERNEL_BUILDER(Name("WriteRecord").Device(DEVICE_CPU), WriteRecordOp);