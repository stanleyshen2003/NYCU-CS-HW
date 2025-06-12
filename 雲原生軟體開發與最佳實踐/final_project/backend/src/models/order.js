const mongoose = require("mongoose");

const FileSchema = new mongoose.Schema({
  length: { type: Number, required: true },
  chunkSize: { type: Number, required: true },
  uploadDate: { type: Date, required: true },
  filename: { type: String, required: true },
  md5: { type: String, required: true },
  contentType: { type: String, required: true },
});

const OrderSchema = mongoose.Schema(
  {
    title: { type: String, required: true, index: true },
    description: { type: String, required: true, index: true },
    creator: { type: String, required: true, index: true },
    fab_name: {
      type: String,
      required: true,
      index: true,
      validate: {
        validator: function (value) {
          return ["Fab A", "Fab B", "Fab C"].includes(value);
        },
      },
      message: (props) =>
        `${props.value} is not a valid fab name. Fab name must be one of: Fab A, Fab B, Fab C`,
    },
    lab_name: {
      type: String,
      required: true,
      index: true,
      validate: {
        validator: function (value) {
          return ["化學實驗室", "表面分析實驗室", "成分分析實驗室"].includes(
            value
          );
        },
      },
      message: (props) =>
        `${props.value} is not a valid lab name. Lab name must be one of: 化學實驗室, 表面分析實驗室, 成分分析實驗室`,
    },
    priority: { type: Number, required: true },
    is_completed: {
      type: Boolean,
      required: true,
      default: false,
      index: true,
    },
    attachments: [
      {
        file: { type: mongoose.Schema.Types.ObjectId, ref: "uploads.files" },
      },
    ],
  },
  {
    timestamps: true,
  }
);

const File = mongoose.model("uploads.files", FileSchema);
const Order = mongoose.model("Order_collection", OrderSchema);
module.exports = { Order, File };
