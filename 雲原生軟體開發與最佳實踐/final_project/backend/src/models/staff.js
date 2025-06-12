const mongoose = require("mongoose");

const Staff_Schema = mongoose.Schema(
  {
    email: { type: String, required: true },
    name: {type: String, required: true},
    password: { type: String, required: true },
    department_name: {
      type: String,
      required: true,
      validate: {
        validator: function (value) {
          return [
            "Fab A",
            "Fab B",
            "Fab C",
            "化學實驗室",
            "表面分析實驗室",
            "成分分析實驗室",
          ].includes(value);
        },
        message: (props) =>
          `${props.value} is not a valid department name. Department name must be one of: Fab A, Fab B, Fab C, 化學實驗室, 表面分析實驗室, 成分分析實驗室`,
      },
    },
  },
  {
    timestamps: true,
  }
);

const Staff = mongoose.model("Staff", Staff_Schema);
module.exports = Staff;
