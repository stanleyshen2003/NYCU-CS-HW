const express = require("express");
const mongoose = require("mongoose");
const orderRoute = require("./routes/order.js");
const staffRoute = require("./routes/staff.js");
const fileUpload = require("express-fileupload");
const dotenv = require("dotenv");
const cors = require("cors");

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(
  fileUpload({
    limits: { fileSize: 50 * 1024 * 1024 }, // 50MB limit
    useTempFiles: false,
    tempFileDir: "/tmp/",
  })
);

app.use("/api/orders", orderRoute);
app.use("/api/staffs", staffRoute);

const mongoURI = process.env.MONGO_URI;
const port = process.env.BACKEND_PORT || 8888;

mongoose
  .connect(mongoURI)
  .then(() => {
    console.log("Connected to database!");
    app.listen(port, () => {
      console.log(`Server is running on port ${port}`);
    });
  })
  .catch(() => {
    console.log("Connection failed!");
  });
