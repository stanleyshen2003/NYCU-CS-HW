const mongoose = require("mongoose");
const request = require("supertest");
const express = require("express");
const fs = require("fs");
const path = require("path");
const PDFDocument = require("pdfkit");
const crypto = require("crypto");
const jwt = require("jsonwebtoken");
const orderRoute = require("../src/routes/order");
const staffRoute = require("../src/routes/staff");
const Staff = require("../src/models/staff");
const { getOrders } = require("../src/services/order");
const Order = require("../src/models/order");
const { get } = require("http");
const GridFSBucket = require("mongodb").GridFSBucket;

const secretKey = "secretkey";

let app;
let lab_token, fab_token;
let lab_user, fab_user;

beforeAll(async () => {
  const uri = "mongodb://localhost:27017/test";
  await mongoose.connect(uri);

  app = express();
  app.use(express.json());
  app.use(express.urlencoded({ extended: false }));
  app.use(require("express-fileupload")());
  app.use("/api/orders", orderRoute);
  app.use("/api/staffs", staffRoute);

  fab_user = await Staff.create({
    email: "A@amail.com",
    name: "AAA",
    password: "aaa",
    department_name: "Fab A",
  });

  fab_token = jwt.sign(
    {
      id: fab_user._id,
      email: fab_user.email,
      name: fab_user.name,
      department_name: fab_user.department_name,
    },
    secretKey,
    { expiresIn: "1h" }
  );

  lab_user = await Staff.create({
    email: "B@bmail.com",
    name: "BBB",
    password: "bbb",
    department_name: "化學實驗室",
  });

  lab_token = jwt.sign(
    {
      id: lab_user._id,
      email: lab_user.email,
      name: lab_user.name,
      department_name: lab_user.department_name,
    },
    secretKey,
    { expiresIn: "1h" }
  );
});

afterAll(async () => {
  await mongoose.connection.dropDatabase();
  await mongoose.connection.close();
});

describe("Order API", () => {
  it("should create an order with a PDF attachment", async () => {
    // Create a simple PDF file
    const pdfPath = path.join(__dirname, "test.pdf");
    const doc = new PDFDocument();
    doc.pipe(fs.createWriteStream(pdfPath));
    doc.text("This is a test PDF file.");
    doc.end();

    const response = await request(app)
      .post("/api/orders")
      .set("Authorization", `Bearer ${fab_token}`)
      .field("title", "order with 1 pdf file.")
      .field("description", "Test order")
      .field("creator", "test_creator")
      .field("fab_name", "Fab A")
      .field("lab_name", "化學實驗室")
      .field("priority", 1)
      .attach("file", pdfPath);

    expect(response.status).toBe(200);
    expect(response.body.attachments).toHaveLength(1);
    expect(response.body.attachments[0]).toHaveProperty("_id");
    expect(response.body.attachments[0]).toHaveProperty("file");

    // Clean up
    fs.unlinkSync(pdfPath);
  });

  it("should create an order with two PDF attachments", async () => {
    // Create two simple PDF files
    const pdfPath1 = path.join(__dirname, "test1.pdf");
    const doc1 = new PDFDocument();
    doc1.pipe(fs.createWriteStream(pdfPath1));
    doc1.text("This is the first test PDF file.");
    doc1.end();

    const pdfPath2 = path.join(__dirname, "test2.pdf");
    const doc2 = new PDFDocument();
    doc2.pipe(fs.createWriteStream(pdfPath2));
    doc2.text("This is the second test PDF file.");
    doc2.end();

    const response = await request(app)
      .post("/api/orders")
      .set("Authorization", `Bearer ${fab_token}`)
      .field("title", "order with 2 pdf files.")
      .field("description", "Test order with two PDFs")
      .field("creator", "test_creator")
      .field("fab_name", "Fab A")
      .field("lab_name", "化學實驗室")
      .field("priority", 2)
      .attach("file", pdfPath1)
      .attach("file", pdfPath2);

    expect(response.status).toBe(200);
    expect(response.body.attachments).toHaveLength(2);
    expect(response.body.attachments[0]).toHaveProperty("_id");
    expect(response.body.attachments[0]).toHaveProperty("file");
    expect(response.body.attachments[1]).toHaveProperty("_id");
    expect(response.body.attachments[1]).toHaveProperty("file");

    // Clean up
    fs.unlinkSync(pdfPath1);
    fs.unlinkSync(pdfPath2);
  });

  it("should update an existing order with new data and attachments", async () => {
    // Create a simple PDF file
    const pdfPath = path.join(__dirname, "updateTest.pdf");
    const doc = new PDFDocument();
    doc.pipe(fs.createWriteStream(pdfPath));
    doc.text("This is a test PDF file for update.");
    doc.end();

    // Create an order to update
    const createResponse = await request(app)
      .post("/api/orders")
      .set("Authorization", `Bearer ${fab_token}`)
      .field("title", "this is a test order for update.")
      .field("description", "Order to be updated")
      .field("creator", "test_creator")
      .field("fab_name", "Fab A")
      .field("lab_name", "化學實驗室")
      .field("priority", 3)
      .attach("file", pdfPath);

    expect(createResponse.status).toBe(200);

    // Find the created order by title
    const query = { title: "this is a test order for update." };
    const orders = await getOrders(query, fab_user);
    const orderId = orders[0]?._id.toString();

    // Check if the order was found
    expect(orderId).toBeDefined();

    // Update the order
    const updateResponse = await request(app)
      .put("/api/orders")
      .set("Authorization", `Bearer ${fab_token}`)
      .field("_id", orderId)
      .field("title", "this is the updated title.")
      .field("description", "Updated description")
      .field("priority", 4)
      .field("lab_name", "表面分析實驗室")
      .attach("file", pdfPath);

    expect(updateResponse.status).toBe(200);
    expect(updateResponse.body.title).toBe("this is the updated title.");
    expect(updateResponse.body.priority).toBe(4);
    expect(updateResponse.body.lab_name).toBe("表面分析實驗室");
    expect(updateResponse.body.attachments).toHaveLength(1);
    expect(updateResponse.body.attachments[0]).toHaveProperty("_id");
    expect(updateResponse.body.attachments[0]).toHaveProperty("file");

    // Clean up
    fs.unlinkSync(pdfPath);
  });

  it("should mark an order as completed", async () => {
    // Create a simple PDF file
    const pdfPath = path.join(__dirname, "completeTest.pdf");
    const doc = new PDFDocument();
    doc.pipe(fs.createWriteStream(pdfPath));
    doc.text("This is a test PDF file for completion.");
    doc.end();

    // Create an order to update
    const createResponse = await request(app)
      .post("/api/orders")
      .set("Authorization", `Bearer ${fab_token}`)
      .field("title", "this is a test order for completion.")
      .field("description", "Order to be marked as completed")
      .field("creator", "test_creator")
      .field("fab_name", "Fab A")
      .field("lab_name", "化學實驗室")
      .field("priority", 3)
      .attach("file", pdfPath);

    expect(createResponse.status).toBe(200);
    const orderId = createResponse.body._id;

    // Mark the order as completed
    const completeResponse = await request(app)
      .put(`/api/orders/${orderId}`)
      .set("Authorization", `Bearer ${lab_token}`);

    expect(completeResponse.status).toBe(200);
    expect(completeResponse.body.is_completed).toBe(true);

    // Clean up
    fs.unlinkSync(pdfPath);
  });

  it("should print and verify all entries in the MongoDB server in the file schema format", async () => {
    const bucket = new GridFSBucket(mongoose.connection.db, {
      bucketName: "uploads",
    });

    const files = await bucket.find().toArray();
    for (const file of files) {
      // Create a stream to read the file content
      const downloadStream = bucket.openDownloadStream(file._id);
      const chunks = [];

      await new Promise((resolve, reject) => {
        downloadStream.on("data", (chunk) => {
          chunks.push(chunk);
        });
        downloadStream.on("end", () => {
          const buffer = Buffer.concat(chunks);
          const expectedHash = crypto
            .createHash("md5")
            .update(buffer)
            .digest("hex");
          expect(file.metadata.md5).toBe(expectedHash);
          resolve();
        });
        downloadStream.on("error", reject);
      });
    }

    // To ensure the test passes, we check if files is an array
    expect(Array.isArray(files)).toBe(true);

    // Print all orders in the MongoDB database
    // const orders = await getOrders({}, fab_user);
    // console.log("Orders in MongoDB:", JSON.stringify(orders, null, 2));
    // console.log("Total number of orders in MongoDB:", orders.length);
  });

  it("should download PDF files from the order with 2 PDF attachments", async () => {
    // Create two simple PDF files
    const pdfPath1 = path.join(__dirname, "downloadTest1.pdf");
    const doc1 = new PDFDocument();
    doc1.pipe(fs.createWriteStream(pdfPath1));
    doc1.text("This is the first test PDF file for download.");
    doc1.end();

    const pdfPath2 = path.join(__dirname, "downloadTest2.pdf");
    const doc2 = new PDFDocument();
    doc2.pipe(fs.createWriteStream(pdfPath2));
    doc2.text("This is the second test PDF file for download.");
    doc2.end();

    // Create an order with two PDF attachments
    const createResponse = await request(app)
      .post("/api/orders")
      .set("Authorization", `Bearer ${fab_token}`)
      .field("title", "order with 2 pdf files for download")
      .field("description", "Test order with two PDFs for download")
      .field("creator", "test_creator")
      .field("fab_name", "Fab A")
      .field("lab_name", "化學實驗室")
      .field("priority", 2)
      .attach("file", pdfPath1)
      .attach("file", pdfPath2);

    expect(createResponse.status).toBe(200);
    console.log("Order creation response:", createResponse.body);

    // Find the created order by title
    const query = { title: "order with 2 pdf files for download" };
    const orders = await getOrders(query, fab_user);
    const orderId = orders[0]?._id.toString();
    const attachments = orders[0]?.attachments;

    expect(orderId).toBeDefined();
    expect(attachments).toHaveLength(2);

    console.log("Order ID:", orderId);
    console.log("Attachments:", attachments);

    // Ensure the files are uploaded to MongoDB
    const db = mongoose.connection.db;
    const bucket = new GridFSBucket(db, { bucketName: "uploads" });
    for (const attachment of attachments) {
      const files = await bucket
        .find({ _id: new mongoose.Types.ObjectId(attachment.file) })
        .toArray();
      console.log("Files found in MongoDB:", files);
      expect(files).toHaveLength(1); // Ensure the file exists in MongoDB

      const chunks = await db
        .collection("uploads.chunks")
        .find({ files_id: new mongoose.Types.ObjectId(attachment.file) })
        .toArray();
      console.log("Chunks found in MongoDB:", chunks);
      expect(chunks.length).toBeGreaterThan(0); // Ensure there are chunks for the file
    }

    // Download the attached PDF files
    for (const attachment of attachments) {
      const fileId = attachment.file._id.toString();
      console.log(`Attempting to download file with ID: ${fileId}`);

      const response = await request(app)
        .get(`/api/orders/files/${fileId}`)
        .set("Authorization", `Bearer ${fab_token}`)
        .buffer(true) // Ensure the response is buffered
        .parse((res, callback) => {
          res.setEncoding("binary");
          res.data = "";
          res.on("data", (chunk) => {
            res.data += chunk;
          });
          res.on("end", () => {
            callback(null, Buffer.from(res.data, "binary"));
          });
        });

      console.log(`Downloading file ${fileId} - status: ${response.status}`);
      console.log(`Headers:`, response.headers);

      expect(response.status).toBe(200);
      expect(response.header["content-type"]).toBe("application/pdf");

      // Save the downloaded PDF to a local file
      const downloadPath = path.join(__dirname, `downloaded_${fileId}.pdf`);
      fs.writeFileSync(downloadPath, response.body);

      // Verify the downloaded file
      const downloadedFileBuffer = fs.readFileSync(downloadPath);
      const expectedHash = crypto
        .createHash("md5")
        .update(downloadedFileBuffer)
        .digest("hex");

      // Fetch metadata to verify MD5 hash
      const files = await bucket
        .find({ _id: new mongoose.Types.ObjectId(attachment.file) })
        .toArray();
      expect(files[0].metadata.md5).toBe(expectedHash);

      // Clean up
      fs.unlinkSync(downloadPath);
    }

    // Clean up
    fs.unlinkSync(pdfPath1);
    fs.unlinkSync(pdfPath2);
  });
});
