const mongoose = require("mongoose");
const request = require("supertest");
const express = require("express");
const jwt = require("jsonwebtoken");
const orderRoute = require("../src/routes/order");
const staffRoute = require("../src/routes/staff");
const Staff = require("../src/models/staff");
const PDFDocument = require("pdfkit");
const fs = require("fs");
const path = require("path");

const secretKey = "secretkey";

let app;
let token;

beforeAll(async () => {
  const uri = "mongodb://localhost:27017/test";
  await mongoose.connect(uri);

  app = express();
  app.use(express.json());
  app.use(express.urlencoded({ extended: false }));
  app.use(require("express-fileupload")());
  app.use("/api/orders", orderRoute);
  app.use("/api/staffs", staffRoute);

  // Create a test user with a valid department name
  const staff = await Staff.create({
    email: "seanmamasde@example.com",
    name: "seanmamasde",
    password: "seanmamasdes_password",
    department_name: "Fab A", // Valid department name
  });

  // Generate a valid JWT token
  token = jwt.sign(
    {
      email: staff.email,
      id: staff._id,
      department_name: staff.department_name,
      name: staff.name,
    },
    secretKey,
    { expiresIn: "1h" }
  );
});

afterAll(async () => {
  await mongoose.connection.dropDatabase();
  await mongoose.connection.close();
});

describe("JWT Authentication", () => {
  it("should create an order with a valid token", async () => {
    // Create a simple PDF file
    const pdfPath = path.join(__dirname, "authTest.pdf");
    const doc = new PDFDocument();
    doc.pipe(fs.createWriteStream(pdfPath));
    doc.text("This is a test PDF file for auth.");
    doc.end();

    const response = await request(app)
      .post("/api/orders")
      .set("Authorization", `Bearer ${token}`)
      .field("title", "an order with a valid token")
      .field("description", "Order created with a valid token")
      .field("creator", "test_creator") // Adding required creator field
      .field("priority", 1)
      .field("lab_name", "化學實驗室")
      .field("fab_name", "Fab A")
      .attach("file", pdfPath);

    expect(response.status).toBe(200);
    expect(response.body).toHaveProperty("_id");
    expect(response.body.title).toBe("an order with a valid token");

    // Clean up
    fs.unlinkSync(pdfPath);
  });

  it("should fail to create an order without a token", async () => {
    const response = await request(app)
      .post("/api/orders")
      .field("title", "an order failed since it's without token")
      .field("description", "Order created without a token")
      .field("priority", 1)
      .field("lab_name", "化學實驗室")
      .field("creator", "test_creator")
      .field("fab_name", "Fab A");

    expect(response.status).toBe(401);
    expect(response.body.message).toBe("Token missing");
  });

  it("should fail to create an order with an invalid token", async () => {
    const response = await request(app)
      .post("/api/orders")
      .set("Authorization", "Bearer invalidtoken")
      .field("title", "an order failed cus token invalid")
      .field("description", "Order created with an invalid token")
      .field("creator", "test_creator")
      .field("priority", 1)
      .field("lab_name", "化學實驗室")
      .field("fab_name", "Fab A");

    expect(response.status).toBe(403);
    expect(response.body.message).toBe("Token invalid");
  });
});
