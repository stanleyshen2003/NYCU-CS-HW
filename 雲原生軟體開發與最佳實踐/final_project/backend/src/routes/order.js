const express = require("express");
const mongoose = require("mongoose");
const {
  getOrders,
  createOrder,
  updateOrder,
  markOrderAsCompleted,
  getFileStream,
} = require("../services/order.js");
const authenticateToken = require("../middleware/authenticateToken");
const router = express.Router();

router.get("/files/:fileId", authenticateToken, async (req, res) => {
  try {
    const fileId = new mongoose.Types.ObjectId(req.params.fileId);
    console.log(`Fetching file with ID: ${fileId}`);

    const downloadStream = await getFileStream(fileId);

    res.setHeader("Content-Type", "application/pdf");

    downloadStream.on("data", (chunk) => {
      res.write(chunk);
    });

    downloadStream.on("error", (err) => {
      console.error("Download stream error:", err);
      res.sendStatus(404);
    });

    downloadStream.on("end", () => {
      res.end();
    });
  } catch (error) {
    console.error("Error fetching file:", error);
    res.status(500).send("Error fetching file: " + error.message);
  }
});

// get all orders
router.get("/", authenticateToken, async (req, res) => {
  try {
    const filters = req.query;
    const orders = await getOrders(filters, req.user);
    res.status(200).json(orders);
  } catch (error) {
    console.error("Error fetching orders:", error);
    res.status(500).json({ message: error.message });
  }
});

// create order
router.post("/", authenticateToken, async (req, res) => {
  try {
    const order = await createOrder(req.body, req.user, req.files);
    res.status(200).json(order);
  } catch (error) {
    console.error("Error creating order:", error);
    res.status(500).json({ message: error.message });
  }
});

// update order
router.put("/", authenticateToken, async (req, res) => {
  try {
    const orderId = req.body._id; // Ensure _id is included in the request body
    const updatedOrder = await updateOrder(
      orderId,
      req.body,
      req.files,
      req.user
    );
    res.status(200).json(updatedOrder);
  } catch (error) {
    console.error("Error updating order:", error);
    res.status(500).json({ message: error.message });
  }
});

// mark order as completed
router.put("/:orderId", authenticateToken, async (req, res) => {
  try {
    const { orderId } = req.params;
    const updatedOrder = await markOrderAsCompleted(orderId, req.user);
    res.status(200).json(updatedOrder);
  } catch (error) {
    console.error("Error marking order as completed:", error);
    res.status(500).json({ message: error.message });
  }
});

module.exports = router;
