const Staff = require("../models/staff.js");
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const secretKey = process.env.JWT_KEY || 'secretkey';

const getStaffs = async (req, res) => {
  try {
    const staff = await Staff.find({});
    res.status(200).json(staff);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

const getStaff = async (req, res) => {
  try {
    const { id } = req.params;
    const staff = await Staff.findById(id);
    res.status(200).json(staff);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

const createStaff = async (req, res) => {
  try {
    // convert password into hash
    req.body.password = bcrypt.hashSync(req.body.password, 10);
    const staff = await Staff.create(req.body);
    res.status(200).json(staff);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

const deleteStaff = async (req, res) => {
  try {
    const { id } = req.params;

    const staff = await Staff.findByIdAndDelete(id);

    if (!staff) {
      return res.status(404).json({ message: "Staff not found" });
    }

    res.status(200).json({ message: "Staff deleted successfully" });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

const signinStaff = async (req, res) => {
  try {
    const { email, password } = req.body;
    // Find staff by email
    const staff = await Staff.findOne({ email });
    if (!staff) {
      return res.status(401).json({ message: "Invalid email" });
    }
    const isMatch = bcrypt.compareSync(password, staff.password);
    if (!isMatch) {
      return res.status(401).json({ message: "Invalid password" });
    }

    
    // Generate JWT
    const token = jwt.sign(
      { email: staff.email, id: staff._id, name: staff.name, department_name: staff.department_name },
      secretKey,
      { expiresIn: '1h' }
    );

    var position = null;
    if (staff.department_name === 'Fab A' || staff.department_name === 'Fab B' || staff.department_name === 'Fab C') {
      position = 'Fab';
    } else {
      position = 'Lab';
    }

    return res.status(200).json({ token, position, name:staff.name, department_name:staff.department_name });
  } catch (err) {
    console.error("Error signing in staff", err);
    // console.error(err);
    return res.status(500).json({ message: err.message });
  }
};

module.exports = {
    getStaffs,
    getStaff,
    createStaff,
    deleteStaff,
    signinStaff,
};