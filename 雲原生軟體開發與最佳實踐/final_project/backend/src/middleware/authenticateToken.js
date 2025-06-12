const jwt = require("jsonwebtoken");
const secretKey = process.env.JWT_KEY || "secretkey";

const authenticateToken = (req, res, next) => {
  const authHeader = req.headers["authorization"];
  const token = authHeader && authHeader.split(" ")[1];

  if (!token) {
    return res.status(401).json({ message: "Token missing" });
  }

  jwt.verify(token, secretKey, (err, user) => {
    if (err) {
      return res.status(403).json({ message: "Token invalid" });
    }
    req.user = user;
    next();
  });
};

module.exports = authenticateToken;
