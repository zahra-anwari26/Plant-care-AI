import express from "express";
import { diagnosePlant } from "./../Controllers/diagnosisController.js";;
import upload from "./../middlewares/uploadMiddleware.js";
const router = express.Router();
router.post("/", upload.single("image"), diagnosePlant);
export default router;