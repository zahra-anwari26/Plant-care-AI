
import express from "express";
import cors from "cors";
import helmet from "helmet";
import dotenv from "dotenv";
import diagnosisRoutes from "./Routes/diagnosisRoutes.js";
dotenv.config();
const app = express();
app.use(helmet());
app.use(cors({
    origin: "http://localhost:5173", 
}));

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use("/api/diagnosis", diagnosisRoutes);


app.get("/", (req, res) => {
    res.json({
        success: true,
        message: "AgriVision AI Backend is running!"
    });
});


const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
