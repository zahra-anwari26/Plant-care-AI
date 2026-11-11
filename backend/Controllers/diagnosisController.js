import axios from "axios";

export const diagnosePlant = async (req, res) => {
    try {
      
        if (!req.file) {
            return res.status(400).json({
                success: false,
                message: "Please upload a plant image."
            });
        }

     
        const formData = new FormData();

        const imageBlob = new Blob(
            [req.file.buffer],
            { type: req.file.mimetype }
        );

        formData.append(
            "image",
            imageBlob,
            req.file.originalname
        );

        // ارسال عکس به سرویس هوش مصنوعی Python
        const response = await axios.post(
            `${process.env.PYTHON_API_URL}/predict`,
            formData
        );

     
        return res.status(200).json({
            success: true,
            result: response.data
        });

    } catch (error) {
        console.error("Diagnosis Error:", error.message);
        return res.status(500).json({
            success: false,
            message: "Plant diagnosis failed."
        });
    }
};