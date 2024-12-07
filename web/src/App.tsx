// src/App.tsx
import { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";

const App: React.FC = () => {
  const webcamRef = useRef<Webcam>(null);
  const [emotion, setEmotion] = useState<string | null>(null);

  const [detectionOptions, setDetectionOptions] = useState({
    faceDetection: true,
    objectDetection: false,
    brightness: 50,
    saturation: 50,
    imageSize: "medium",
  });

  const captureAndDetectEmotion = async () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        const base64Image = imageSrc.split(",")[1]; // Get base64 part of the image

        try {
          const response = await fetch(
            "http://127.0.0.1:8000/api/detect-emotion/",
            {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({ image: base64Image }),
            }
          );

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }

          const data = await response.json();
          setEmotion(data.emotion);
        } catch (error) {
          console.error("Error fetching emotion detection:", error);
        }
      }
    }
  };

  useEffect(() => {
    const intervalId = setInterval(() => {
      captureAndDetectEmotion();
    }, 1000); // Capture every 1 second

    return () => clearInterval(intervalId); // Cleanup on unmount
  }, []);

  const handleToggle = (option: keyof typeof detectionOptions) => {
    setDetectionOptions((prev) => ({
      ...prev,
      [option]: !prev[option],
    }));
  };

  return (
    <div className="flex flex-col md:flex-row h-screen">
      <div className="flex-1 flex justify-center items-center bg-gray-200">
        <Webcam audio={false} screenshotFormat="image/jpeg" ref={webcamRef} />

        {emotion && (
          <img
            src={emotion}
            alt="Detected Faces"
            className="w-full h-full object-cover"
          />
        )}
      </div>

      <div className="w-full md:w-64 p-4 bg-white border-l md:border-l-2 border-gray-300">
        <h2 className="text-lg font-semibold mb-4">Face Detection Options</h2>
        <div className="mb-4">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={detectionOptions.faceDetection}
              onChange={() => handleToggle("faceDetection")}
              className="toggle toggle-success"
            />
            Face Detection:{" "}
            {detectionOptions.faceDetection ? "Enabled" : "Disabled"}
          </label>
        </div>
        <div className="mb-4">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={detectionOptions.objectDetection}
              onChange={() => handleToggle("objectDetection")}
              className="toggle toggle-success"
            />
            Object Detection:{" "}
            {detectionOptions.objectDetection ? "Enabled" : "Disabled"}
          </label>
        </div>
        <div className="mb-4">
          <label>Brightness:</label>
          <input
            type="range"
            min="0"
            max="100"
            value={detectionOptions.brightness}
            onChange={(e) =>
              setDetectionOptions({
                ...detectionOptions,
                brightness: Number(e.target.value),
              })
            }
            className="w-full range"
          />
        </div>
        <div className="mb-4">
          <label>Saturation:</label>
          <input
            type="range"
            min="0"
            max="100"
            value={detectionOptions.saturation}
            onChange={(e) =>
              setDetectionOptions({
                ...detectionOptions,
                saturation: Number(e.target.value),
              })
            }
            className="w-full range"
          />
        </div>
        <div className="mb-4">
          <label>Image Size:</label>
          <select
            value={detectionOptions.imageSize}
            onChange={(e) =>
              setDetectionOptions({
                ...detectionOptions,
                imageSize: e.target.value,
              })
            }
            className="select select-bordered w-full max-w-xs"
          >
            <option value="small">Small</option>
            <option value="medium">Medium</option>
            <option value="large">Large</option>
          </select>
        </div>
      </div>
    </div>
  );
};

export default App;
