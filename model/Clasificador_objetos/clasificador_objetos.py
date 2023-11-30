from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path, threshold=0.5):
        self.model = YOLO(model_path)
        self.threshold = threshold
        self.class_colors = {
            "boton_azul": (255, 0, 0),  # Blue
            "boton_rosa": (147, 20, 255),  # Pink
            "boton_blanco": (255, 255, 255),  # White
            "pantalla": (0, 255, 0)  # Green
        }

    def detect(self, frame):
        results = self.model(frame)[0]
        detections = []

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            class_name = results.names[int(class_id)].lower()  # Convert to lower case to match keys in the dictionary

            if score > self.threshold:  # Use self.threshold
                bbox_color = self.class_colors.get(class_name, (0, 255, 0))  # Default to green if class name not found
                detections.append((x1, y1, x2, y2, class_name, bbox_color))

        return detections
