# insect-detection-project

## 🎯 Insect Detection Using Computer Vision

The system uses drone-captured images of pheromone traps placed across the farm to detect the **presence and number of insects**. No species classification is performed — the goal is only to identify how many insects are trapped in each zone.

### 🧠 Model Overview

- **Type**: Object Detection / Counting Model
- **Purpose**: Detect insect blobs or shapes in trap images to estimate insect density
- **Approach**:
  - Train a model on labeled images of traps with bounding boxes around insects
  - Count the number of detected regions as a proxy for insect population
  - Use the count to decide whether the zone requires pesticide spraying

### 🔧 Model Architecture

- **Base Model**: YOLOv8 / EfficientDet / RetinaNet (configurable)
- **Input**: High-resolution images from drone-captured pheromone traps
- **Output**: Bounding boxes + confidence scores for each detected insect
- **Loss Function**: Standard object detection loss (CIoU / BCE)
- **Evaluation Metrics**:
  - Precision / Recall
  - mAP@0.5 for detection accuracy
  - Counting accuracy vs human labels

🧪 Training Pipeline
Data preprocessing (resize, normalize)

Model training using PyTorch or TensorFlow

Validation using held-out trap image sets

Deployed as part of the field-monitoring pipeline

📦 Dataset
Images collected  stick on  pheromone traps
Over six diffrent insect types
Manual labeling using tools like LabelImg or CVAT

Augmentation: rotation, blur, brightness adjustments

### 🏷️ Sample Labels Format
```json
{
  "filename": "trap_grid_03.jpg",
  "annotations": [
    {"x": 120, "y": 85, "width": 30, "height": 28},
    {"x": 300, "y": 140, "width": 25, "height": 25}
  ]
}

