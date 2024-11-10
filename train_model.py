from ultralytics import YOLO
import torch

# Load the YOLO model
model = YOLO('yolov8m.pt')

# Path to the dataset configuration file
path = '/home/ankush/Eternal/Eternal_Projects/MSIL_POC/Training_data_tyre/data_yaml.yaml'

# Set training parameters
batch_size = 8  # Reduce the batch size
epochs = 100
imgsz = 640

# Specify the result path and name
result_path = '/home/ankush/Eternal/Eternal_Projects/MSIL_POC/Training_data_tyre/results'
result_name = 'tyre_detection_model'

# Train the model
results = model.train(data=path, epochs=epochs, imgsz=imgsz, batch=batch_size, project=result_path, name=result_name)

# Clear cache to free up GPU memory
torch.cuda.empty_cache()
