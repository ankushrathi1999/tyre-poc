import cv2
import numpy as np
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

# Path to the image file
image_path = "/home/ankush/Eternal/Eternal_Projects/MSIL_POC/model_classi/final_rim/train/type4/rim2_type4_24.jpg"

# Load the image using OpenCV
image = cv2.imread(image_path)

# Check if the image is loaded correctly
if image is None:
    raise ValueError(f"Error: Could not open or find the image at {image_path}.")

# Convert the image to RGB as Doctr expects RGB images
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Save the RGB image to a temporary file-like object for doctr
# Use a compatible format (e.g., PNG) and in-memory buffer
_, image_buffer = cv2.imencode('.png', image_rgb)
image_file = DocumentFile.from_images([image_buffer.tobytes()])  # Convert buffer to bytes

# Initialize Doctr OCR predictor
model = ocr_predictor(pretrained=True)

# Perform OCR using Doctr
results = model(image_file)

# Draw the detected text on the image
for page in results.pages:
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                (x1, y1, x2, y2) = word.geometry[0]
                text = word.value
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Display the output image with detected text
cv2.imshow('Detected Text', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the output image
# cv2.imwrite('/path/to/save/output_image.png', image)

# Print all detected text
for page in results.pages:
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                print(f"Detected text: {word.value} with confidence {word.confidence:.2f}")

