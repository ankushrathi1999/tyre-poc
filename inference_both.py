import cv2
import random
import numpy as np
from ultralytics import YOLO

# Define the model paths and video paths
detection_model_path_tyre = "/home/ankush/Eternal/Eternal_Projects/MSIL_POC/model/model_tyre.pt"
classification_model_path_tyre = "/home/ankush/Eternal/Eternal_Projects/MSIL_POC/model/model_classi_tyre_new.pt"
vdo_path_tyre = "/home/ankush/Eternal/Eternal_Projects/MSIL_POC/model_testing_vedio/rec10.mp4"

detection_model_path_rim = "/home/ankush/Eternal/Eternal_Projects/MSIL_POC/model/model_rim.pt"
classification_model_path_rim = "/home/ankush/Eternal/Eternal_Projects/MSIL_POC/model/model_classi_rim_new.pt"
vdo_path_rim = "/home/ankush/Eternal/Eternal_Projects/MSIL_POC/model_testing_vedio/rim1.mp4"

# Define the window names
window_name_tyre = 'Tyre Detection'
window_name_rim = 'Rim Detection'

# Function to plot one bounding box on the image
def plot_one_box(x, img, color=(0, 255, 0), label=None, line_thickness=1):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

# Function to process a video frame and apply object detection
def process_frame_tyre(frame, model_dt, model_cls):
    results_dt = model_dt.predict(frame, conf=0.25)
    
    for result in results_dt:
        bboxs = result.boxes.xyxy.cpu()
        conf = result.boxes.conf.cpu()
        cls = result.boxes.cls.cpu()
        
        for bbox, cnf, cs in zip(bboxs, conf, cls):
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])
            class_id = int(cs)
            # class_label = model_dt.names[class_id]  # Get class name from class id
            if (xmin >= 700 and xmax <= 1600) and (ymin >= 300 and ymax <= 1630):
                roi_resized = frame[ymin:ymax, xmin:xmax]
                # Run classification on the RoI
                class_results = model_cls.predict(roi_resized)

                # Get the predicted class label
                class_label = class_results[0].names[class_results[0].probs.top1]  # Assuming top1 gives the desired class
                class_label_str = str(class_label)  # Convert to string
                if class_label == 'type1':
                    class_out = '1 blue line'
                if class_label == 'type2':
                    class_out = '1 orange line'
                if class_label == 'type3':
                    class_out = '1 yellow and 1 orange line'
                if class_label == 'type4':
                    class_out = '1 blue and 1 orange line'
                if class_label == 'type5':
                    class_out = '3 green line'
                if class_label == 'type6':
                    class_out = '2 blue line'

                # Plot the bounding box with the class label
                plot_one_box([xmin, ymin, xmax, ymax], frame, line_thickness=3, label=class_out)
    
    return frame

def process_frame_rim(frame, model_dt, model_cls):
    # Flag to check if a tyre box is printed in the current frame
    tyre_in_roi = False

    # Run the object detection model on the frame
    results_dt = model_dt.predict(frame, conf=0.25)

    for result in results_dt:
        bboxs = result.boxes.xyxy.cpu()
        conf = result.boxes.conf.cpu()
        cls = result.boxes.cls.cpu()

        for bbox, cnf, cs in zip(bboxs, conf, cls):
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])
            class_id = int(cs)
            class_label = model_dt.names[class_id]  # Get class name from detection model
            roi_resized = frame[ymin:ymax, xmin:xmax]

            # Check if the detected object is within the region of interest (ROI)
            if (xmin >= 50 and xmax <= 3000) and (ymin >= 50 and ymax <= 3000):
                # If the detected class is 'tyre', run classification on the ROI
                if class_label == 'tyre':
                    # Run classification on the ROI
                    class_results = model_cls.predict(roi_resized)
                    class_label_cls = class_results[0].names[class_results[0].probs.top1]  # Get predicted class label
                    class_label_str = str(class_label_cls)

                    # Check if the tyre is fully inside the ROI
                    if (xmin > 50 and xmax < 3000) and (ymin > 50 and ymax < 3000):
                        tyre_in_roi = True  # Tyre is fully inside ROI

                else:
                    # Use the detection model's class label if not 'tyre'
                    class_label_str = class_label

                # Determine the final output label
                if class_label_str == 'type1':
                    class_out = 'Black, 15_inch, 5_nut'
                elif class_label_str == 'type2':
                    class_out = 'Silver, 16_inch, 5_nut'
                elif class_label_str == 'type3':
                    class_out = 'Alloy, 15_inch, 5_nut'
                elif class_label_str == 'type4':
                    class_out = 'Alloy, 16_inch, 5_nut'
                elif class_label_str == 'type5':
                    class_out = 'Alloy, 15_inch, 5_nut'
                elif class_label_str == 'type6':
                    class_out = 'Black, 15_inch, 4_nut'
                elif class_label_str == 'type7':
                    class_out = 'Alloy, 15_inch, 4_nut'
                elif class_label_str == 'type8':
                    class_out = 'Alloy, 16_inch, 4_nut'
                elif class_label_str == 'type9':
                    class_out = 'Black, 16_inch, 5_nut'
                elif class_label_str == 'type10':
                    class_out = 'Alloy, 16_inch, 5_nut'
                elif class_label_str == 'type11':
                    class_out = 'Alloy(Diamond), 16_inch, 5_nut'
                else:
                    class_out = class_label_str

                # Print the class label's bounding box only if tyre is in the ROI
                if tyre_in_roi:
                    plot_one_box([xmin, ymin, xmax, ymax], frame, color=(0, 255, 0), line_thickness=3, label=class_out)

            # Check if "nut open" or "nut close" should be printed
            if tyre_in_roi and ("nut_open" in class_out.lower() or "nut_close" in class_out.lower()):
                # Set the color for "nut open" or "nut close"
                if "nut_open" in class_out.lower():
                    box_color = (0, 255, 0)  # Green for "nut open"
                elif "nut_close" in class_out.lower():
                    box_color = (255, 0, 0)  # Blue for "nut close"

                # Plot the bounding box with the correct label and color
                plot_one_box([xmin, ymin, xmax, ymax], frame, color=box_color, line_thickness=3, label=class_out)

    return frame


def main():
    # Load the models
    model_tyre = YOLO(detection_model_path_tyre)
    model_cls_tyre = YOLO(classification_model_path_tyre)

    model_rim = YOLO(detection_model_path_rim)
    model_cls_rim = YOLO(classification_model_path_rim)
    
    # Initialize video capture
    cap_tyre = cv2.VideoCapture(vdo_path_tyre)
    cap_rim = cv2.VideoCapture(vdo_path_rim)

    # Create a VideoCapture object by cctv
    # cap_tyre = cv2.VideoCapture("rtsp://admin:Ankush123@192.168.1.65:554/Streaming/channels/101")
    # cap_rim = cv2.VideoCapture("rtsp://admin:Ankush123@192.168.1.64:554/Streaming/channels/101")

    # Check if the camera opened successfully 
    if not cap_tyre.isOpened() or not cap_rim.isOpened():
        print("Error: Could not open video or camera.")
        return

    while True:
        success_tyre, img_tyre = cap_tyre.read()
        success_rim, img_rim = cap_rim.read()
        
        if not success_tyre or not success_rim:
            print('problem')
            break
        
        img_tyre = process_frame_tyre(img_tyre, model_tyre, model_cls_tyre)
        img_rim = process_frame_rim(img_rim, model_rim, model_cls_rim)
        
        # Resize the images to fit side by side
        img_tyre = cv2.resize(img_tyre, (960, 900))
        img_rim = cv2.resize(img_rim, (960, 900))
        
        # Create a horizontal stack of the two images
        img_stack = np.hstack((img_tyre, img_rim))
        
        # Display the output
        # cv2.imshow(window_name_tyre, img_tyre)
        # cv2.imshow(window_name_rim, img_rim)
        cv2.imshow("Eternal Robotics", img_stack)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    # cap.release()
    # out_vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()