import cv2
import os

def extract_frames_from_videos(video_folder, output_folder, frame_rate=1):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all video files in the specified folder
    for filename in os.listdir(video_folder):
        video_path = os.path.join(video_folder, filename)

        # Check if the file is a video (you can add more extensions if needed)
        if os.path.isfile(video_path) and filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"Processing video: {filename}")
            extract_frames(video_path, output_folder, frame_rate)

def extract_frames(video_path, output_folder, frame_rate):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the video
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    
    # Calculate the frame interval based on the desired frame rate
    frame_interval = max(int(fps / frame_rate), 1)

    frame_count = 0
    saved_frame_count = 0

    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()

        # If no frame is returned, break the loop (end of video)
        if not ret:
            break

        # Save the frame only if it matches the interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame {saved_frame_count} from {video_path} to {frame_filename}")
            saved_frame_count += 1

        frame_count += 1

    # Release the video capture object
    video_capture.release()
    print(f"Completed processing {video_path}.")

# Example usage
video_folder = 'Tyre'           # Replace with your folder containing video files
output_folder = 'tyre_frame'   # Replace with your desired output folder path
frame_rate = 1                    # Number of frames to save per second

extract_frames_from_videos(video_folder, output_folder, frame_rate)

