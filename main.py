import cv2

from ultralytics import YOLO

from my_tracker import MyTracker
from ultralytics.trackers.track import register_tracker
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_yaml

# Load the YOLOv8 model
model = YOLO("yolov8n.pt", verbose=False)

tracker = check_yaml('my_botsort.yml')
cfg = IterableSimpleNamespace(**yaml_load(tracker))


my_tracker = MyTracker(args=cfg)

# Open the video file
video_path = "cam2.mp4"
cap = cv2.VideoCapture(video_path)

isFirstFrame = True
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, classes=0, verbose=False)
        if(isFirstFrame):
            model.predictor.trackers = [my_tracker]
            isFirstFrame = False

                # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()