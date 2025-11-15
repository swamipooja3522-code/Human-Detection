import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading

# Load YOLO model (COCO dataset → person = class 0)
model = YOLO("yolov8n.pt")

root = tk.Tk()
root.title("Person Detection GUI")
root.geometry("950x800")

panel = None
selected_image_path = None
selected_video_path = None

result_text = tk.StringVar()
result_text.set("Person Count Will Appear Here")


# ----------------------------
# AUTO CAMERA DETECTOR
# ----------------------------
def get_webcam():
    for cam_id in range(3):  # Try 0,1,2
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            cap.release()
            return cam_id
    return None


# ----------------------------
# PERSON DETECTION FROM RESULTS
# ----------------------------
def get_person_boxes(result):
    boxes = result.boxes.xyxy
    classes = result.boxes.cls
    person_boxes = []

    for box, cls in zip(boxes, classes):
        if int(cls) == 0:  # PERSON class
            person_boxes.append(box)

    return person_boxes


# ----------------------------
# IMAGE SELECT
# ----------------------------
def select_image():
    global selected_image_path, panel
    selected_image_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if selected_image_path:
        img = Image.open(selected_image_path).resize((500, 400))
        img = ImageTk.PhotoImage(img)

        if panel is None:
            panel = tk.Label(root, image=img)
            panel.image = img
            panel.pack(pady=20)
        else:
            panel.configure(image=img)
            panel.image = img


# ----------------------------
# IMAGE PERSON DETECT
# ----------------------------
def detect_image():
    global selected_image_path, panel
    if not selected_image_path:
        result_text.set("Please select an image first!")
        return

    results = model(selected_image_path)
    result = results[0]

    # Person only
    person_boxes = get_person_boxes(result)
    person_count = len(person_boxes)

    result_text.set(f"Persons Detected: {person_count}")

    # Annotated image save
    output_path = "person_output.jpg"
    result.save(output_path)

    # Show detected image in GUI
    img = Image.open(output_path).resize((500, 400))
    img = ImageTk.PhotoImage(img)
    panel.configure(image=img)
    panel.image = img


# ----------------------------
# VIDEO SELECT
# ----------------------------
def select_video():
    global selected_video_path
    selected_video_path = filedialog.askopenfilename(
        title="Select Video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
    )


# ----------------------------
# VIDEO PERSON DETECT
# ----------------------------
def detect_video():
    if not selected_video_path:
        result_text.set("Please select a video first!")
        return

    threading.Thread(target=run_video_detection).start()


def run_video_detection():
    cap = cv2.VideoCapture(selected_video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        result = results[0]

        person_boxes = get_person_boxes(result)
        person_count = len(person_boxes)

        frame2 = frame.copy()

        # Draw person boxes
        for box in person_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Count on screen
        cv2.putText(frame2, f"Persons: {person_count}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 3)

        cv2.imshow("Person Video Detection (Press Q)", frame2)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ----------------------------
# WEBCAM PERSON DETECT
# ----------------------------
def detect_webcam():
    threading.Thread(target=run_webcam_detection).start()


def run_webcam_detection():
    cam_id = get_webcam()
    if cam_id is None:
        result_text.set("No webcam found!")
        return

    cap = cv2.VideoCapture(cam_id)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        result = results[0]

        person_boxes = get_person_boxes(result)
        person_count = len(person_boxes)

        frame2 = frame.copy()

        for box in person_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame2, f"Persons: {person_count}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 3)

        cv2.imshow("Person Webcam Detection (Press Q)", frame2)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ----------------------------
# BUTTONS
# ----------------------------
tk.Button(root, text="Select Image", width=22, height=2,
          command=select_image).pack(pady=5)

tk.Button(root, text="Detect Person in Image", width=22, height=2,
          command=detect_image).pack(pady=5)

tk.Button(root, text="Select Video", width=22, height=2,
          command=select_video).pack(pady=5)

tk.Button(root, text="Detect Person in Video", width=22, height=2,
          command=detect_video).pack(pady=5)

tk.Button(root, text="Detect Person via Webcam", width=22, height=2,
          command=detect_webcam).pack(pady=5)

tk.Label(root, textvariable=result_text, font=("Arial", 16)).pack(pady=10)

root.mainloop()