import torch
import cv2
import numpy as np
import mss
import time
import yaml
from PIL import Image
from yolov5 import YOLOv5
import keyboard
import warnings
import ctypes
import pygetwindow as gw
import logging

warnings.filterwarnings("ignore", category=FutureWarning)

# Set up logging to print to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define colors for each class
COLORS = {
    'Block': (255, 0, 0),       # Blue
    'Inverse': (128, 0, 128),   # Purple
    'Pipe': (0, 255, 0),        # Green
    'Slider': (255, 255, 0),    # Yellow
    'Player': (255, 192, 203),  # Pink
    'Bomb': (0, 0, 255)         # Red
}

# Load class names from data.yaml
def load_class_names(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['names']

# Define paths
weights_path = 'runs/train/beatblockv52/weights/best.pt'
data_yaml_path = 'data.yaml'

# Load class names
class_names = load_class_names(data_yaml_path)

# Load the model
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = YOLOv5(weights_path, device=device)

text = ''

# Define ctypes mouse move function
def move_mouse(x, y):
    ctypes.windll.user32.SetCursorPos(x, y)


def process_frame(frame):
    # Convert the frame to PIL Image
    image = Image.fromarray(frame)

    # Perform inference
    results = model.predict(image)

    # Draw results
    img_np = np.array(image)
    detections = []
    player_center = None

    for *xyxy, conf, cls in results.xyxy[0]:
        if conf > 0.6:  # Only consider predictions with confidence greater than 60%
            label = f'{class_names[int(cls)]} {conf:.2f}'  # Use class_names list
            xyxy = torch.tensor(xyxy).view(-1).tolist()
            detections.append((xyxy, label))  # Save detection coordinates and labels for further processing

            # Check if the detected class is 'player'
            if class_names[int(cls)] == 'player':
                player_center = (
                    (xyxy[0] + xyxy[2]) / 2,
                    (xyxy[1] + xyxy[3]) / 2
                )
            color = COLORS.get(class_names[int(cls)], (0, 0, 255))
            cv2.rectangle(img_np, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
            cv2.putText(img_np, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img_np, detections, player_center

def move_mouse_to_nearest_block(detections, window, screen_center, player_center, frame):
    global text
    if not detections:
        return

    # Filter detections for 'Slider', 'Block', and 'Pipe'
    slider_block_pipe_detections = [(det, label) for det, label in detections if label.startswith('Slider') or label.startswith('Block') or label.startswith('Pipe')]

    target_center = player_center if player_center else screen_center

    distances = []
    pipe_block = None

    for det, label in slider_block_pipe_detections:
        if len(det) != 4:
            logging.warning(f"Unexpected detection format: {det}")
            continue

        x1, y1, x2, y2 = map(int, det)
        block_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        screen_block_center = (
            window.left + block_center[0] * (window.width / window.width),
            window.top + block_center[1] * (window.height / window.height)
        )

        distance = np.linalg.norm(np.array(screen_block_center) - np.array(target_center))
        if label.startswith('Pipe'):
            if distance < 100:
                pipe_block = (distance, screen_block_center)
        if distance < 150:
            distances.append((distance, screen_block_center, label))
        

    if pipe_block:
        _, nearest_pipe_block_center = pipe_block
        swipe_start = np.array(nearest_pipe_block_center) - np.array([100, 0])
        swipe_end = np.array(nearest_pipe_block_center) + np.array([100, 0])
        #logging.info(f"Pipe, swiping from {swipe_start} to {swipe_end}.")
        text = f"Pipe, swiping from {swipe_start} to {swipe_end}."
        move_mouse(int(swipe_start[0]), int(swipe_start[1]))
        time.sleep(0.001)
        move_mouse(int(swipe_end[0]), int(swipe_end[1]))
    elif distances:
        _, nearest_block_center, label = min(distances, key=lambda x: x[0])
        if label.startswith('Slider'):
            #logging.info(f"Following Slider | {nearest_block_center}.")
            text = f"Following Slider | {nearest_block_center}."
        elif label.startswith('Block'):
            #logging.info(f'Moving to Block | {nearest_block_center}.')
            text = f'Moving to Block | {nearest_block_center}.'

    # Draw background rectangle and text on frame
    if text:  # Ensure text is not empty
        text_position = (10, 60)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        text_rect_x1 = text_position[0] - 10
        text_rect_y1 = text_position[1] - text_height - 10
        text_rect_x2 = text_position[0] + text_width + 10
        text_rect_y2 = text_position[1] + baseline + 10

        # Draw the background rectangle
        cv2.rectangle(frame, (text_rect_x1, text_rect_y1), (text_rect_x2, text_rect_y2), (0, 0, 0), cv2.FILLED)
        
        # Draw the text
        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    
    if 'nearest_block_center' in locals():
        move_mouse(int(nearest_block_center[0]), int(nearest_block_center[1]))


def main():
    app_name = 'Beatblock'  # Replace with the actual window title
    windows = gw.getWindowsWithTitle(app_name)

    if not windows:
        print(f"No window found with title '{app_name}'")
        return
    
    window = windows[0]

    monitor = {
        'left': window.left,
        'top': window.top,
        'width': window.width,
        'height': window.height
    }
    
    window_name = "YOLOv5 Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    screen_center = (monitor['left'] + monitor['width'] // 2, monitor['top'] + monitor['height'] // 2)

    with mss.mss() as sct:
        while True:
            if keyboard.is_pressed('q'):
                break

            start_time = time.time()
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR

            processed_frame, detections, player_center = process_frame(frame)

            # Move the mouse to the nearest 'Slider', 'Block', or perform swipe action if a 'Pipe' is detected
            move_mouse_to_nearest_block(detections, window, screen_center, player_center, processed_frame)

            # Calculate and display FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            fps_text = f"FPS: {fps:.2f}"
            cv2.putText(processed_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

            # Display the processed image
            cv2.imshow(window_name, processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
