from ultralytics import YOLO
import cv2
import torch
import numpy as np
import time
import sys
import os
from simple_lama_inpainting import SimpleLama

# Load the exported TensorRT model
yolo1 = YOLO("yolo26n.engine")

# Basic Object List to Avoid Errors
object_list = [0]

# Ignore List for Ignorable Objects
ignore_list = [1]

# Load LaMa
try:
    lama1 = SimpleLama()
except Exception as e:
    print("LaMa broke :(")
    exit(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_IN_H = 240
MODEL_IN_W = 432
USE_LAMA = True

# Start Video Capture
cap = cv2.VideoCapture(0)

# Catch Feed Error
if not cap.isOpened():
    print("Error opening camera")
    sys.exit(1)

frame_count = 0

# --- INITIALIZE FPS VARIABLES ---
fps_start_time = time.time()
fps_frame_counter = 0
fps_to_display = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    original_h, original_w = frame.shape[:2]
    frame_resized = cv2.resize(
        frame, (MODEL_IN_W, MODEL_IN_H),
        interpolation=cv2.INTER_AREA
    )

    # STEP 1: YOLO Detection
    results = yolo1.track(
        frame_resized, persist=True, classes=[0], conf=0.3, verbose=False, tracker = "bytetrack.yaml", task = "detect"
    ) # tracker = 

    boxes = results[0].boxes
    mask_np = None

    if boxes is not None and len(boxes) > 0:
        # STEP 2: Build mask from YOLO boxes
        combined_mask = np.zeros((MODEL_IN_H, MODEL_IN_W), dtype=np.uint8)
        
        # Track Objects Too
        if results[0].boxes.id is not None:
            object_ids = results[0].boxes.id
    	    # Convert to List
            object_list = object_ids.cpu().numpy().astype(int).tolist()

        for box in boxes:
        
            if box.id not in ignore_list:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                x1 = max(0, min(x1, MODEL_IN_W - 1))
                y1 = max(0, min(y1, MODEL_IN_H - 1))
                x2 = max(0, min(x2, MODEL_IN_W - 1))
                y2 = max(0, min(y2, MODEL_IN_H - 1))

                combined_mask[y1:y2, x1:x2] = 1

        mask_np = combined_mask

        kernel = np.ones((11, 11), np.uint8)
        mask_np = cv2.dilate(mask_np, kernel, iterations=2)

        mask_np_255 = (mask_np * 255).astype(np.uint8)
        mask_smooth = cv2.GaussianBlur(mask_np_255, (15, 15), 0)
        mask_smooth = np.ascontiguousarray(mask_smooth, dtype=np.uint8)

    if mask_np is not None and np.any(mask_np > 0):
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        try:
            frame_rgb_np = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
            mask_binary = (mask_smooth > 128).astype(np.uint8) * 255
            mask_binary_np = np.ascontiguousarray(mask_binary, dtype=np.uint8)

            lama_output = lama1(frame_rgb_np, mask_binary_np)

            lama_output_np = np.ascontiguousarray(lama_output, dtype=np.uint8)
            stage1_result = cv2.cvtColor(lama_output_np, cv2.COLOR_RGB2BGR)

        except Exception as e:
            print(f"⚠ LaMa error: {e}")
            stage1_result = frame_resized

        final_frame = stage1_result
    else:
        final_frame = frame_resized

    final_display = cv2.resize(final_frame, (original_w, original_h))

    fps_frame_counter += 1
    current_time = time.time()
    elapsed_time = current_time - fps_start_time

    if elapsed_time > 1.0:
        fps_to_display = fps_frame_counter / elapsed_time
        fps_frame_counter = 0
        fps_start_time = current_time

    cv2.putText(
        final_display, f"FPS: {fps_to_display:.1f}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )
    
    # Print List of Detected Objects
    for o, objectx in enumerate(object_list):
    	#Get new Y-coordinate
    	curr_y = 60 + (o * 30)
    	# 60 is the initial y-start, and each line gets + 30
    	
    	cv2.putText(
    		final_display, str(objectx), (30, curr_y), 
    		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    	)
    	
    # Test Toggle
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        if 1 not in ignore_list:
            ignore_list.append(1)
              
        elif 1 in ignore_list:
            ignore_list.remove(1)

    cv2.imshow("Output", final_display)

    # Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
       

cap.release()
cv2.destroyAllWindows()
print("Ended Stream")
