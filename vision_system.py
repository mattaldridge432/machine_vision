#!/usr/bin/env python3
"""
Advanced Machine Vision System with Socket Control
A comprehensive computer vision platform with real-time processing and remote control.
"""

import cv2
import numpy as np
import socket
import threading
import json
import time
import queue
from collections import deque
from datetime import datetime
import os
import sys

class RingBuffer:
   """Efficient ring buffer for video frame storage"""
   def __init__(self, size):
       self.size = size
       self.buffer = deque(maxlen=size)
   
   def append(self, item):
       self.buffer.append(item)
   
   def get_all(self):
       return list(self.buffer)
   
   def clear(self):
       self.buffer.clear()

class VisionSystem:
   """Main machine vision system with extensive capabilities"""
   
   def __init__(self):
       # Camera settings
       self.camera_id = 0
       self.cap = None
       self.frame_width = 640
       self.frame_height = 480
       self.fps = 30
       
       # System state
       self.running = False
       self.paused = False
       self.recording = False
       self.show_display = True
       
       # Detection modes
       self.detection_modes = {
           'motion': False,
           'object': False,
           'color': False,
           'edge': False,
           'contour': False,
           'face': False,
           'litter': False,
           'vehicle': False,
           'person': False,
           'line': False,
           'circle': False,
           'template': False
       }
       
       # Motion detection settings
       self.motion_threshold = 25
       self.motion_min_area = 500
       self.motion_blur_size = 21
       self.motion_sensitivity = 1.0
       
       # Object detection settings (background subtraction)
       self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
           history=500, varThreshold=16, detectShadows=True
       )
       self.object_min_area = 1000
       self.object_max_area = 50000
       
       # Color detection settings
       self.color_lower = np.array([0, 100, 100])
       self.color_upper = np.array([10, 255, 255])
       self.color_blur = 5
       
       # Edge detection settings
       self.canny_threshold1 = 50
       self.canny_threshold2 = 150
       self.canny_aperture = 3
       
       # Contour detection settings
       self.contour_mode = cv2.RETR_EXTERNAL
       self.contour_method = cv2.CHAIN_APPROX_SIMPLE
       self.contour_min_area = 100
       
       # Face detection settings
       try:
           self.face_cascade = cv2.CascadeClassifier(
               cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
           )
           self.face_scale_factor = 1.1
           self.face_min_neighbors = 5
           self.face_min_size = (30, 30)
       except:
           self.face_cascade = None
           print("Warning: Face cascade not loaded")
       
       # Line detection settings
       self.hough_rho = 1
       self.hough_theta = np.pi/180
       self.hough_threshold = 100
       self.hough_min_line_length = 50
       self.hough_max_line_gap = 10
       
       # Circle detection settings
       self.circle_dp = 1
       self.circle_min_dist = 50
       self.circle_param1 = 50
       self.circle_param2 = 30
       self.circle_min_radius = 10
       self.circle_max_radius = 100
       
       # Template matching settings
       self.template = None
       self.template_threshold = 0.8
       self.template_method = cv2.TM_CCOEFF_NORMED
       
       # Litter detection settings (specialized motion + shape analysis)
       self.litter_enabled = False
       self.litter_cooldown = 5.0  # seconds between detections
       self.litter_last_detection = 0
       self.litter_motion_threshold = 30
       self.litter_min_area = 200
       self.litter_max_area = 10000
       
       # Recording settings
       self.output_dir = 'recordings'
       self.pre_event_seconds = 10
       self.post_event_seconds = 10
       self.save_with_boxes = True
       self.video_codec = 'mp4v'
       self.video_extension = '.mp4'
       
       # Frame buffers
       self.pre_buffer = RingBuffer(int(self.fps * self.pre_event_seconds))
       self.post_buffer = []
       self.post_buffer_active = False
       self.post_buffer_frames_needed = 0
       
       # Display settings
       self.show_fps = True
       self.show_info = True
       self.box_color = (0, 255, 0)
       self.box_thickness = 2
       self.text_color = (0, 255, 0)
       self.text_scale = 0.6
       self.text_thickness = 2
       
       # Image processing settings
       self.apply_blur = False
       self.blur_kernel = 5
       self.apply_sharpen = False
       self.apply_denoise = False
       self.brightness = 0
       self.contrast = 1.0
       self.saturation = 1.0
       self.apply_grayscale = False
       
       # ROI (Region of Interest) settings
       self.roi_enabled = False
       self.roi_x = 0
       self.roi_y = 0
       self.roi_width = 640
       self.roi_height = 480
       
       # Background frame for motion detection
       self.background = None
       self.prev_frame = None
       
       # Performance tracking
       self.frame_count = 0
       self.start_time = time.time()
       self.current_fps = 0
       
       # Socket server
       self.socket_host = '127.0.0.1'
       self.socket_port = 5555
       self.socket_server = None
       self.socket_thread = None
       self.command_queue = queue.Queue()
       
       # Create output directory
       os.makedirs(self.output_dir, exist_ok=True)
   
   def start_camera(self):
       """Initialize camera capture"""
       self.cap = cv2.VideoCapture(self.camera_id)
       self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
       self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
       self.cap.set(cv2.CAP_PROP_FPS, self.fps)
       
       if not self.cap.isOpened():
           raise Exception(f"Cannot open camera {self.camera_id}")
       
       # Get actual camera properties
       self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
       self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
       self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
       if self.fps == 0:
           self.fps = 30
       
       print(f"Camera initialized: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
   
   def preprocess_frame(self, frame):
       """Apply image preprocessing filters"""
       processed = frame.copy()
       
       # Brightness and contrast
       if self.brightness != 0 or self.contrast != 1.0:
           processed = cv2.convertScaleAbs(processed, alpha=self.contrast, beta=self.brightness)
       
       # Saturation
       if self.saturation != 1.0:
           hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV).astype(np.float32)
           hsv[:,:,1] = hsv[:,:,1] * self.saturation
           hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
           processed = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
       
       # Blur
       if self.apply_blur and self.blur_kernel > 1:
           ksize = self.blur_kernel if self.blur_kernel % 2 == 1 else self.blur_kernel + 1
           processed = cv2.GaussianBlur(processed, (ksize, ksize), 0)
       
       # Denoise
       if self.apply_denoise:
           processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
       
       # Sharpen
       if self.apply_sharpen:
           kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
           processed = cv2.filter2D(processed, -1, kernel)
       
       # Grayscale
       if self.apply_grayscale:
           processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
           processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
       
       return processed
   
   def apply_roi(self, frame):
       """Extract region of interest"""
       if self.roi_enabled:
           x, y = self.roi_x, self.roi_y
           w, h = self.roi_width, self.roi_height
           return frame[y:y+h, x:x+w]
       return frame
   
   def detect_motion(self, frame):
       """Detect motion in frame"""
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       gray = cv2.GaussianBlur(gray, (self.motion_blur_size, self.motion_blur_size), 0)
       
       if self.background is None:
           self.background = gray
           return [], frame
       
       # Compute difference
       frame_delta = cv2.absdiff(self.background, gray)
       thresh = cv2.threshold(frame_delta, int(self.motion_threshold * self.motion_sensitivity), 
                              255, cv2.THRESH_BINARY)[1]
       thresh = cv2.dilate(thresh, None, iterations=2)
       
       # Find contours
       contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
       detections = []
       for contour in contours:
           area = cv2.contourArea(contour)
           if area < self.motion_min_area:
               continue
           
           x, y, w, h = cv2.boundingRect(contour)
           detections.append({'type': 'motion', 'box': (x, y, w, h), 'area': area})
           cv2.rectangle(frame, (x, y), (x+w, y+h), self.box_color, self.box_thickness)
           cv2.putText(frame, f'Motion: {int(area)}', (x, y-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_color, 1)
       
       # Update background
       self.background = cv2.addWeighted(self.background, 0.95, gray, 0.05, 0)
       
       return detections, frame
   
   def detect_objects(self, frame):
       """Detect objects using background subtraction"""
       fg_mask = self.bg_subtractor.apply(frame)
       fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, 
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
       
       contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
       detections = []
       for contour in contours:
           area = cv2.contourArea(contour)
           if area < self.object_min_area or area > self.object_max_area:
               continue
           
           x, y, w, h = cv2.boundingRect(contour)
           detections.append({'type': 'object', 'box': (x, y, w, h), 'area': area})
           cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), self.box_thickness)
           cv2.putText(frame, f'Object: {int(area)}', (x, y-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, (255, 0, 0), 1)
       
       return detections, frame
   
   def detect_color(self, frame):
       """Detect specific color range"""
       hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
       if self.color_blur > 0:
           hsv = cv2.GaussianBlur(hsv, (self.color_blur, self.color_blur), 0)
       
       mask = cv2.inRange(hsv, self.color_lower, self.color_upper)
       mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
       
       contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
       detections = []
       for contour in contours:
           area = cv2.contourArea(contour)
           if area < 100:
               continue
           
           x, y, w, h = cv2.boundingRect(contour)
           detections.append({'type': 'color', 'box': (x, y, w, h), 'area': area})
           cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), self.box_thickness)
           cv2.putText(frame, f'Color: {int(area)}', (x, y-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, (0, 255, 255), 1)
       
       return detections, frame
   
   def detect_edges(self, frame):
       """Detect edges using Canny"""
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       edges = cv2.Canny(gray, self.canny_threshold1, self.canny_threshold2, 
                        apertureSize=self.canny_aperture)
       
       # Overlay edges on frame
       frame_copy = frame.copy()
       frame_copy[edges != 0] = [0, 0, 255]
       
       return [{'type': 'edges'}], frame_copy
   
   def detect_contours(self, frame):
       """Detect contours in frame"""
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
       
       contours, _ = cv2.findContours(thresh, self.contour_mode, self.contour_method)
       
       detections = []
       for contour in contours:
           area = cv2.contourArea(contour)
           if area < self.contour_min_area:
               continue
           
           cv2.drawContours(frame, [contour], -1, (255, 255, 0), 2)
           M = cv2.moments(contour)
           if M['m00'] != 0:
               cx = int(M['m10']/M['m00'])
               cy = int(M['m01']/M['m00'])
               detections.append({'type': 'contour', 'center': (cx, cy), 'area': area})
               cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)
       
       return detections, frame
   
   def detect_faces(self, frame):
       """Detect faces using Haar Cascade"""
       if self.face_cascade is None:
           return [], frame
       
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       faces = self.face_cascade.detectMultiScale(
           gray, self.face_scale_factor, self.face_min_neighbors, 
           minSize=self.face_min_size
       )
       
       detections = []
       for (x, y, w, h) in faces:
           detections.append({'type': 'face', 'box': (x, y, w, h)})
           cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), self.box_thickness)
           cv2.putText(frame, 'Face', (x, y-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, (255, 0, 255), 1)
       
       return detections, frame
   
   def detect_lines(self, frame):
       """Detect lines using Hough transform"""
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       edges = cv2.Canny(gray, 50, 150, apertureSize=3)
       
       lines = cv2.HoughLinesP(edges, self.hough_rho, self.hough_theta, 
                              self.hough_threshold, minLineLength=self.hough_min_line_length,
                              maxLineGap=self.hough_max_line_gap)
       
       detections = []
       if lines is not None:
           for line in lines:
               x1, y1, x2, y2 = line[0]
               cv2.line(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
               detections.append({'type': 'line', 'points': (x1, y1, x2, y2)})
       
       return detections, frame
   
   def detect_circles(self, frame):
       """Detect circles using Hough transform"""
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       gray = cv2.medianBlur(gray, 5)
       
       circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, self.circle_dp, 
                                 self.circle_min_dist, param1=self.circle_param1,
                                 param2=self.circle_param2, 
                                 minRadius=self.circle_min_radius,
                                 maxRadius=self.circle_max_radius)
       
       detections = []
       if circles is not None:
           circles = np.uint16(np.around(circles))
           for circle in circles[0, :]:
               cx, cy, r = circle
               cv2.circle(frame, (cx, cy), r, (147, 20, 255), 2)
               cv2.circle(frame, (cx, cy), 2, (147, 20, 255), 3)
               detections.append({'type': 'circle', 'center': (int(cx), int(cy)), 'radius': int(r)})
       
       return detections, frame
   
   def detect_template(self, frame):
       """Match template in frame"""
       if self.template is None:
           return [], frame
       
       result = cv2.matchTemplate(frame, self.template, self.template_method)
       locations = np.where(result >= self.template_threshold)
       
       detections = []
       h, w = self.template.shape[:2]
       
       for pt in zip(*locations[::-1]):
           cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 127), 2)
           detections.append({'type': 'template', 'box': (pt[0], pt[1], w, h)})
       
       return detections, frame
   
   def detect_litter(self, frame):
       """Specialized litter detection (motion + shape analysis)"""
       current_time = time.time()
       if current_time - self.litter_last_detection < self.litter_cooldown:
           return [], frame
       
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       gray = cv2.GaussianBlur(gray, (21, 21), 0)
       
       if self.prev_frame is None:
           self.prev_frame = gray
           return [], frame
       
       # Detect sudden changes (dropping motion)
       frame_delta = cv2.absdiff(self.prev_frame, gray)
       thresh = cv2.threshold(frame_delta, self.litter_motion_threshold, 255, cv2.THRESH_BINARY)[1]
       thresh = cv2.dilate(thresh, None, iterations=2)
       
       contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
       detections = []
       for contour in contours:
           area = cv2.contourArea(contour)
           if area < self.litter_min_area or area > self.litter_max_area:
               continue
           
           # Check aspect ratio and compactness for litter-like objects
           x, y, w, h = cv2.boundingRect(contour)
           aspect_ratio = w / float(h)
           perimeter = cv2.arcLength(contour, True)
           compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
           
           # Filter for small, compact objects
           if 0.2 < aspect_ratio < 5 and compactness > 0.1:
               detections.append({'type': 'litter', 'box': (x, y, w, h), 'area': area})
               cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), self.box_thickness + 1)
               cv2.putText(frame, 'LITTER DETECTED!', (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
               
               # Trigger recording
               self.litter_last_detection = current_time
               self.trigger_event_recording('litter')
       
       self.prev_frame = gray
       return detections, frame
   
   def trigger_event_recording(self, event_type):
       """Trigger recording of event with pre and post buffers"""
       if self.post_buffer_active:
           return  # Already recording an event
       
       timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
       filename = f'{event_type}_{timestamp}{self.video_extension}'
       filepath = os.path.join(self.output_dir, filename)
       
       print(f"\n{'='*60}")
       print(f"EVENT DETECTED: {event_type.upper()}")
       print(f"Recording to: {filename}")
       print(f"{'='*60}\n")
       
       # Start post-buffer collection
       self.post_buffer_active = True
       self.post_buffer = []
       self.post_buffer_frames_needed = int(self.fps * self.post_event_seconds)
       
       # Start recording thread
       threading.Thread(target=self._save_event_video, 
                       args=(filepath,), daemon=True).start()
   
   def _save_event_video(self, filepath):
       """Save event video with pre and post buffers"""
       fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
       out = cv2.VideoWriter(filepath, fourcc, self.fps, 
                            (self.frame_width, self.frame_height))
       
       if not out.isOpened():
           print(f"Error: Could not open video writer for {filepath}")
           return
       
       # Write pre-buffer frames
       pre_frames = self.pre_buffer.get_all()
       for frame_data in pre_frames:
           if self.save_with_boxes:
               out.write(frame_data['annotated'])
           else:
               out.write(frame_data['original'])
       
       # Wait for post-buffer to fill
       while len(self.post_buffer) < self.post_buffer_frames_needed:
           time.sleep(0.1)
       
       # Write post-buffer frames
       for frame_data in self.post_buffer:
           if self.save_with_boxes:
               out.write(frame_data['annotated'])
           else:
               out.write(frame_data['original'])
       
       out.release()
       self.post_buffer_active = False
       self.post_buffer = []
       
       total_duration = self.pre_event_seconds + self.post_event_seconds
       print(f"\nVideo saved: {filepath} ({total_duration}s)")
   
   def process_frame(self, frame):
       """Process frame with all enabled detection modes"""
       original_frame = frame.copy()
       processed_frame = self.preprocess_frame(frame)
       
       # Apply ROI if enabled
       if self.roi_enabled:
           cv2.rectangle(processed_frame, (self.roi_x, self.roi_y),
                        (self.roi_x + self.roi_width, self.roi_y + self.roi_height),
                        (255, 255, 255), 2)
       
       all_detections = []
       
       # Run enabled detection modes
       if self.detection_modes['motion']:
           detections, processed_frame = self.detect_motion(processed_frame)
           all_detections.extend(detections)
       
       if self.detection_modes['object']:
           detections, processed_frame = self.detect_objects(processed_frame)
           all_detections.extend(detections)
       
       if self.detection_modes['color']:
           detections, processed_frame = self.detect_color(processed_frame)
           all_detections.extend(detections)
       
       if self.detection_modes['edge']:
           detections, processed_frame = self.detect_edges(processed_frame)
           all_detections.extend(detections)
       
       if self.detection_modes['contour']:
           detections, processed_frame = self.detect_contours(processed_frame)
           all_detections.extend(detections)
       
       if self.detection_modes['face']:
           detections, processed_frame = self.detect_faces(processed_frame)
           all_detections.extend(detections)
       
       if self.detection_modes['line']:
           detections, processed_frame = self.detect_lines(processed_frame)
           all_detections.extend(detections)
       
       if self.detection_modes['circle']:
           detections, processed_frame = self.detect_circles(processed_frame)
           all_detections.extend(detections)
       
       if self.detection_modes['template']:
           detections, processed_frame = self.detect_template(processed_frame)
           all_detections.extend(detections)
       
       if self.detection_modes['litter']:
           detections, processed_frame = self.detect_litter(processed_frame)
           all_detections.extend(detections)
       
       return processed_frame, original_frame, all_detections
   
   def draw_info(self, frame):
       """Draw information overlay on frame"""
       if not self.show_info:
           return frame
       
       info_y = 30
       line_height = 25
       
       if self.show_fps:
           cv2.putText(frame, f'FPS: {self.current_fps:.1f}', (10, info_y),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, self.text_thickness)
           info_y += line_height
       
       # Show active modes
       active_modes = [mode for mode, enabled in self.detection_modes.items() if enabled]
       if active_modes:
           modes_text = f'Modes: {", ".join(active_modes)}'
           cv2.putText(frame, modes_text, (10, info_y),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
           info_y += line_height
       
       # Show recording status
       if self.post_buffer_active:
           cv2.putText(frame, 'RECORDING EVENT', (10, info_y),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
       
       return frame
   
   def run(self):
       """Main processing loop"""
       self.running = True
       self.start_camera()
       self.start_socket_server()
       
       print("\n" + "="*60)
       print("MACHINE VISION SYSTEM STARTED")
       print("="*60)
       print(f"Camera: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
       print(f"Socket server: {self.socket_host}:{self.socket_port}")
       print(f"Output directory: {self.output_dir}")
       print("\nPress 'q' to quit")
       print("="*60 + "\n")
       
       fps_update_time = time.time()
       fps_frame_count = 0
       
       while self.running:
           # Process socket commands
           while not self.command_queue.empty():
               try:
                   command = self.command_queue.get_nowait()
                   self.process_command(command)
               except queue.Empty:
                   break
           
           if self.paused:
               time.sleep(0.1)
               continue
           
           ret, frame = self.cap.read()
           if not ret:
               print("Error: Cannot read frame from camera")
               break
           
           # Process frame
           processed_frame, original_frame, detections = self.process_frame(frame)
           
           # Add to pre-buffer (always collecting)
           self.pre_buffer.append({
               'original': original_frame,
               'annotated': processed_frame.copy()
           })
           
           # Add to post-buffer if active
           if self.post_buffer_active:
               self.post_buffer.append({
                   'original': original_frame,
                   'annotated': processed_frame.copy()
               })
           
           # Draw info overlay
           display_frame = self.draw_info(processed_frame)
           
           # Display frame
           if self.show_display:
               cv2.imshow('Machine Vision System', display_frame)
           
           # Calculate FPS
           fps_frame_count += 1
           if time.time() - fps_update_time >= 1.0:
               self.current_fps = fps_frame_count / (time.time() - fps_update_time)
               fps_frame_count = 0
               fps_update_time = time.time()
           
           # Handle keyboard input
           key = cv2.waitKey(1) & 0xFF
           if key == ord('q'):
               break
       
       self.cleanup()
   
   def cleanup(self):
       """Clean up resources"""
       print("\nShutting down...")
       self.running = False
       
       if self.cap:
           self.cap.release()
       cv2.destroyAllWindows()
       
       if self.socket_server:
           self.socket_server.close()
       
       print("System stopped.")
   
   def start_socket_server(self):
       """Start socket server for remote control"""
       self.socket_thread = threading.Thread(target=self._socket_server_thread, daemon=True)
       self.socket_thread.start()
   
   def _socket_server_thread(self):
       """Socket server thread"""
       self.socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
       self.socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
       
       try:
           self.socket_server.bind((self.socket_host, self.socket_port))
           self.socket_server.listen(5)
           print(f"Socket server listening on {self.socket_host}:{self.socket_port}")
       except Exception as e:
           print(f"Error starting socket server: {e}")
           return
       
       while self.running:
           try:
               self.socket_server.settimeout(1.0)
               client, addr = self.socket_server.accept()
               threading.Thread(target=self._handle_client, 
                              args=(client, addr), daemon=True).start()
           except socket.timeout:
               continue
           except Exception as e:
               if self.running:
                   print(f"Socket error: {e}")
               break
   
   def _handle_client(self, client, addr):
       """Handle client connection"""
       print(f"Client connected: {addr}")
       
       try:
           while self.running:
               data = client.recv(4096).decode('utf-8').strip()
               if not data:
                   break
               
               response = self.process_command(data)
               client.send((response + '\n').encode('utf-8'))
       except Exception as e:
           print(f"Client error: {e}")
       finally:
           client.close()
           print(f"Client disconnected: {addr}")
   
   def process_command(self, command_str):
       """Process remote command"""
       try:
           parts = command_str.strip().split(None, 1)
           if not parts:
               return "ERROR: Empty command"
           
           cmd = parts[0].lower()
           args = parts[1] if len(parts) > 1 else ""
           
           # System commands
           if cmd == 'help':
               return self.get_help()
           elif cmd == 'status':
               return self.get_status()
           elif cmd == 'quit' or cmd == 'exit':
               self.running = False
               return "OK: System shutting down"
           elif cmd == 'pause':
               self.paused = True
               return "OK: System paused"
           elif cmd == 'resume':
               self.paused = False
               return "OK: System resumed"
           
           # Detection mode commands
           elif cmd == 'enable':
               return self.enable_mode(args)
           elif cmd == 'disable':
               return self.disable_mode(args)
           
           # Parameter commands
           elif cmd == 'set':
               return self.set_parameter(args)
           elif cmd == 'get':
               return self.get_parameter(args)
           
           # Recording commands
           elif cmd == 'record':
               return self.trigger_manual_recording()
           elif cmd == 'snapshot':
               return self.save_snapshot()
           
           # Template commands
           elif cmd == 'load_template':
               return self.load_template(args)
           
           # Display commands
           elif cmd == 'show':
               self.show_display = True
               return "OK: Display enabled"
           elif cmd == 'hide':
               self.show_display = False
               return "OK: Display disabled"
           
           else:
               return f"ERROR: Unknown command '{cmd}'. Type 'help' for command list."
       
       except Exception as e:
           return f"ERROR: {str(e)}"
   
   def enable_mode(self, mode):
       """Enable detection mode"""
       mode = mode.lower().strip()
       if mode in self.detection_modes:
           self.detection_modes[mode] = True
           return f"OK: {mode} mode enabled"
       return f"ERROR: Unknown mode '{mode}'"
   
   def disable_mode(self, mode):
       """Disable detection mode"""
       mode = mode.lower().strip()
       if mode in self.detection_modes:
           self.detection_modes[mode] = False
           return f"OK: {mode} mode disabled"
       return f"ERROR: Unknown mode '{mode}'"
   
   def set_parameter(self, param_str):
       """Set system parameter"""
       try:
           parts = param_str.split(None, 1)
           if len(parts) != 2:
               return "ERROR: Usage: set <parameter> <value>"
           
           param, value_str = parts
           param = param.lower()
           
           # Parse value
           if value_str.lower() in ['true', 'on', 'yes']:
               value = True
           elif value_str.lower() in ['false', 'off', 'no']:
               value = False
           elif ',' in value_str:  # Array/tuple
               value = eval(value_str)
           else:
               try:
                   value = int(value_str)
               except:
                   try:
                       value = float(value_str)
                   except:
                       value = value_str
           
           # Set parameter
           if hasattr(self, param):
               setattr(self, param, value)
               return f"OK: {param} = {value}"
           else:
               return f"ERROR: Unknown parameter '{param}'"
       
       except Exception as e:
           return f"ERROR: {str(e)}"
   
   def get_parameter(self, param):
       """Get system parameter value"""
       param = param.lower().strip()
       if hasattr(self, param):
           value = getattr(self, param)
           return f"OK: {param} = {value}"
       return f"ERROR: Unknown parameter '{param}'"
   
   def trigger_manual_recording(self):
       """Manually trigger event recording"""
       self.trigger_event_recording('manual')
       return "OK: Manual recording triggered"
   
   def save_snapshot(self):
       """Save current frame as image"""
       timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
       filename = f'snapshot_{timestamp}.jpg'
       filepath = os.path.join(self.output_dir, filename)
       
       if self.cap and self.cap.isOpened():
           ret, frame = self.cap.read()
           if ret:
               cv2.imwrite(filepath, frame)
               return f"OK: Snapshot saved to {filename}"
       return "ERROR: Could not capture snapshot"
   
   def load_template(self, filepath):
       """Load template image for matching"""
       try:
           self.template = cv2.imread(filepath)
           if self.template is None:
               return f"ERROR: Could not load template from {filepath}"
           return f"OK: Template loaded from {filepath}"
       except Exception as e:
           return f"ERROR: {str(e)}"
   
   def get_status(self):
       """Get system status"""
       status = {
           'running': self.running,
           'paused': self.paused,
           'fps': round(self.current_fps, 2),
           'resolution': f'{self.frame_width}x{self.frame_height}',
           'active_modes': [m for m, e in self.detection_modes.items() if e],
           'recording_event': self.post_buffer_active
       }
       return f"OK: {json.dumps(status, indent=2)}"
   
   def get_help(self):
       """Get help text"""
       help_text = """
MACHINE VISION SYSTEM - COMMAND REFERENCE
==========================================

SYSTEM COMMANDS:
 help                    - Show this help message
 status                  - Get system status
 pause                   - Pause processing
 resume                  - Resume processing
 quit / exit             - Shutdown system
 show                    - Enable video display
 hide                    - Disable video display

DETECTION MODES:
 enable <mode>           - Enable detection mode
 disable <mode>          - Disable detection mode
 
 Available modes:
   motion      - Motion detection
   object      - Object detection (background subtraction)
   color       - Color-based detection
   edge        - Edge detection (Canny)
   contour     - Contour detection
   face        - Face detection
   litter      - Litter drop detection (events auto-recorded)
   vehicle     - Vehicle detection
   person      - Person detection
   line        - Line detection (Hough)
   circle      - Circle detection (Hough)
   template    - Template matching

RECORDING COMMANDS:
 record                  - Manually trigger event recording
 snapshot                - Save current frame as image
 load_template <path>    - Load template image for matching

PARAMETER COMMANDS:
 set <param> <value>     - Set parameter value
 get <param>             - Get parameter value

KEY PARAMETERS:
 Motion Detection:
   motion_threshold        - Threshold for motion (default: 25)
   motion_min_area         - Minimum motion area (default: 500)
   motion_sensitivity      - Sensitivity multiplier (default: 1.0)
 
 Object Detection:
   object_min_area         - Min object area (default: 1000)
   object_max_area         - Max object area (default: 50000)
 
 Color Detection:
   color_lower             - Lower HSV bound (array: [H,S,V])
   color_upper             - Upper HSV bound (array: [H,S,V])
 
 Edge Detection:
   canny_threshold1        - Canny low threshold (default: 50)
   canny_threshold2        - Canny high threshold (default: 150)
 
 Litter Detection:
   litter_cooldown         - Seconds between detections (default: 5.0)
   litter_min_area         - Min litter area (default: 200)
   litter_max_area         - Max litter area (default: 10000)
 
 Recording Settings:
   pre_event_seconds       - Seconds before event (default: 10)
   post_event_seconds      - Seconds after event (default: 10)
   save_with_boxes         - Save with bounding boxes (true/false)
 
 Image Processing:
   brightness              - Brightness adjustment (-100 to 100)
   contrast                - Contrast multiplier (0.5 to 3.0)
   saturation              - Saturation multiplier (0.0 to 2.0)
   apply_blur              - Enable Gaussian blur (true/false)
   blur_kernel             - Blur kernel size (3, 5, 7, etc.)
   apply_sharpen           - Enable sharpening (true/false)
 
 Display Settings:
   show_fps                - Show FPS counter (true/false)
   show_info               - Show info overlay (true/false)
   box_color               - Bounding box color (B,G,R tuple)
   box_thickness           - Box thickness (1-5)
 
 ROI (Region of Interest):
   roi_enabled             - Enable ROI (true/false)
   roi_x, roi_y            - ROI position
   roi_width, roi_height   - ROI dimensions

EXAMPLES:
 enable litter           - Start litter detection
 set motion_threshold 30 - Adjust motion sensitivity
 set brightness 20       - Increase brightness
 set color_lower [100,100,100] - Set color range
 record                  - Trigger manual recording
 disable face            - Turn off face detection

USAGE SCENARIOS:
 
 1. LITTER DETECTION (Main Use Case):
    enable litter
    set litter_min_area 200
    set pre_event_seconds 10
    set post_event_seconds 10
    - System will auto-record 20s clips when litter is dropped
 
 2. TRAFFIC MONITORING:
    enable vehicle
    enable motion
    set motion_min_area 2000
 
 3. SECURITY/INTRUSION:
    enable person
    enable motion
    set roi_enabled true
    set roi_x 100
    set roi_y 100
 
 4. COLOR TRACKING:
    enable color
    set color_lower [0,100,100]
    set color_upper [10,255,255]
 
 5. SHAPE DETECTION:
    enable circle
    enable line
    set circle_min_radius 20

==========================================
"""
       return help_text

def main():
   """Main entry point"""
   print("""
                                                                  ADVANCED MACHINE VISION SYSTEM v1.0                        Real-time Computer Vision with Socket Control                                                                 
   """)
   
   system = VisionSystem()
   
   try:
       system.run()
   except KeyboardInterrupt:
       print("\nInterrupted by user")
       system.cleanup()
   except Exception as e:
       print(f"\nError: {e}")
       import traceback
       traceback.print_exc()
       system.cleanup()

if __name__ == '__main__':
   main()
