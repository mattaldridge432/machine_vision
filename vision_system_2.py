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
       
       # Enhanced litter detection pipeline (v2 additions)
       self.litter_pipeline = None  # Will be initialized when camera starts
       self.litter_use_advanced_pipeline = True
       self.litter_pipeline_status = "Not initialized"
       self.litter_min_confidence = 0.7
       
       # Pipeline stage toggles (all enabled by default)
       self.litter_enable_persistence = True
       self.litter_enable_stationary = True
       self.litter_enable_background = True
       self.litter_enable_shape = True
       self.litter_enable_size = True
       self.litter_enable_temporal = True
       
       # Enhanced parameters
       self.litter_persistence_frames = 30
       self.litter_stationary_frames = 15
       self.litter_stationary_threshold = 5
       self.litter_background_learning_rate = 0.01
       self.litter_background_threshold = 30
       self.litter_shape_aspect_min = 0.2
       self.litter_shape_aspect_max = 4.0
       self.litter_sudden_appearance_window = 5
       
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
       
       # Initialize advanced litter detection pipeline
       if self.litter_use_advanced_pipeline:
           self.litter_pipeline = LitterVerificationPipeline(fps=self.fps)
           self.litter_pipeline.min_area = self.litter_min_area
           self.litter_pipeline.max_area = self.litter_max_area
           self.litter_pipeline.enable_persistence_check = self.litter_enable_persistence
           self.litter_pipeline.enable_stationary_check = self.litter_enable_stationary
           self.litter_pipeline.enable_background_check = self.litter_enable_background
           self.litter_pipeline.enable_shape_check = self.litter_enable_shape
           self.litter_pipeline.enable_size_check = self.litter_enable_size
           self.litter_pipeline.enable_temporal_check = self.litter_enable_temporal
           self.litter_pipeline.persistence_tracker.frames_required = self.litter_persistence_frames
           self.litter_pipeline.stationary_verifier.frames_required = self.litter_stationary_frames
           self.litter_pipeline.stationary_verifier.stationary_threshold = self.litter_stationary_threshold
           self.litter_pipeline.background_learner.learning_rate = self.litter_background_learning_rate
           self.litter_pipeline.background_learner.difference_threshold = self.litter_background_threshold
           print("Advanced litter detection pipeline initialized")
       
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
       """Enhanced litter detection with multi-stage verification"""
       current_time = time.time()
       
       # Check cooldown
       if current_time - self.litter_last_detection < self.litter_cooldown:
           return [], frame
       
       # Use advanced pipeline if enabled
       if self.litter_use_advanced_pipeline and self.litter_pipeline is not None:
           # Update background continuously
           self.litter_pipeline.update_background(frame)
           
           # Check if pipeline is ready
           if not self.litter_pipeline.is_ready():
               # Draw status message
               status_frame = frame.copy()
               init_frames_left = (self.litter_pipeline.background_learner.initialization_frames - 
                                 self.litter_pipeline.background_learner.frame_count)
               cv2.putText(status_frame, f'Learning background... ({init_frames_left} frames)', 
                          (10, frame.shape[0] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
               self.litter_pipeline_status = "Learning background"
               return [], status_frame
           
           # Run verification pipeline
           verified_detections, annotated_frame, status = self.litter_pipeline.verify_litter(frame)
           self.litter_pipeline_status = status
           
           # Trigger recording if litter verified
           if verified_detections:
               self.litter_last_detection = current_time
               self.trigger_event_recording('litter_verified')
               
               # Draw prominent alert
               cv2.putText(annotated_frame, 'LITTER VERIFIED!', 
                          (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
               cv2.putText(annotated_frame, 'RECORDING EVENT', 
                          (10, 100),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
           
           # Draw status
           cv2.putText(annotated_frame, f'Pipeline: {status}', 
                      (10, annotated_frame.shape[0] - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
           
           detections = [{'type': 'litter_verified', 'box': d['box']} for d in verified_detections]
           return detections, annotated_frame
       
       else:
           # Fallback to original simple detection
           gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
           gray = cv2.GaussianBlur(gray, (21, 21), 0)
           
           if self.prev_frame is None:
               self.prev_frame = gray
               return [], frame
           
           frame_delta = cv2.absdiff(self.prev_frame, gray)
           thresh = cv2.threshold(frame_delta, self.litter_motion_threshold, 255, cv2.THRESH_BINARY)[1]
           thresh = cv2.dilate(thresh, None, iterations=2)
           
           contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
           
           detections = []
           for contour in contours:
               area = cv2.contourArea(contour)
               if area < self.litter_min_area or area > self.litter_max_area:
                   continue
               
               x, y, w, h = cv2.boundingRect(contour)
               aspect_ratio = w / float(h)
               perimeter = cv2.arcLength(contour, True)
               compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
               
               if 0.2 < aspect_ratio < 5 and compactness > 0.1:
                   detections.append({'type': 'litter', 'box': (x, y, w, h), 'area': area})
                   cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), self.box_thickness + 1)
                   cv2.putText(frame, 'LITTER DETECTED!', (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                   
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
           'recording_event': self.post_buffer_active,
           'litter_pipeline_status': getattr(self, 'litter_pipeline_status', 'N/A'),
           'litter_advanced_mode': getattr(self, 'litter_use_advanced_pipeline', False)
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
   litter_use_advanced_pipeline - Use multi-stage verification (true/false)
   litter_enable_persistence   - Require object persistence (true/false)
   litter_enable_stationary    - Require object to be stationary (true/false)
   litter_enable_background    - Use background learning (true/false)
   litter_enable_shape         - Use shape analysis (true/false)
   litter_enable_size          - Enforce size constraints (true/false)
   litter_enable_temporal      - Check for sudden appearance (true/false)
   litter_persistence_frames   - Frames object must persist (default: 30)
   litter_stationary_frames    - Frames object must be still (default: 15)
   litter_stationary_threshold - Max movement pixels (default: 5)
   litter_background_learning_rate - Background adaptation (default: 0.01)
   litter_background_threshold - Foreground detection threshold (default: 30)
   litter_shape_aspect_min     - Min aspect ratio (default: 0.2)
   litter_shape_aspect_max     - Max aspect ratio (default: 4.0)
 
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

# ============================================================================
# ENHANCEMENT 1: Persistence Verification System
# Ensures detected objects remain in frame (not transient like people walking)
# ============================================================================

class PersistenceTracker:
   """Track object persistence to eliminate false positives"""
   
   def __init__(self, frames_required=30, iou_threshold=0.3):
       self.frames_required = frames_required  # Frames object must persist
       self.iou_threshold = iou_threshold      # Intersection over Union threshold
       self.tracked_objects = []               # List of tracked objects
       self.next_id = 0
   
   def calculate_iou(self, box1, box2):
       """Calculate Intersection over Union between two boxes"""
       x1, y1, w1, h1 = box1
       x2, y2, w2, h2 = box2
       
       # Calculate intersection
       x_left = max(x1, x2)
       y_top = max(y1, y2)
       x_right = min(x1 + w1, x2 + w2)
       y_bottom = min(y1 + h1, y2 + h2)
       
       if x_right < x_left or y_bottom < y_top:
           return 0.0
       
       intersection = (x_right - x_left) * (y_bottom - y_top)
       union = w1 * h1 + w2 * h2 - intersection
       
       return intersection / union if union > 0 else 0.0
   
   def update(self, detected_boxes):
       """Update tracker with new detections"""
       current_time = time.time()
       
       # Match detections to existing tracked objects
       matched_ids = set()
       new_detections = []
       
       for box in detected_boxes:
           best_match_id = None
           best_iou = 0
           
           for obj in self.tracked_objects:
               iou = self.calculate_iou(box, obj['box'])
               if iou > best_iou and iou > self.iou_threshold:
                   best_iou = iou
                   best_match_id = obj['id']
           
           if best_match_id is not None:
               # Update existing object
               for obj in self.tracked_objects:
                   if obj['id'] == best_match_id:
                       obj['box'] = box
                       obj['last_seen'] = current_time
                       obj['frame_count'] += 1
                       matched_ids.add(best_match_id)
                       break
           else:
               # New object
               new_detections.append(box)
       
       # Add new objects
       for box in new_detections:
           self.tracked_objects.append({
               'id': self.next_id,
               'box': box,
               'first_seen': current_time,
               'last_seen': current_time,
               'frame_count': 1,
               'confirmed': False
           })
           self.next_id += 1
       
       # Remove stale objects (not seen for 2 seconds)
       self.tracked_objects = [
           obj for obj in self.tracked_objects 
           if current_time - obj['last_seen'] < 2.0
       ]
       
       # Mark objects as confirmed if they've persisted
       confirmed_objects = []
       for obj in self.tracked_objects:
           if obj['frame_count'] >= self.frames_required and not obj['confirmed']:
               obj['confirmed'] = True
               confirmed_objects.append(obj)
       
       return confirmed_objects
   
   def get_persistent_boxes(self):
       """Get boxes of confirmed persistent objects"""
       return [obj['box'] for obj in self.tracked_objects if obj['confirmed']]


# ============================================================================
# ENHANCEMENT 2: Stationary Object Verification
# Ensures object has stopped moving (not a person walking through)
# ============================================================================

class StationaryVerifier:
   """Verify that detected objects have stopped moving"""
   
   def __init__(self, stationary_threshold=5, frames_required=15):
       self.stationary_threshold = stationary_threshold  # Max pixel movement
       self.frames_required = frames_required            # Frames to confirm
       self.object_history = {}                          # Track object positions
   
   def update(self, tracked_objects):
       """Update with tracked objects and return stationary ones"""
       stationary_objects = []
       
       for obj in tracked_objects:
           obj_id = obj['id']
           box = obj['box']
           x, y, w, h = box
           center = (x + w//2, y + h//2)
           
           if obj_id not in self.object_history:
               self.object_history[obj_id] = {
                   'positions': [center],
                   'stationary_count': 0
               }
           else:
               history = self.object_history[obj_id]
               last_pos = history['positions'][-1]
               
               # Calculate movement
               dx = abs(center[0] - last_pos[0])
               dy = abs(center[1] - last_pos[1])
               movement = (dx**2 + dy**2)**0.5
               
               # Update position history (keep last 30 positions)
               history['positions'].append(center)
               if len(history['positions']) > 30:
                   history['positions'].pop(0)
               
               # Check if stationary
               if movement < self.stationary_threshold:
                   history['stationary_count'] += 1
               else:
                   history['stationary_count'] = 0
               
               # Confirm if stationary long enough
               if history['stationary_count'] >= self.frames_required:
                   stationary_objects.append(obj)
       
       # Clean up old entries
       active_ids = {obj['id'] for obj in tracked_objects}
       self.object_history = {
           k: v for k, v in self.object_history.items() 
           if k in active_ids
       }
       
       return stationary_objects


# ============================================================================
# ENHANCEMENT 3: Advanced Background Learning
# Learns what's permanent in scene vs what's new (litter)
# ============================================================================

class BackgroundLearner:
   """Learn and adapt to scene background"""
   
   def __init__(self, learning_rate=0.01, difference_threshold=30):
       self.background_model = None
       self.learning_rate = learning_rate
       self.difference_threshold = difference_threshold
       self.frame_count = 0
       self.initialization_frames = 60  # Learn background for 2 seconds
   
   def update(self, frame):
       """Update background model"""
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       gray = cv2.GaussianBlur(gray, (21, 21), 0).astype(np.float32)
       
       if self.background_model is None:
           self.background_model = gray.copy()
       else:
           # Adaptive background update
           cv2.accumulateWeighted(gray, self.background_model, self.learning_rate)
       
       self.frame_count += 1
   
   def get_foreground_mask(self, frame):
       """Get mask of foreground objects (new items)"""
       if self.background_model is None or self.frame_count < self.initialization_frames:
           return None
       
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       gray = cv2.GaussianBlur(gray, (21, 21), 0)
       
       # Calculate difference from background
       diff = cv2.absdiff(gray, self.background_model.astype(np.uint8))
       _, mask = cv2.threshold(diff, self.difference_threshold, 255, cv2.THRESH_BINARY)
       
       # Morphological operations to clean up mask
       kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
       mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
       mask = cv2.dilate(mask, kernel, iterations=2)
       
       return mask
   
   def is_initialized(self):
       """Check if background model is ready"""
       return self.frame_count >= self.initialization_frames


# ============================================================================
# ENHANCEMENT 4: Shape and Texture Analysis
# Analyze object properties to distinguish litter from people/vehicles
# ============================================================================

class ShapeAnalyzer:
   """Analyze object shape characteristics"""
   
   def __init__(self):
       self.valid_aspect_ratios = (0.2, 4.0)    # Width/height ratio range
       self.valid_solidity_range = (0.3, 0.95)   # Filled vs outline ratio
       self.valid_extent_range = (0.3, 0.9)      # Bounding box fill ratio
   
   def analyze_contour(self, contour, frame):
       """Analyze contour properties to determine if it's litter-like"""
       area = cv2.contourArea(contour)
       if area < 100:
           return None
       
       x, y, w, h = cv2.boundingRect(contour)
       
       # Aspect ratio (should not be too elongated)
       aspect_ratio = w / float(h) if h > 0 else 0
       
       # Solidity (convex hull vs actual area)
       hull = cv2.convexHull(contour)
       hull_area = cv2.contourArea(hull)
       solidity = area / hull_area if hull_area > 0 else 0
       
       # Extent (area vs bounding box)
       extent = area / (w * h) if (w * h) > 0 else 0
       
       # Perimeter and circularity
       perimeter = cv2.arcLength(contour, True)
       circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
       
       # Calculate shape features
       features = {
           'area': area,
           'aspect_ratio': aspect_ratio,
           'solidity': solidity,
           'extent': extent,
           'circularity': circularity,
           'perimeter': perimeter,
           'box': (x, y, w, h)
       }
       
       # Determine if shape is litter-like
       is_valid = True
       reasons = []
       
       # Check aspect ratio (not too elongated like a person)
       if not (self.valid_aspect_ratios[0] <= aspect_ratio <= self.valid_aspect_ratios[1]):
           is_valid = False
           reasons.append(f"aspect_ratio:{aspect_ratio:.2f}")
       
       # Check solidity (should be fairly solid, not hollow)
       if not (self.valid_solidity_range[0] <= solidity <= self.valid_solidity_range[1]):
           is_valid = False
           reasons.append(f"solidity:{solidity:.2f}")
       
       # Check extent (should fill bounding box reasonably)
       if not (self.valid_extent_range[0] <= extent <= self.valid_extent_range[1]):
           is_valid = False
           reasons.append(f"extent:{extent:.2f}")
       
       features['is_valid'] = is_valid
       features['rejection_reasons'] = reasons
       
       return features
   
   def filter_litter_candidates(self, contours, frame):
       """Filter contours to find litter candidates"""
       candidates = []
       
       for contour in contours:
           analysis = self.analyze_contour(contour, frame)
           if analysis and analysis['is_valid']:
               candidates.append(analysis)
       
       return candidates


# ============================================================================
# ENHANCEMENT 5: Multi-Stage Litter Verification Pipeline
# Combines all techniques for robust litter detection
# ============================================================================

class LitterVerificationPipeline:
   """Complete verification pipeline for litter detection"""
   
   def __init__(self, fps=30):
       # Initialize all verification stages
       self.persistence_tracker = PersistenceTracker(
           frames_required=30,  # 1 second at 30fps
           iou_threshold=0.3
       )
       self.stationary_verifier = StationaryVerifier(
           stationary_threshold=5,
           frames_required=15  # 0.5 seconds
       )
       self.background_learner = BackgroundLearner(
           learning_rate=0.01,
           difference_threshold=30
       )
       self.shape_analyzer = ShapeAnalyzer()
       
       # Pipeline settings
       self.min_confidence_score = 0.7
       self.enable_persistence_check = True
       self.enable_stationary_check = True
       self.enable_background_check = True
       self.enable_shape_check = True
       self.enable_size_check = True
       self.enable_temporal_check = True
       
       # Size constraints
       self.min_area = 200
       self.max_area = 10000
       self.min_width = 10
       self.min_height = 10
       
       # Temporal verification
       self.require_sudden_appearance = True
       self.appearance_frames_window = 5  # Must appear within 5 frames
       
       # Detection state
       self.prev_foreground_mask = None
       self.detection_history = []
       
   def update_background(self, frame):
       """Update background model"""
       self.background_learner.update(frame)
   
   def is_ready(self):
       """Check if pipeline is ready for detection"""
       return self.background_learner.is_initialized()
   
   def verify_litter(self, frame):
       """Run complete verification pipeline"""
       if not self.is_ready():
           return [], frame, "Background learning in progress"
       
       # Stage 1: Get foreground mask
       fg_mask = self.background_learner.get_foreground_mask(frame)
       if fg_mask is None:
           return [], frame, "No foreground mask"
       
       # Stage 2: Find contours in foreground
       contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
       if len(contours) == 0:
           return [], frame, "No contours found"
       
       # Stage 3: Shape analysis
       if self.enable_shape_check:
           shape_candidates = self.shape_analyzer.filter_litter_candidates(contours, frame)
           if not shape_candidates:
               return [], frame, "No valid shapes detected"
       else:
           shape_candidates = [{'box': cv2.boundingRect(c)} for c in contours]
       
       # Stage 4: Size filtering
       if self.enable_size_check:
           size_filtered = []
           for candidate in shape_candidates:
               x, y, w, h = candidate['box']
               area = candidate.get('area', w * h)
               if (self.min_area <= area <= self.max_area and 
                   w >= self.min_width and h >= self.min_height):
                   size_filtered.append(candidate)
           
           if not size_filtered:
               return [], frame, "No objects in valid size range"
           shape_candidates = size_filtered
       
       # Stage 5: Persistence tracking
       detected_boxes = [c['box'] for c in shape_candidates]
       
       if self.enable_persistence_check:
           confirmed_objects = self.persistence_tracker.update(detected_boxes)
           if not confirmed_objects:
               return [], frame, f"Objects not persistent (need {self.persistence_tracker.frames_required} frames)"
       else:
           # Create simple object list
           confirmed_objects = [{'box': box, 'id': i} for i, box in enumerate(detected_boxes)]
       
       # Stage 6: Stationary verification
       if self.enable_stationary_check:
           stationary_objects = self.stationary_verifier.update(confirmed_objects)
           if not stationary_objects:
               return [], frame, f"Objects still moving (need {self.stationary_verifier.frames_required} stationary frames)"
       else:
           stationary_objects = confirmed_objects
       
       # Stage 7: Temporal check (sudden appearance)
       if self.enable_temporal_check and self.require_sudden_appearance:
           # Check if object appeared suddenly (not gradually)
           valid_objects = []
           for obj in stationary_objects:
               # Object must have been confirmed quickly (sudden appearance)
               if obj.get('frame_count', 999) <= self.appearance_frames_window + self.persistence_tracker.frames_required:
                   valid_objects.append(obj)
           
           if not valid_objects:
               return [], frame, "No sudden appearance detected (gradual change)"
           stationary_objects = valid_objects
       
       # Success! Draw detections
       annotated_frame = frame.copy()
       verified_detections = []
       
       for obj in stationary_objects:
           x, y, w, h = obj['box']
           
           # Draw bounding box
           cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
           
           # Draw label with confidence
           label = f"LITTER (ID:{obj['id']})"
           cv2.putText(annotated_frame, label, (x, y-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
           
           # Add frame count indicator
           frames_info = f"Frames: {obj.get('frame_count', 0)}"
           cv2.putText(annotated_frame, frames_info, (x, y+h+20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
           
           verified_detections.append(obj)
       
       status = f"VERIFIED: {len(verified_detections)} litter object(s)"
       return verified_detections, annotated_frame, status
   
   def reset(self):
       """Reset all tracking"""
       self.persistence_tracker = PersistenceTracker()
       self.stationary_verifier = StationaryVerifier()
       self.detection_history = []
