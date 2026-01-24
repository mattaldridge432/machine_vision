#!/usr/bin/env python3
import cv2
import numpy as np
import csv
import time
from datetime import datetime
from pathlib import Path
import sys
from collections import deque
import mss
import threading

# ================= CONFIG =================

# Resolution presets (width, height)
RESOLUTIONS = {
    'LOW': (640, 480),
    'MEDIUM': (1280, 720),
    'HIGH': (1920, 1080),
    'ULTRA': (2560, 1440)
}

CURRENT_RESOLUTION = 'MEDIUM'
FRAME_SIZE = RESOLUTIONS[CURRENT_RESOLUTION]
FPS = 30.0

# Window position (adjust these to position the capture region)
WINDOW_X = 100
WINDOW_Y = 100

# Detection parameters
MIN_AREA = 800
MAX_AREA = 50000
STATIC_FRAMES = 60
OVERLAP_THRESH = 0.6

# Change detection thresholds
BACKGROUND_LEARN_FRAMES = 100
DIFF_THRESHOLD = 30
MIN_CHANGE_RATIO = 0.15
PERSISTENCE_CHECK_FRAMES = 20

# Adaptive background relearning
RELEARN_INTERVAL_FRAMES = 1800
RELEARN_BLEND_RATE = 0.001
RELEARN_GLOBAL_CHANGE_THRESHOLD = 0.20

# Background comparison settings
BG_COMPARE_THRESHOLD = 25
NEW_OBJECT_THRESHOLD = 0.3

# Face and plate detection parameters
FACE_DETECTION_ENABLED = True
PLATE_DETECTION_ENABLED = True
MIN_HAND_TO_LITTER_DISTANCE_INCREASE = 50
MIN_CAR_TO_LITTER_DISTANCE_INCREASE = 100
FACE_PROXIMITY_THRESHOLD = 300
PLATE_PROXIMITY_THRESHOLD = 400

# Event snippet settings
BEFORE_EVENT_SECONDS = 10
AFTER_EVENT_SECONDS = 10
BEFORE_EVENT_FRAMES = int(BEFORE_EVENT_SECONDS * FPS)
AFTER_EVENT_FRAMES = int(AFTER_EVENT_SECONDS * FPS)

# Bounding box export settings
EXPORT_WITH_BOUNDING_BOXES = True

STORAGE_DIR = Path("recordings")
EVENTS_DIR = STORAGE_DIR / "litter_events"
STORAGE_DIR.mkdir(exist_ok=True)
EVENTS_DIR.mkdir(exist_ok=True)

# =========================================


def iou(a, b):
    """Calculate Intersection over Union for two bounding boxes"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def distance_between_boxes(box1, box2):
    """Calculate distance between centers of two bounding boxes"""
    x1, y1, x2, y2 = box1
    cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
    
    x1, y1, x2, y2 = box2
    cx2, cy2 = (x1 + x2) / 2, (y1 + y2) / 2
    
    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)


class TrackedObject:
    """Represents a detected object being tracked across frames"""
    def __init__(self, bbox, frame, roi_snapshot=None):
        self.bbox = bbox
        self.first = frame
        self.last = frame
        self.static = 1
        self.reported = False
        self.max_static = 0
        self.initial_roi = roi_snapshot
        self.evaluated = False
        self.is_new_object = False
        self.bg_diff_ratio = 0
        
    def update(self, bbox, frame):
        """Update object position and tracking info"""
        self.bbox = bbox
        self.last = frame
        self.static += 1
        self.max_static = max(self.max_static, self.static)
        
    def get_center(self):
        """Get center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class FaceTracker:
    """Tracks detected faces across frames"""
    def __init__(self, bbox, frame_num):
        self.bbox = bbox
        self.first_frame = frame_num
        self.last_frame = frame_num
        self.history = [bbox]
        
    def update(self, bbox, frame_num):
        self.bbox = bbox
        self.last_frame = frame_num
        self.history.append(bbox)
        if len(self.history) > 30:
            self.history.pop(0)
            
    def get_center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class PlateTracker:
    """Tracks detected license plates across frames"""
    def __init__(self, bbox, frame_num):
        self.bbox = bbox
        self.first_frame = frame_num
        self.last_frame = frame_num
        self.history = [bbox]
        
    def update(self, bbox, frame_num):
        self.bbox = bbox
        self.last_frame = frame_num
        self.history.append(bbox)
        if len(self.history) > 30:
            self.history.pop(0)
            
    def get_center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class LitterEvent:
    """Represents a detected litter event with snippet creation"""
    def __init__(self, event_id, frame_num, obj, timestamp):
        self.event_id = event_id
        self.detection_frame = frame_num
        self.obj = obj
        self.timestamp = timestamp
        self.start_frame = max(0, frame_num - BEFORE_EVENT_FRAMES)
        self.end_frame = frame_num + AFTER_EVENT_FRAMES
        self.snippet_complete = False
        self.snippet_path = None
        self.associated_face = None
        self.associated_plate = None
        self.has_culprit = False


class LitterDetector:
    """Main litter detection and recording system"""
    
    def __init__(self):
        self.sct = mss.mss()
        self.capture_region = {
            'left': WINDOW_X,
            'top': WINDOW_Y,
            'width': FRAME_SIZE[0],
            'height': FRAME_SIZE[1]
        }
        
        self.bg_subtractor = None
        self.baseline_background = None
        self.bg_learned = False
        self.bg_learning_counter = 0
        
        self.objects = []
        self.frame_num = 0
        self.recording = False
        self.paused = False
        self.playback = False
        self.running = True
        self.frames_since_last_relearn = 0

        # Frame buffer for pre-event capture
        self.frame_buffer = deque(maxlen=BEFORE_EVENT_FRAMES)
        self.detection_buffer = deque(maxlen=BEFORE_EVENT_FRAMES)
        
        # Face and plate detection
        self.face_cascade = None
        self.plate_cascade = None
        self.tracked_faces = []
        self.tracked_plates = []
        self.initialize_detectors()
        
        # Event tracking
        self.active_events = []
        self.completed_events = []
        self.total_events = 0
        self.events_without_culprit = 0
        
        # CSV for event log
        self.event_log_file = None
        self.event_log_writer = None
        
        # Playback components
        self.event_snippets = []
        self.play_cap = None
        self.play_frame = 0
        self.play_paused = False
        self.last_playback_frame = None
        self.current_playback_index = 0
        self.last_frame_time = 0
        
        # Statistics
        self.false_positives_filtered = 0
        self.session_start = None
        self.session_id = None
        
        # Window name
        self.window_name = "Litter Detection - Screen Capture"
        
    def initialize_detectors(self):
        """Initialize face and license plate cascade classifiers"""
        try:
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            if self.face_cascade.empty():
                print("WARNING: Face cascade not loaded properly")
            else:
                print("Face detection initialized")
                
            plate_cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
            self.plate_cascade = cv2.CascadeClassifier(plate_cascade_path)
            if self.plate_cascade.empty():
                print("WARNING: Plate cascade not found, using alternative detection")
                self.plate_cascade = None
            else:
                print("License plate detection initialized")
                
        except Exception as e:
            print(f"Error initializing detectors: {e}")
            self.face_cascade = None
            self.plate_cascade = None
        
    def detect_faces(self, frame):
        """Detect faces in frame"""
        if not FACE_DETECTION_ENABLED or self.face_cascade is None:
            return []
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        return [(x, y, x+w, y+h) for (x, y, w, h) in faces]
    
    def detect_plates(self, frame):
        """Detect license plates in frame"""
        if not PLATE_DETECTION_ENABLED:
            return []
            
        if self.plate_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            plates = self.plate_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 15)
            )
            return [(x, y, x+w, y+h) for (x, y, w, h) in plates]
        
        return self.detect_plates_by_contours(frame)
    
    def detect_plates_by_contours(self, frame):
        """Detect license plate-like rectangles using contours"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)
        
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        
        plates = []
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                
                if 2.0 <= aspect_ratio <= 6.0 and w >= 50 and h >= 15:
                    plates.append((x, y, x+w, y+h))
                    
        return plates
    
    def update_face_tracking(self, face_detections):
        """Update tracked faces with new detections"""
        matched_faces = set()
        
        for det in face_detections:
            best_face, best_score = None, 0
            
            for face in self.tracked_faces:
                score = iou(det, face.bbox)
                if score > 0.3:
                    if score > best_score:
                        best_face, best_score = face, score
            
            if best_face:
                best_face.update(det, self.frame_num)
                matched_faces.add(best_face)
            else:
                new_face = FaceTracker(det, self.frame_num)
                self.tracked_faces.append(new_face)
                matched_faces.add(new_face)
        
        self.tracked_faces = [f for f in self.tracked_faces 
                             if self.frame_num - f.last_frame < 60]
    
    def update_plate_tracking(self, plate_detections):
        """Update tracked plates with new detections"""
        matched_plates = set()
        
        for det in plate_detections:
            best_plate, best_score = None, 0
            
            for plate in self.tracked_plates:
                score = iou(det, plate.bbox)
                if score > 0.3:
                    if score > best_score:
                        best_plate, best_score = plate, score
            
            if best_plate:
                best_plate.update(det, self.frame_num)
                matched_plates.add(best_plate)
            else:
                new_plate = PlateTracker(det, self.frame_num)
                self.tracked_plates.append(new_plate)
                matched_plates.add(new_plate)
        
        self.tracked_plates = [p for p in self.tracked_plates 
                              if self.frame_num - p.last_frame < 60]
    
    def check_hand_movement_to_litter(self, face, litter_obj):
        """Check if hand moved away from face toward litter location"""
        face_x1, face_y1, face_x2, face_y2 = face.bbox
        face_cx = (face_x1 + face_x2) / 2
        face_cy = (face_y1 + face_y2) / 2
        
        estimated_hand_start = (face_cx, face_y2)
        
        litter_center = litter_obj.get_center()
        
        initial_dist = np.sqrt((estimated_hand_start[0] - face_cx)**2 + 
                              (estimated_hand_start[1] - face_cy)**2)
        final_dist = np.sqrt((litter_center[0] - face_cx)**2 + 
                            (litter_center[1] - face_cy)**2)
        
        distance_increase = final_dist - initial_dist
        
        return distance_increase >= MIN_HAND_TO_LITTER_DISTANCE_INCREASE
    
    def check_car_movement_to_litter(self, plate, litter_obj):
        """Check if car moved away from plate location"""
        if len(plate.history) < 2:
            return False
            
        early_pos = plate.history[0]
        late_pos = plate.history[-1]
        
        early_center = ((early_pos[0] + early_pos[2])/2, (early_pos[1] + early_pos[3])/2)
        late_center = ((late_pos[0] + late_pos[2])/2, (late_pos[1] + late_pos[3])/2)
        
        litter_center = litter_obj.get_center()
        
        early_dist = np.sqrt((early_center[0] - litter_center[0])**2 + 
                            (early_center[1] - litter_center[1])**2)
        
        late_dist = np.sqrt((late_center[0] - litter_center[0])**2 + 
                           (late_center[1] - litter_center[1])**2)
        
        distance_increase = late_dist - early_dist
        
        return distance_increase >= MIN_CAR_TO_LITTER_DISTANCE_INCREASE
    
    def find_associated_culprit(self, litter_obj):
        """Find face or plate associated with litter object"""
        associated_face = None
        associated_plate = None
        
        for face in self.tracked_faces:
            dist = distance_between_boxes(face.bbox, litter_obj.bbox)
            if dist < FACE_PROXIMITY_THRESHOLD:
                if self.check_hand_movement_to_litter(face, litter_obj):
                    associated_face = face
                    break
        
        for plate in self.tracked_plates:
            dist = distance_between_boxes(plate.bbox, litter_obj.bbox)
            if dist < PLATE_PROXIMITY_THRESHOLD:
                if self.check_car_movement_to_litter(plate, litter_obj):
                    associated_plate = plate
                    break
        
        return associated_face, associated_plate
    
    def draw_bounding_boxes_on_frame(self, frame):
        """Draw bounding boxes on frame for export"""
        frame_copy = frame.copy()
        
        for face in self.tracked_faces:
            x1, y1, x2, y2 = face.bbox
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame_copy, "FACE", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        for plate in self.tracked_plates:
            x1, y1, x2, y2 = plate.bbox
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame_copy, "PLATE", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        for obj in self.objects:
            x1, y1, x2, y2 = obj.bbox
            
            if obj.reported:
                color = (0, 255, 0)
                label = "LITTER"
            elif obj.evaluated and not obj.is_new_object:
                color = (128, 128, 128)
                label = "Filtered"
            elif obj.static >= STATIC_FRAMES:
                color = (0, 165, 255)
                label = "Evaluating"
            else:
                color = (0, 0, 255)
                label = f"Tracking: {obj.static}"
            
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_copy, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame_copy
        
    def initialize_capture(self):
        """Initialize screen capture"""
        print(f"Screen capture initialized: {FRAME_SIZE[0]}x{FRAME_SIZE[1]}")
        print(f"Capture region: X={WINDOW_X}, Y={WINDOW_Y}")
        print("The transparent overlay window will capture this screen region")
        
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
        
        self.session_start = datetime.now()
        return True
    
    def capture_frame(self):
        """Capture frame from screen region"""
        try:
            screenshot = self.sct.grab(self.capture_region)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame = cv2.resize(frame, FRAME_SIZE)
            return frame
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
        
    def learn_baseline_background(self, frame):
        """Learn the initial clean background state"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.baseline_background is None:
            self.baseline_background = gray_blurred.astype(float)
        else:
            cv2.accumulateWeighted(gray_blurred, self.baseline_background, 0.05)
        
        self.bg_learning_counter += 1
        
        if self.bg_learning_counter >= BACKGROUND_LEARN_FRAMES:
            self.bg_learned = True
            self.frames_since_last_relearn = 0
            print(f"Baseline background learned - monitoring for litter events")
    
    def adaptive_background_update(self, frame):
        """Gradually update background to adapt to lighting changes"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        baseline_gray = self.baseline_background.astype(np.uint8)
        diff = cv2.absdiff(gray_blurred, baseline_gray)
        _, thresh = cv2.threshold(diff, BG_COMPARE_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        total_pixels = thresh.size
        changed_pixels = np.count_nonzero(thresh)
        global_change_ratio = changed_pixels / total_pixels if total_pixels > 0 else 0
        
        if global_change_ratio > RELEARN_GLOBAL_CHANGE_THRESHOLD and len(self.active_events) == 0:
            cv2.accumulateWeighted(gray_blurred, self.baseline_background, RELEARN_BLEND_RATE)

    def compare_to_baseline(self, frame, bbox):
        """Compare region to baseline background"""
        x1, y1, x2, y2 = bbox
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        current_roi = gray_blurred[y1:y2, x1:x2]
        
        baseline_roi = self.baseline_background[y1:y2, x1:x2].astype(np.uint8)
        
        diff = cv2.absdiff(current_roi, baseline_roi)
        _, thresh = cv2.threshold(diff, BG_COMPARE_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        total_pixels = thresh.size
        changed_pixels = np.count_nonzero(thresh)
        change_ratio = changed_pixels / total_pixels if total_pixels > 0 else 0
        
        return change_ratio
        
    def is_truly_new_object(self, obj, frame):
        """Determine if tracked object is actually new"""
        change_ratio = self.compare_to_baseline(frame, obj.bbox)
        obj.bg_diff_ratio = change_ratio
        
        return change_ratio >= NEW_OBJECT_THRESHOLD
        
    def start_recording(self):
        """Start a new recording session"""
        if self.recording:
            print("Already recording!")
            return
            
        if self.playback:
            print("Cannot start recording while in playback mode.")
            return
            
        # Reset state
        self.baseline_background = None
        self.bg_learned = False
        self.bg_learning_counter = 0
        self.frame_buffer.clear()
        self.detection_buffer.clear()
        self.active_events.clear()
        self.objects.clear()
        self.tracked_faces.clear()
        self.tracked_plates.clear()
        self.total_events = 0
        self.false_positives_filtered = 0
        self.events_without_culprit = 0
        self.frame_num = 0
        self.frames_since_last_relearn = 0

        # Create session ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create event log CSV
        event_log_path = EVENTS_DIR / f"session_{self.session_id}_log.csv"
        self.event_log_file = open(event_log_path, "w", newline="")
        self.event_log_writer = csv.writer(self.event_log_file)
        self.event_log_writer.writerow([
            "event_id", "session_id", "datetime", "frame", 
            "x1", "y1", "x2", "y2", "width", "height", "area",
            "static_frames", "bg_diff_ratio", "has_face", "has_plate", "snippet_path"
        ])
        
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
        
        self.recording = True
        self.paused = False
        
        print(f"Session {self.session_id} started - monitoring for litter events")
        print("Events require face or license plate detection to be saved")
        print(f"Bounding box export: {'ENABLED' if EXPORT_WITH_BOUNDING_BOXES else 'DISABLED'}")
        
    def stop_recording(self):
        """Stop current recording session"""
        if not self.recording:
            print("Not currently recording!")
            return
        
        print(f"Finalizing {len(self.active_events)} active events...")
        for event in self.active_events:
            if not event.snippet_complete:
                if event.has_culprit:
                    self.finalize_event_snippet(event)
                else:
                    if hasattr(event, 'writer') and event.writer is not None:
                        event.writer.release()
                    if event.snippet_path and event.snippet_path.exists():
                        event.snippet_path.unlink()
        
        self.recording = False
        
        if self.event_log_file:
            self.event_log_file.close()
            self.event_log_file = None
            
        print(f"\nSession {self.session_id} stopped")
        print(f"Total litter events detected: {self.total_events}")
        print(f"Events without face/plate: {self.events_without_culprit}")
        print(f"False positives filtered: {self.false_positives_filtered}")
        print(f"Event snippets saved: {len(self.completed_events)}")
        if self.completed_events:
            print(f"Snippets location: {EVENTS_DIR}")
        
    def create_event_snippet(self, event):
        """Create video snippet for a litter event"""
        snippet_filename = f"event_{self.session_id}_{event.event_id:03d}.mp4"
        snippet_path = EVENTS_DIR / snippet_filename
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(snippet_path), fourcc, FPS, FRAME_SIZE)
        
        if not writer.isOpened():
            print(f"ERROR: Could not create snippet writer for event {event.event_id}")
            return None
            
        event.snippet_path = snippet_path
        return writer
        
    def log_event(self, obj, frame):
        """Log a litter detection event and check for culprit"""
        self.total_events += 1
        event_id = self.total_events
        now = datetime.now()
        
        x1, y1, x2, y2 = obj.bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Find associated face or plate
        associated_face, associated_plate = self.find_associated_culprit(obj)
        
        # Create event object
        event = LitterEvent(event_id, self.frame_num, obj, now)
        event.associated_face = associated_face
        event.associated_plate = associated_plate
        event.has_culprit = (associated_face is not None or associated_plate is not None)
        
        if not event.has_culprit:
            self.events_without_culprit += 1
            print(f"[FILTERED - NO CULPRIT] Event {event_id} at frame {self.frame_num}")
            print(f"  No face or license plate associated with litter")
            obj.reported = True
            return
        
        self.active_events.append(event)
        
        # Create video writer
        writer = self.create_event_snippet(event)
        if writer is None:
            return
            
        culprit_type = "FACE" if associated_face else "PLATE"
        print(f"[LITTER EVENT {event_id}] Frame {self.frame_num}: {culprit_type} DETECTED")
        print(f"  Writing snippet with {len(self.frame_buffer)} pre-event frames")
        
        for i, buffered_frame in enumerate(self.frame_buffer):
            if EXPORT_WITH_BOUNDING_BOXES and i < len(self.detection_buffer):
                frame_to_write = self.apply_detections_from_buffer(buffered_frame, i)
            else:
                frame_to_write = buffered_frame
            writer.write(frame_to_write)
        
        event.writer = writer
        obj.reported = True
        
        print(f"  Location: ({x1},{y1}), Size: {width}x{height}")
        print(f"  Recording 10 sec before + 10 sec after...")
    
    def apply_detections_from_buffer(self, frame, buffer_index):
        """Apply stored detection data to a buffered frame"""
        if buffer_index >= len(self.detection_buffer):
            return frame
        
        detection_data = self.detection_buffer[buffer_index]
        frame_copy = frame.copy()
        
        for face_bbox in detection_data['faces']:
            x1, y1, x2, y2 = face_bbox
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame_copy, "FACE", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        for plate_bbox in detection_data['plates']:
            x1, y1, x2, y2 = plate_bbox
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame_copy, "PLATE", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        for obj_data in detection_data['objects']:
            x1, y1, x2, y2 = obj_data['bbox']
            color = obj_data['color']
            label = obj_data['label']
            
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_copy, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame_copy
        
    def finalize_event_snippet(self, event):
        """Finalize and close an event snippet"""
        if hasattr(event, 'writer') and event.writer is not None:
            event.writer.release()
            event.snippet_complete = True
            self.completed_events.append(event)
            self.event_snippets.append(event.snippet_path)
            
            x1, y1, x2, y2 = event.obj.bbox
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            self.event_log_writer.writerow([
                event.event_id,
                self.session_id,
                event.timestamp.isoformat(),
                event.detection_frame,
                x1, y1, x2, y2,
                width, height, area,
                event.obj.static,
                f"{event.obj.bg_diff_ratio:.3f}",
                "Yes" if event.associated_face else "No",
                "Yes" if event.associated_plate else "No",
                str(event.snippet_path.name)
            ])
            
            print(f"  ✓ Event {event.event_id} snippet complete: {event.snippet_path.name}")
        
    def start_playback(self):
        """Start playback of event snippets"""
        if not self.event_snippets:
            print("No event snippets available!")
            return
            
        if self.recording:
            print("Cannot start playback while recording.")
            return
        
        self.current_playback_index = 0
        self.load_event_snippet(0)
        
    def load_event_snippet(self, index):
        """Load specific event snippet"""
        if index < 0 or index >= len(self.event_snippets):
            return False
            
        snippet_path = self.event_snippets[index]
        
        if self.play_cap:
            self.play_cap.release()
            
        self.play_cap = cv2.VideoCapture(str(snippet_path))
        
        if not self.play_cap.isOpened():
            print(f"ERROR: Could not open {snippet_path}")
            return False
            
        self.playback = True
        self.play_frame = 0
        self.play_paused = False
        self.last_playback_frame = None
        self.current_playback_index = index
        self.last_frame_time = time.time()
        
        print(f"\nPlaying event {index + 1}/{len(self.event_snippets)}: {snippet_path.name}")
        return True
        
    def next_event_snippet(self):
        """Play next event snippet"""
        next_index = (self.current_playback_index + 1) % len(self.event_snippets)
        self.load_event_snippet(next_index)
        
    def previous_event_snippet(self):
        """Play previous event snippet"""
        prev_index = (self.current_playback_index - 1) % len(self.event_snippets)
        self.load_event_snippet(prev_index)
        
    def stop_playback(self):
        """Stop playback"""
        if not self.playback:
            return
            
        if self.play_cap:
            self.play_cap.release()
            self.play_cap = None
        self.playback = False
        print("Playback stopped")
        
    def process_frame(self, frame):
        """Process frame for litter detection"""
        self.frame_buffer.append(frame.copy())
        
        face_detections = self.detect_faces(frame)
        plate_detections = self.detect_plates(frame)
        
        self.update_face_tracking(face_detections)
        self.update_plate_tracking(plate_detections)
        
        detection_data = {
            'faces': [face.bbox for face in self.tracked_faces],
            'plates': [plate.bbox for plate in self.tracked_plates],
            'objects': []
        }
        
        if not self.bg_learned:
            self.learn_baseline_background(frame)
            self.detection_buffer.append(detection_data)
            return None
            
        self.frames_since_last_relearn += 1
        if self.frames_since_last_relearn >= RELEARN_INTERVAL_FRAMES:
            self.adaptive_background_update(frame)
            self.frames_since_last_relearn = 0
        
        blurred = cv2.GaussianBlur(frame, (21, 21), 0)
        fg_mask = self.bg_subtractor.apply(blurred, learningRate=0.001)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if MIN_AREA < area < MAX_AREA:
                x, y, w, h = cv2.boundingRect(c)
                detections.append((x, y, x + w, y + h))
        
        self.update_tracking(detections, frame)
        
        for obj in self.objects:
            x1, y1, x2, y2 = obj.bbox
            
            if obj.reported:
                color = (0, 255, 0)
                label = "LITTER"
            elif obj.evaluated and not obj.is_new_object:
                color = (128, 128, 128)
                label = "Filtered"
            elif obj.static >= STATIC_FRAMES:
                color = (0, 165, 255)
                label = "Evaluating"
            else:
                color = (0, 0, 255)
                label = f"Tracking: {obj.static}"
            
            detection_data['objects'].append({
                'bbox': (x1, y1, x2, y2),
                'color': color,
                'label': label
            })
        
        self.detection_buffer.append(detection_data)
        
        for event in self.active_events[:]:
            if hasattr(event, 'writer') and event.writer is not None:
                if EXPORT_WITH_BOUNDING_BOXES:
                    frame_to_write = self.draw_bounding_boxes_on_frame(frame)
                else:
                    frame_to_write = frame
                event.writer.write(frame_to_write)
                
                frames_after = self.frame_num - event.detection_frame
                if frames_after >= AFTER_EVENT_FRAMES:
                    self.finalize_event_snippet(event)
                    self.active_events.remove(event)
        
        return fg_mask
        
    def update_tracking(self, detections, frame):
        """Update object tracking"""
        matched_objects = set()
        
        for det in detections:
            best_obj, best_score = None, 0
            
            for obj in self.objects:
                score = iou(det, obj.bbox)
                if score > best_score:
                    best_obj, best_score = obj, score
                    
            if best_score > OVERLAP_THRESH:
                best_obj.update(det, self.frame_num)
                matched_objects.add(best_obj)
            else:
                new_obj = TrackedObject(det, self.frame_num)
                self.objects.append(new_obj)
                matched_objects.add(new_obj)
        
        for obj in self.objects[:]:
            if self.frame_num - obj.last > 30:
                self.objects.remove(obj)
                continue
                
            if obj not in matched_objects:
                continue
            
            if obj.static >= STATIC_FRAMES and not obj.evaluated:
                obj.evaluated = True
                obj.is_new_object = self.is_truly_new_object(obj, frame)
                
                if obj.is_new_object:
                    if not obj.reported:
                        self.log_event(obj, frame)
                else:
                    self.false_positives_filtered += 1
                    print(f"[FILTERED] Frame {self.frame_num}: Motion settled")
        
    def draw_overlays(self, frame):
        """Draw UI overlays"""
        now = datetime.now()
        
        # Draw semi-transparent border to indicate capture region
        overlay = frame.copy()
        border_thickness = 3
        cv2.rectangle(overlay, (0, 0), (FRAME_SIZE[0]-1, FRAME_SIZE[1]-1), (0, 255, 0), border_thickness)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        if self.playback:
            total = int(self.play_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            status = f"EVENT {self.current_playback_index + 1}/{len(self.event_snippets)} | Frame {self.play_frame}/{total}"
            color = (255, 165, 0)
        elif self.recording:
            if self.paused:
                status = "PAUSED"
                color = (0, 165, 255)
            elif not self.bg_learned:
                status = f"LEARNING BG ({self.bg_learning_counter}/{BACKGROUND_LEARN_FRAMES})"
                color = (0, 255, 255)
            else:
                status = f"MONITORING | Faces: {len(self.tracked_faces)} | Plates: {len(self.tracked_plates)}"
                color = (0, 0, 255)
        else:
            status = "STANDBY - Press R to Start"
            color = (255, 255, 255)
            
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, now.strftime('%Y-%m-%d %H:%M:%S'),
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if self.recording and not self.playback:
            cv2.putText(frame, f"Events: {self.total_events} | No Culprit: {self.events_without_culprit}",
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            for face in self.tracked_faces:
                x1, y1, x2, y2 = face.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, "FACE", (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            for plate in self.tracked_plates:
                x1, y1, x2, y2 = plate.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, "PLATE", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            for obj in self.objects:
                x1, y1, x2, y2 = obj.bbox
                
                if obj.reported:
                    color = (0, 255, 0)
                    label = "LITTER"
                elif obj.evaluated and not obj.is_new_object:
                    color = (128, 128, 128)
                    label = "Filtered"
                elif obj.static >= STATIC_FRAMES:
                    color = (0, 165, 255)
                    label = "Evaluating"
                else:
                    color = (0, 0, 255)
                    label = f"Tracking: {obj.static}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
        
    def show_help(self):
        """Display help"""
        print("\n" + "="*70)
        print("LITTER DETECTION - SCREEN CAPTURE MODE")
        print("="*70)
        print("\nFEATURES:")
        print("- Captures from specified screen coordinates")
        print("- Detects litter events in captured region")
        print("- Identifies human faces and license plates")
        print("- Only saves events with identified culprit (face or plate)")
        print("- Tracks hand-to-litter and car-to-litter movement")
        print(f"- Bounding box export: {'ENABLED' if EXPORT_WITH_BOUNDING_BOXES else 'DISABLED'}")
        print("\nSETUP:")
        print(f"- Capture Region: X={WINDOW_X}, Y={WINDOW_Y}, Size={FRAME_SIZE[0]}x{FRAME_SIZE[1]}")
        print("- Position your video player or content at these coordinates")
        print("- The OpenCV window shows what's being captured (with green border)")
        print("- Works with video players, live streams, etc.")
        print("\nCONTROLS:")
        print("  R - Start Monitoring")
        print("  S - Stop Monitoring")
        print("  L - Play Event Snippets")
        print("  N - Next Event (during playback)")
        print("  SPACE - Pause/Resume (during playback)")
        print("  Q - Stop Playback")
        print("  H - Show Help")
        print("  ESC - Exit")
        print("="*70 + "\n")
        
    def run(self):
        """Main loop"""
        if not self.initialize_capture():
            return
            
        self.show_help()
        
        existing = sorted(EVENTS_DIR.glob("event_*.mp4"))
        self.event_snippets = existing
        if existing:
            print(f"Found {len(existing)} existing snippets")
        
        # Create window with transparency support
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow(self.window_name, WINDOW_X + FRAME_SIZE[0] + 50, WINDOW_Y)
        cv2.resizeWindow(self.window_name, FRAME_SIZE[0], FRAME_SIZE[1])
        
        print("\nWindow created - you can move it to see both windows")
        print("The content shown is being captured from the screen region")
        
        while self.running:
            try:
                if self.playback:
                    if not self.play_paused:
                        current_time = time.time()
                        elapsed = current_time - self.last_frame_time
                        frame_duration = 1.0 / FPS
                        
                        if elapsed >= frame_duration:
                            ret, frame = self.play_cap.read()
                            if not ret:
                                if self.current_playback_index < len(self.event_snippets) - 1:
                                    self.next_event_snippet()
                                else:
                                    self.stop_playback()
                                continue
                            self.last_playback_frame = frame.copy()
                            self.play_frame += 1
                            self.last_frame_time = current_time
                        else:
                            frame = self.last_playback_frame.copy() if self.last_playback_frame is not None else None
                            if frame is None:
                                time.sleep(0.001)
                                continue
                    else:
                        frame = self.last_playback_frame.copy() if self.last_playback_frame is not None else None
                        if frame is None:
                            continue
                else:
                    frame = self.capture_frame()
                    if frame is None:
                        continue
                    
                frame = cv2.resize(frame, FRAME_SIZE)
                
                if self.recording and not self.paused and not self.playback:
                    self.frame_num += 1
                    self.process_frame(frame)
                
                frame = self.draw_overlays(frame)
                cv2.imshow(self.window_name, frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:
                    self.running = False
                elif key == ord('r') or key == ord('R'):
                    if not self.recording and not self.playback:
                        self.start_recording()
                elif key == ord('s') or key == ord('S'):
                    if self.recording:
                        self.stop_recording()
                elif key == ord('p') or key == ord('P'):
                    if self.playback:
                        self.play_paused = not self.play_paused
                    elif self.recording:
                        self.paused = not self.paused
                elif key == ord('l') or key == ord('L'):
                    if not self.playback and not self.recording:
                        self.start_playback()
                elif key == ord('q') or key == ord('Q'):
                    if self.playback:
                        self.stop_playback()
                elif key == 32:
                    if self.playback:
                        self.play_paused = not self.play_paused
                elif key == ord('n') or key == ord('N'):
                    if self.playback:
                        self.next_event_snippet()
                elif key == ord('h') or key == ord('H'):
                    self.show_help()
                    
            except Exception as e:
                print(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        self.cleanup()
        
    def cleanup(self):
        """Cleanup"""
        if self.recording:
            self.stop_recording()
        if self.playback:
            self.stop_playback()
        cv2.destroyAllWindows()
        print("Program terminated")


def main():
    try:
        detector = LitterDetector()
        detector.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
