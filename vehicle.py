#!/usr/bin/env python3
"""
Advanced Vehicle Litter Detection System
Detects litter thrown from vehicles using computer vision
Optimized for Raspberry Pi 4 and low-spec hardware
"""

import cv2
import numpy as np
import json
import socket
import threading
import queue
from datetime import datetime
from collections import deque
from pathlib import Path
import time


class LitterDetectionConfig:
    """Configuration management for the detection system"""
    
    def __init__(self):
        # Video settings
        self.camera_index = 0
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 15  # Lower FPS for efficiency
        
        # Detection parameters
        self.motion_threshold = 25
        self.min_contour_area = 100
        self.max_contour_area = 5000
        self.blur_kernel = (5, 5)
        
        # Tracking parameters
        self.max_tracking_distance = 50
        self.min_litter_lifetime = 15  # frames
        self.vehicle_exit_buffer = 10  # frames after vehicle exits
        
        # Buffer settings
        self.pre_event_seconds = 10
        self.post_event_seconds = 10
        
        # Display settings
        self.show_bounding_boxes = True
        self.show_motion_mask = False
        self.show_debug_info = True
        
        # Export settings
        self.export_with_boxes = True
        self.output_directory = "litter_events"
        self.video_codec = "mp4v"
        
        # Socket control
        self.control_port = 5555
        self.control_enabled = True


class MovingObject:
    """Represents a tracked moving object (vehicle or litter)"""
    
    def __init__(self, contour, frame_id, position):
        self.contour = contour
        self.positions = deque(maxlen=30)
        self.positions.append(position)
        self.first_seen = frame_id
        self.last_seen = frame_id
        self.area = cv2.contourArea(contour)
        self.is_vehicle = self.area > 2000  # Larger objects are vehicles
        self.associated_vehicle_id = None
        self.remained_after_vehicle = False
        
    def update(self, contour, frame_id, position):
        self.contour = contour
        self.positions.append(position)
        self.last_seen = frame_id
        self.area = cv2.contourArea(contour)
        
    def get_center(self):
        return self.positions[-1] if self.positions else (0, 0)
    
    def lifetime(self):
        return self.last_seen - self.first_seen
    
    def get_velocity(self):
        if len(self.positions) < 2:
            return (0, 0)
        p1 = self.positions[-1]
        p2 = self.positions[0]
        return (p1[0] - p2[0], p1[1] - p2[1])


class LitterEvent:
    """Represents a detected littering event"""
    
    def __init__(self, litter_obj, vehicle_obj, detection_frame):
        self.litter_id = id(litter_obj)
        self.vehicle_id = id(vehicle_obj)
        self.detection_time = datetime.now()
        self.detection_frame = detection_frame
        self.litter_position = litter_obj.get_center()
        self.vehicle_exit_frame = None
        self.confirmed = False
        
    def to_dict(self):
        return {
            "event_id": f"LITTER_{self.detection_time.strftime('%Y%m%d_%H%M%S')}_{self.litter_id}",
            "timestamp": self.detection_time.isoformat(),
            "detection_frame": self.detection_frame,
            "vehicle_exit_frame": self.vehicle_exit_frame,
            "litter_position": self.litter_position,
            "confirmed": self.confirmed
        }


class FrameBuffer:
    """Circular buffer for storing video frames"""
    
    def __init__(self, max_frames):
        self.buffer = deque(maxlen=max_frames)
        self.frame_ids = deque(maxlen=max_frames)
        
    def add(self, frame, frame_id):
        self.buffer.append(frame.copy())
        self.frame_ids.append(frame_id)
        
    def get_frames_range(self, start_id, end_id):
        frames = []
        for i, fid in enumerate(self.frame_ids):
            if start_id <= fid <= end_id:
                frames.append(self.buffer[i])
        return frames


class LitterDetector:
    """Main litter detection system"""
    
    def __init__(self, config=None):
        self.config = config or LitterDetectionConfig()
        self.running = False
        self.frame_count = 0
        
        # Create output directory
        Path(self.config.output_directory).mkdir(exist_ok=True)
        
        # Video capture
        self.cap = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        
        # Frame buffering
        buffer_size = int((self.config.pre_event_seconds + self.config.post_event_seconds) * self.config.fps)
        self.frame_buffer = FrameBuffer(buffer_size)
        
        # Tracking
        self.tracked_objects = {}
        self.next_object_id = 0
        self.active_events = []
        self.completed_events = queue.Queue()
        
        # Socket control
        self.control_socket = None
        self.control_thread = None
        
        # Performance monitoring
        self.processing_times = deque(maxlen=30)
        
    def initialize_camera(self):
        """Initialize camera with optimal settings"""
        self.cap = cv2.VideoCapture(self.config.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
            
    def start_control_socket(self):
        """Start socket server for receiving control commands"""
        if not self.config.control_enabled:
            return
            
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.control_socket.bind(('localhost', self.config.control_port))
        self.control_socket.listen(1)
        self.control_socket.settimeout(1.0)
        
        self.control_thread = threading.Thread(target=self._control_listener, daemon=True)
        self.control_thread.start()
        print(f"Control socket listening on port {self.config.control_port}")
        
    def _control_listener(self):
        """Listen for control commands via socket"""
        while self.running:
            try:
                conn, addr = self.control_socket.accept()
                with conn:
                    data = conn.recv(1024).decode('utf-8')
                    if data:
                        self._process_control_command(data)
                        conn.sendall(b"OK\n")
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Control socket error: {e}")
                
    def _process_control_command(self, command_str):
        """Process incoming control commands"""
        try:
            cmd = json.loads(command_str)
            
            if 'show_bounding_boxes' in cmd:
                self.config.show_bounding_boxes = bool(cmd['show_bounding_boxes'])
            if 'export_with_boxes' in cmd:
                self.config.export_with_boxes = bool(cmd['export_with_boxes'])
            if 'motion_threshold' in cmd:
                self.config.motion_threshold = int(cmd['motion_threshold'])
            if 'min_contour_area' in cmd:
                self.config.min_contour_area = int(cmd['min_contour_area'])
            if 'show_debug_info' in cmd:
                self.config.show_debug_info = bool(cmd['show_debug_info'])
                
            print(f"Applied command: {cmd}")
        except json.JSONDecodeError:
            print(f"Invalid command format: {command_str}")
            
    def detect_motion(self, frame):
        """Detect motion and return contours of moving objects"""
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Threshold and morphological operations
        _, thresh = cv2.threshold(fg_mask, self.config.motion_threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours, thresh
        
    def track_objects(self, contours, frame_id):
        """Track detected objects across frames"""
        current_positions = []
        
        # Process each detected contour
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.config.min_contour_area < area < self.config.max_contour_area * 10:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    current_positions.append((contour, (cx, cy), area))
        
        # Match with existing tracked objects
        matched_ids = set()
        
        for contour, position, area in current_positions:
            best_match_id = None
            best_distance = self.config.max_tracking_distance
            
            for obj_id, obj in self.tracked_objects.items():
                if obj_id in matched_ids:
                    continue
                    
                obj_pos = obj.get_center()
                distance = np.sqrt((position[0] - obj_pos[0])**2 + (position[1] - obj_pos[1])**2)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match_id = obj_id
            
            if best_match_id is not None:
                # Update existing object
                self.tracked_objects[best_match_id].update(contour, frame_id, position)
                matched_ids.add(best_match_id)
            else:
                # Create new tracked object
                new_obj = MovingObject(contour, frame_id, position)
                self.tracked_objects[self.next_object_id] = new_obj
                self.next_object_id += 1
        
        # Remove stale objects
        stale_ids = [obj_id for obj_id, obj in self.tracked_objects.items() 
                     if frame_id - obj.last_seen > 30]
        for obj_id in stale_ids:
            del self.tracked_objects[obj_id]
            
    def detect_litter_events(self, frame_id):
        """Detect potential littering events"""
        vehicles = {oid: obj for oid, obj in self.tracked_objects.items() if obj.is_vehicle}
        potential_litter = {oid: obj for oid, obj in self.tracked_objects.items() if not obj.is_vehicle}
        
        for litter_id, litter_obj in potential_litter.items():
            # Skip if already associated with an event
            if any(event.litter_id == id(litter_obj) for event in self.active_events):
                continue
            
            litter_pos = litter_obj.get_center()
            
            # Find nearby vehicles
            for vehicle_id, vehicle_obj in vehicles.items():
                vehicle_pos = vehicle_obj.get_center()
                distance = np.sqrt((litter_pos[0] - vehicle_pos[0])**2 + 
                                 (litter_pos[1] - vehicle_pos[1])**2)
                
                # Check if litter appeared near vehicle
                if distance < 150 and litter_obj.lifetime() < 10:
                    litter_obj.associated_vehicle_id = vehicle_id
                    event = LitterEvent(litter_obj, vehicle_obj, frame_id)
                    self.active_events.append(event)
                    print(f"Potential litter event detected at frame {frame_id}")
                    break
                    
    def confirm_litter_events(self, frame_id):
        """Confirm littering events when vehicle has left and litter remains"""
        confirmed_events = []
        
        for event in self.active_events:
            if event.confirmed:
                continue
                
            # Find the litter object
            litter_obj = None
            for obj in self.tracked_objects.values():
                if id(obj) == event.litter_id:
                    litter_obj = obj
                    break
            
            if litter_obj is None:
                continue
            
            # Check if associated vehicle has left
            vehicle_present = any(id(obj) == event.vehicle_id for obj in self.tracked_objects.values())
            
            if not vehicle_present and not event.vehicle_exit_frame:
                event.vehicle_exit_frame = frame_id
                
            # Confirm if litter remains after vehicle left
            if event.vehicle_exit_frame and \
               (frame_id - event.vehicle_exit_frame) > self.config.vehicle_exit_buffer and \
               litter_obj.lifetime() > self.config.min_litter_lifetime:
                
                event.confirmed = True
                confirmed_events.append(event)
                print(f"✓ LITTER EVENT CONFIRMED at frame {frame_id}")
        
        # Move confirmed events to export queue
        for event in confirmed_events:
            self.completed_events.put(event)
            self.active_events.remove(event)
            self._export_event(event, frame_id)
            
    def _export_event(self, event, current_frame):
        """Export video clip and event report"""
        start_frame = max(0, event.detection_frame - int(self.config.pre_event_seconds * self.config.fps))
        end_frame = current_frame + int(self.config.post_event_seconds * self.config.fps)
        
        # Get frames from buffer
        frames = self.frame_buffer.get_frames_range(start_frame, end_frame)
        
        if not frames:
            print("Warning: No frames available for export")
            return
        
        # Create output filename
        timestamp = event.detection_time.strftime('%Y%m%d_%H%M%S')
        video_path = Path(self.config.output_directory) / f"litter_event_{timestamp}.mp4"
        report_path = Path(self.config.output_directory) / f"litter_event_{timestamp}.json"
        
        # Export video
        fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
        out = cv2.VideoWriter(str(video_path), fourcc, self.config.fps,
                            (self.config.frame_width, self.config.frame_height))
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        # Export report
        report = event.to_dict()
        report['video_file'] = str(video_path)
        report['total_frames'] = len(frames)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Event exported: {video_path}")
        
    def draw_display(self, frame, motion_mask):
        """Draw visualization overlay on display frame"""
        display = frame.copy()
        
        # Draw tracked objects
        for obj_id, obj in self.tracked_objects.items():
            x, y = obj.get_center()
            color = (0, 255, 0) if obj.is_vehicle else (0, 165, 255)
            
            if self.config.show_bounding_boxes:
                cv2.drawContours(display, [obj.contour], -1, color, 2)
                
            # Draw tracking point
            cv2.circle(display, (x, y), 4, color, -1)
            
            # Label
            label = f"V{obj_id}" if obj.is_vehicle else f"L{obj_id}"
            cv2.putText(display, label, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1)
        
        # Draw active events
        for event in self.active_events:
            status = "CONFIRMED" if event.confirmed else "DETECTING"
            color = (0, 255, 0) if event.confirmed else (0, 255, 255)
            
            cv2.putText(display, f"⚠ {status}", event.litter_position, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw debug info
        if self.config.show_debug_info:
            fps = len(self.processing_times) / sum(self.processing_times) if self.processing_times else 0
            info_text = [
                f"Frame: {self.frame_count}",
                f"FPS: {fps:.1f}",
                f"Objects: {len(self.tracked_objects)}",
                f"Events: {len(self.active_events)}",
                f"Confirmed: {self.completed_events.qsize()}"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(display, text, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show motion mask if enabled
        if self.config.show_motion_mask:
            motion_rgb = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
            display = cv2.addWeighted(display, 0.7, motion_rgb, 0.3, 0)
        
        return display
        
    def run(self):
        """Main detection loop"""
        self.running = True
        
        try:
            self.initialize_camera()
            self.start_control_socket()
            
            print("Litter Detection System Started")
            print(f"Press 'q' to quit, 's' to toggle settings")
            
            while self.running:
                start_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Store frame in buffer
                self.frame_buffer.add(frame, self.frame_count)
                
                # Detect motion
                contours, motion_mask = self.detect_motion(frame)
                
                # Track objects
                self.track_objects(contours, self.frame_count)
                
                # Detect and confirm litter events
                self.detect_litter_events(self.frame_count)
                self.confirm_litter_events(self.frame_count)
                
                # Draw display
                display_frame = self.draw_display(frame, motion_mask)
                
                # Show window
                cv2.imshow('Litter Detection System', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.config.show_bounding_boxes = not self.config.show_bounding_boxes
                elif key == ord('d'):
                    self.config.show_debug_info = not self.config.show_debug_info
                elif key == ord('m'):
                    self.config.show_motion_mask = not self.config.show_motion_mask
                
                # Performance tracking
                elapsed = time.time() - start_time
                self.processing_times.append(elapsed)
                self.frame_count += 1
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        if self.control_socket:
            self.control_socket.close()
        
        print("Cleanup complete")


def main():
    """Main entry point"""
    config = LitterDetectionConfig()
    detector = LitterDetector(config)
    detector.run()


if __name__ == "__main__":
    main()
