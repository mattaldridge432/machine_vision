#!/usr/bin/env python3
"""
Control Client for Litter Detection System
Send commands to modify detector settings in real-time
"""

import socket
import json
import sys


class LitterDetectorControl:
    """Client for controlling the litter detection system"""
    
    def __init__(self, host='localhost', port=5555):
        self.host = host
        self.port = port
        
    def send_command(self, command_dict):
        """Send a command to the detector"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.host, self.port))
                s.sendall(json.dumps(command_dict).encode('utf-8'))
                response = s.recv(1024).decode('utf-8')
                print(f"Response: {response.strip()}")
                return True
        except ConnectionRefusedError:
            print("Error: Cannot connect to detector. Is it running?")
            return False
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def toggle_bounding_boxes(self, show=True):
        """Toggle bounding boxes in display"""
        return self.send_command({'show_bounding_boxes': show})
    
    def toggle_export_boxes(self, include=True):
        """Toggle bounding boxes in exported videos"""
        return self.send_command({'export_with_boxes': include})
    
    def set_motion_threshold(self, threshold):
        """Set motion detection threshold (0-255)"""
        return self.send_command({'motion_threshold': threshold})
    
    def set_min_contour_area(self, area):
        """Set minimum contour area for detection"""
        return self.send_command({'min_contour_area': area})
    
    def toggle_debug_info(self, show=True):
        """Toggle debug information display"""
        return self.send_command({'show_debug_info': show})
    
    def interactive_mode(self):
        """Interactive control mode"""
        print("=== Litter Detector Control ===")
        print("Commands:")
        print("  1. Toggle bounding boxes (on/off)")
        print("  2. Toggle export boxes (on/off)")
        print("  3. Set motion threshold (0-255)")
        print("  4. Set min contour area")
        print("  5. Toggle debug info (on/off)")
        print("  q. Quit")
        print()
        
        while True:
            choice = input("Enter command: ").strip().lower()
            
            if choice == 'q':
                break
            elif choice == '1':
                state = input("Show boxes? (y/n): ").lower() == 'y'
                self.toggle_bounding_boxes(state)
            elif choice == '2':
                state = input("Export with boxes? (y/n): ").lower() == 'y'
                self.toggle_export_boxes(state)
            elif choice == '3':
                threshold = int(input("Motion threshold (0-255): "))
                self.set_motion_threshold(threshold)
            elif choice == '4':
                area = int(input("Min contour area: "))
                self.set_min_contour_area(area)
            elif choice == '5':
                state = input("Show debug? (y/n): ").lower() == 'y'
                self.toggle_debug_info(state)
            else:
                print("Invalid command")


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        # Command line mode
        controller = LitterDetectorControl()
        
        if sys.argv[1] == 'boxes-on':
            controller.toggle_bounding_boxes(True)
        elif sys.argv[1] == 'boxes-off':
            controller.toggle_bounding_boxes(False)
        elif sys.argv[1] == 'export-boxes-on':
            controller.toggle_export_boxes(True)
        elif sys.argv[1] == 'export-boxes-off':
            controller.toggle_export_boxes(False)
        elif sys.argv[1] == 'threshold':
            controller.set_motion_threshold(int(sys.argv[2]))
        elif sys.argv[1] == 'debug-on':
            controller.toggle_debug_info(True)
        elif sys.argv[1] == 'debug-off':
            controller.toggle_debug_info(False)
        else:
            print(f"Unknown command: {sys.argv[1]}")
    else:
        # Interactive mode
        controller = LitterDetectorControl()
        controller.interactive_mode()


if __name__ == "__main__":
    main()
