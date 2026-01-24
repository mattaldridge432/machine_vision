#!/usr/bin/env python3
"""
Machine Vision System Control Client
Connect and send commands to the vision system via socket.
"""

import socket
import sys

class VisionClient:
    """Client for controlling the vision system"""
    
    def __init__(self, host='127.0.0.1', port=5555):
        self.host = host
        self.port = port
        self.sock = None
    
    def connect(self):
        """Connect to vision system"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            print(f"Connected to vision system at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def send_command(self, command):
        """Send command and get response"""
        try:
            self.sock.send((command + '\n').encode('utf-8'))
            response = self.sock.recv(8192).decode('utf-8')
            return response.strip()
        except Exception as e:
            return f"ERROR: {e}"
    
    def close(self):
        """Close connection"""
        if self.sock:
            self.sock.close()
    
    def interactive_mode(self):
        """Run interactive command prompt"""
        print("\n" + "="*60)
        print("VISION SYSTEM CONTROL CLIENT - Interactive Mode")
        print("="*60)
        print("Type 'help' for command list, 'quit' to exit")
        print("="*60 + "\n")
        
        while True:
            try:
                command = input("vision> ").strip()
                
                if not command:
                    continue
                
                if command.lower() in ['quit', 'exit', 'q']:
                    print("Disconnecting...")
                    break
                
                response = self.send_command(command)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\nInterrupted")
                break
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main entry point"""
    
    # Parse command line arguments
    host = '127.0.0.1'
    port = 5555
    command = None
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("""
Vision System Control Client

Usage:
  python vision_client.py                 - Interactive mode
  python vision_client.py <command>       - Send single command
  python vision_client.py -h              - Show this help

Examples:
  python vision_client.py                 - Start interactive session
  python vision_client.py "enable litter" - Enable litter detection
  python vision_client.py status          - Get system status
  python vision_client.py help            - Show vision system help
            """)
            return
        else:
            command = ' '.join(sys.argv[1:])
    
    # Create client
    client = VisionClient(host, port)
    
    if not client.connect():
        print("\nMake sure the vision system is running first!")
        print("Start it with: python vision_system.py")
        return
    
    try:
        if command:
            # Single command mode
            response = client.send_command(command)
            print(response)
        else:
            # Interactive mode
            client.interactive_mode()
    finally:
        client.close()

if __name__ == '__main__':
    main()
