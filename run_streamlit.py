#!/usr/bin/env python3
"""
Simple script to run Streamlit app with proper path handling
"""

import os
import sys
import subprocess

def main():
    """Run Streamlit app from correct directory."""
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the src directory
    src_dir = os.path.join(current_dir, "deepfake_final_submission", "src")
    
    if not os.path.exists(src_dir):
        print(f"Error: Directory not found: {src_dir}")
        return 1
    
    # Change to src directory
    os.chdir(src_dir)
    
    print(f"Running Streamlit from: {os.getcwd()}")
    print("Starting web interface...")
    print("Opening browser at: http://localhost:8501")
    
    try:
        # Run streamlit with the fixed app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app_fixed.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false"
        ], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        return 1
    except FileNotFoundError:
        print("Error: Streamlit not installed. Please run: pip install streamlit")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
