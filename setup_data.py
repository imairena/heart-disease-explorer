"""
Setup Script: Copy Heart Disease Data Files

This utility script helps set up the project by copying raw data files
from the Downloads folder to the project's data directory.

When you download the UCI Heart Disease dataset, it typically comes as a
folder named 'heart+disease' in your Downloads directory. This script
copies the necessary data files from there to the project's data/ folder.

Usage:
    python setup_data.py

After running this, you can run data_cleaning.py to process the raw data.
"""

import shutil
from pathlib import Path

def main():
    """
    Main function to copy data files from Downloads to project data folder.
    
    This function:
    1. Defines source directory (Downloads/heart+disease)
    2. Defines destination directory (project/data/)
    3. Creates destination directory if it doesn't exist
    4. Copies each data file if it exists in the source
    5. Prints status messages for each file copied
    """
    # Source: Where the downloaded data files are located
    # Typically in ~/Downloads/heart+disease/ after downloading from UCI
    src = Path.home() / 'Downloads' / 'heart+disease'
    
    # Destination: Where we want to copy the files (project's data folder)
    # Path(__file__).parent gets the directory containing this script
    dst = Path(__file__).parent / 'data'
    
    # Check if source directory exists
    if not src.exists():
        print(f"Source not found: {src}")
        print("Please download the heart disease dataset and place it in your Downloads folder.")
        return
    
    # Create destination directory if it doesn't exist
    # exist_ok=True means don't raise an error if it already exists
    dst.mkdir(exist_ok=True)
    
    # List of data files we need to copy
    # These are the 4 processed data files from different locations
    data_files = [
        'processed.cleveland.data',      # Cleveland Clinic Foundation data
        'processed.hungarian.data',      # Hungarian Institute of Cardiology data
        'processed.switzerland.data',    # University Hospital, Zurich data
        'processed.va.data'              # V.A. Medical Center, Long Beach data
    ]
    
    # Copy each file if it exists in the source directory
    for f in data_files:
        src_file = src / f  # Full path to source file
        
        # Only copy if the file actually exists
        if src_file.exists():
            shutil.copy(src_file, dst / f)  # Copy from source to destination
            print(f"Copied {f}")
        else:
            print(f"Warning: {f} not found in {src}")
    
    # Print completion message with next steps
    print("Done! Run: python data_cleaning.py")

if __name__ == '__main__':
    # Only run main() if this script is executed directly
    # (not if it's imported as a module)
    main()
