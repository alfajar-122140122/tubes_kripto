"""
Script to fix ZBar DLL issues on Windows
This script helps to locate and fix issues with the ZBar DLL required for QR code reading
"""

import os
import sys
import shutil
import platform
import site
import glob
import requests
import tempfile
import zipfile
import ctypes
from pathlib import Path

def download_zbar(target_dir):
    """
    Download ZBar DLLs for Windows
    """
    print("Downloading ZBar binaries...")
    
    # URL for the precompiled Windows binaries
    url = "https://sourceforge.net/projects/zbar/files/zbar/0.10/zbar-0.10-setup.exe/download"
    
    # Download the file
    with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as tmp:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Write the file
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            
            tmp_path = tmp.name
        except Exception as e:
            print(f"Error downloading ZBar: {e}")
            return False
    
    # Extract DLLs from the executable
    try:
        with tempfile.TemporaryDirectory() as extract_dir:
            # Extract using 7-zip if available
            try:
                import subprocess
                if platform.system() == 'Windows':
                    # Try to find 7-zip
                    seven_zip = r"C:\Program Files\7-Zip\7z.exe"
                    if os.path.exists(seven_zip):
                        subprocess.run([seven_zip, "x", tmp_path, f"-o{extract_dir}", "-y"], 
                                      check=True, stdout=subprocess.PIPE)
                        
                        # Find the DLLs
                        dll_dir = os.path.join(extract_dir, "$_OUTDIR")
                        if os.path.exists(dll_dir):
                            for dll in glob.glob(os.path.join(dll_dir, "*.dll")):
                                if "zbar" in os.path.basename(dll).lower():
                                    shutil.copy2(dll, target_dir)
                                    print(f"Copied {os.path.basename(dll)} to {target_dir}")
                        return True
            except Exception as e:
                print(f"Error extracting with 7-zip: {e}")
            
            # If 7-zip failed, try with zipfile
            try:
                with zipfile.ZipFile(tmp_path, 'r') as z:
                    # Extract all files to the temp directory
                    z.extractall(extract_dir)
                    
                    # Look for DLLs
                    for root, dirs, files in os.walk(extract_dir):
                        for file in files:
                            if file.lower().endswith('.dll') and "zbar" in file.lower():
                                source_path = os.path.join(root, file)
                                dest_path = os.path.join(target_dir, file)
                                shutil.copy2(source_path, dest_path)
                                print(f"Copied {file} to {target_dir}")
                    return True
            except Exception as e:
                print(f"Error extracting with zipfile: {e}")
                
            # Direct download links as a fallback
            try:
                direct_urls = {
                    'libzbar-0.dll': 'https://github.com/mchehab/zbar/raw/master/zbar.dll',
                    'zbar64.dll': 'https://github.com/NaturalHistoryMuseum/pyzbar/raw/master/pyzbar/zbar64.dll'
                }
                
                for name, url in direct_urls.items():
                    try:
                        response = requests.get(url)
                        response.raise_for_status()
                        
                        with open(os.path.join(target_dir, name), 'wb') as f:
                            f.write(response.content)
                        print(f"Downloaded {name} to {target_dir}")
                    except Exception as e:
                        print(f"Error downloading {name}: {e}")
                
                return True
            except Exception as e:
                print(f"Error with direct download: {e}")
    
    except Exception as e:
        print(f"Error extracting ZBar DLLs: {e}")
    
    finally:
        # Clean up the temp file
        try:
            os.unlink(tmp_path)
        except:
            pass
    
    return False

def get_dll_search_path():
    """
    Get DLL search paths for Windows
    """
    paths = []
    
    # System paths
    system_paths = os.environ.get('PATH', '').split(os.pathsep)
    paths.extend(system_paths)
    
    # Python directory
    paths.append(os.path.dirname(sys.executable))
    
    # Site-packages directories
    site_packages_dirs = site.getsitepackages()
    paths.extend(site_packages_dirs)
    
    # User site-packages
    user_site = site.getusersitepackages()
    if user_site:
        paths.append(user_site)
    
    # Current directory
    paths.append(os.getcwd())
    
    return paths

def find_dll(dll_name):
    """
    Find a DLL in the search paths
    """
    search_paths = get_dll_search_path()
    
    for path in search_paths:
        dll_path = os.path.join(path, dll_name)
        if os.path.exists(dll_path):
            return dll_path
    
    return None

def fix_zbar():
    """
    Try to fix ZBar DLL issues
    """
    print("Checking for ZBar DLLs...")
    
    # Check if we're on Windows
    if platform.system() != 'Windows':
        print("This script is intended for Windows only.")
        return False
    
    # Check Python architecture
    is_64bit = platform.architecture()[0] == '64bit'
    print(f"Python architecture: {'64-bit' if is_64bit else '32-bit'}")
    
    # DLL names to check
    dll_names = ['libzbar-64.dll', 'zbar64.dll'] if is_64bit else ['libzbar-0.dll', 'zbar.dll']
    
    # Check if the DLLs exist
    dll_found = False
    for dll_name in dll_names:
        dll_path = find_dll(dll_name)
        if dll_path:
            print(f"Found {dll_name} at {dll_path}")
            dll_found = True
            
            # Test if the DLL can be loaded
            try:
                handle = ctypes.windll.LoadLibrary(dll_path)
                print(f"Successfully loaded {dll_name}")
            except Exception as e:
                print(f"Error loading {dll_name}: {e}")
                dll_found = False
    
    # If DLL not found or couldn't be loaded, download it
    if not dll_found:
        print("ZBar DLLs not found or couldn't be loaded. Attempting to download...")
        
        # Find pyzbar package directory
        pyzbar_dir = None
        for site_dir in site.getsitepackages():
            potential_dir = os.path.join(site_dir, 'pyzbar')
            if os.path.exists(potential_dir) and os.path.isdir(potential_dir):
                pyzbar_dir = potential_dir
                break
        
        if not pyzbar_dir:
            user_site = site.getusersitepackages()
            potential_dir = os.path.join(user_site, 'pyzbar')
            if os.path.exists(potential_dir):
                pyzbar_dir = potential_dir
        
        if not pyzbar_dir:
            print("Could not find pyzbar package directory.")
            return False
        
        print(f"Found pyzbar directory: {pyzbar_dir}")
        
        # Download ZBar DLLs to pyzbar directory
        if download_zbar(pyzbar_dir):
            print("ZBar DLLs downloaded and installed successfully.")
            
            # Try importing pyzbar.pyzbar to verify fix
            try:
                from pyzbar.pyzbar import decode
                print("Successfully imported pyzbar.pyzbar module.")
                return True
            except Exception as e:
                print(f"Error importing pyzbar.pyzbar: {e}")
                return False
        else:
            print("Failed to download ZBar DLLs.")
            return False
    
    return dll_found

def main():
    """
    Main function
    """
    print("ZBar DLL Fix Tool")
    print("=================")
    
    success = fix_zbar()
    
    if success:
        print("\nZBar DLLs are properly installed. QR code reading should work now.")
        print("Try running your application again.")
    else:
        print("\nCould not fix ZBar DLL issues automatically.")
        print("Please try the following manual steps:")
        print("1. Download ZBar from https://sourceforge.net/projects/zbar/files/zbar/0.10/")
        print("2. Install it on your system")
        print("3. Copy the appropriate DLL files (zbar.dll or zbar64.dll) to your Python's site-packages/pyzbar directory")
        print("   or to your system's PATH")
        print("\nAlternatively, you can reinstall pyzbar using:")
        print("pip uninstall -y pyzbar")
        print("pip install pyzbar")

if __name__ == "__main__":
    main()
