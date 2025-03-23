import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse

def convert_coordinates(size, box):
    """
    Convert VOC XML bounding box coordinates to YOLO format
    
    Args:
        size: tuple of (width, height) - image dimensions
        box: tuple of (xmin, ymin, xmax, ymax) - VOC format bounding box
        
    Returns:
        tuple of (x_center, y_center, width, height) - YOLO format coordinates
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    
    # Extract coordinates
    xmin, ymin, xmax, ymax = box
    
    # Calculate YOLO coordinates
    x_center = (xmin + xmax) / 2.0 * dw
    y_center = (ymin + ymax) / 2.0 * dh
    w = (xmax - xmin) * dw
    h = (ymax - ymin) * dh
    
    return (x_center, y_center, w, h)

def convert_xml_to_yolo(xml_file, class_mapping):
    """
    Convert a single XML file to YOLO format
    
    Args:
        xml_file: path to XML file
        class_mapping: dictionary mapping class names to class ids
        
    Returns:
        list of strings in YOLO format
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get image size
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # Check for invalid dimensions
        if width <= 0 or height <= 0:
            print(f"Warning: Invalid image dimensions (width={width}, height={height}) in {xml_file}. Skipping...")
            return []
        
        yolo_lines = []
        
        # Process each object
        for obj in root.findall('object'):
            # Get class name and check if it's in our mapping
            class_name = obj.find('name').text
            if class_name not in class_mapping:
                print(f"Warning: Class '{class_name}' not found in class mapping. Skipping...")
                continue
                
            # Get class id
            class_id = class_mapping[class_name]
            
            # Get bounding box coordinates
            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            ymin = float(xmlbox.find('ymin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymax = float(xmlbox.find('ymax').text)
            
            # Ensure coordinates are valid
            if xmin >= xmax or ymin >= ymax:
                print(f"Warning: Invalid bounding box (xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}) in {xml_file}. Skipping...")
                continue
                
            # Convert to YOLO format
            x_center, y_center, w, h = convert_coordinates((width, height), (xmin, ymin, xmax, ymax))
            
            # Format YOLO line
            yolo_line = f"{class_id} {x_center:.15f} {y_center:.15f} {w:.15f} {h:.15f}"
            yolo_lines.append(yolo_line)
        
        return yolo_lines
    except Exception as e:
        print(f"Error processing {xml_file}: {e}")
        return []

def process_folder(xml_folder, output_folder, class_file=None):
    """
    Process all XML files in a folder and convert them to YOLO format
    
    Args:
        xml_folder: path to folder containing XML files
        output_folder: path to folder where YOLO text files will be saved
        class_file: path to file containing class names (one per line)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Create class mapping
    class_mapping = {}
    if class_file:
        with open(class_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
            class_mapping = {name: i for i, name in enumerate(class_names)}
    else:
        # If no class file is provided, let's try to infer classes from the XML files
        print("No class file provided. Will try to infer classes from XML files.")
        unique_classes = set()
        
        # First pass: collect all unique class names
        for xml_file in glob.glob(os.path.join(xml_folder, '*.xml')):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                unique_classes.add(class_name)
        
        # Create mapping
        for i, name in enumerate(sorted(unique_classes)):
            class_mapping[name] = i
            
        print(f"Inferred classes: {class_mapping}")
        
        # Save inferred classes to a file
        with open(os.path.join(output_folder, 'classes.txt'), 'w') as f:
            for name in sorted(unique_classes):
                f.write(f"{name}\n")
    
    # Process all XML files
    xml_files = glob.glob(os.path.join(xml_folder, '*.xml'))
    total_files = len(xml_files)
    
    print(f"Found {total_files} XML files to process")
    
    for i, xml_file in enumerate(xml_files):
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(xml_file))[0]
        
        # Convert to YOLO format
        yolo_lines = convert_xml_to_yolo(xml_file, class_mapping)
        
        # Save to output file
        txt_file = os.path.join(output_folder, f"{base_name}.txt")
        with open(txt_file, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        # Print progress
        if (i + 1) % 100 == 0 or (i + 1) == total_files:
            print(f"Processed {i + 1}/{total_files} files")
    
    print(f"Conversion complete. YOLO format annotations saved to {output_folder}")

def main():
    parser = argparse.ArgumentParser(description='Convert XML annotations to YOLO format')
    parser.add_argument('--xml_folder', required=True, help='Path to folder containing XML annotations', default=r"D:\NCKHSV.2024-2025\MultiMediapipe\archive\Test\Test\Annotations")
    parser.add_argument('--output_folder', required=True, help='Path to folder where YOLO text files will be saved', default=r"D:\NCKHSV.2024-2025\MultiMediapipe\Yolo_data\Valid\labels")
    parser.add_argument('--class_file', help='Path to file containing class names (one per line)')
    
    args = parser.parse_args()
    
    process_folder(args.xml_folder, args.output_folder, args.class_file)

if __name__ == "__main__":
    main()

    # python convert.py --xml_folder "D:\NCKHSV.2024-2025\MultiMediapipe\archive\Val\Val\Annotations" --output_folder "D:\NCKHSV.2024-2025\MultiMediapipe\Yolo_data\Valid\labels"