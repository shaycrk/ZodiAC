#!/usr/bin/env python3
"""
Enhanced OCR Form Parser
========================

A comprehensive script to parse hand-filled forms using OCR.
Supports multiple OCR engines, image preprocessing, and structured output.

Features:
- Image alignment to template
- Multiple OCR engines (Tesseract, EasyOCR)
- Configurable ROI (Region of Interest) definitions
- Image preprocessing for better OCR accuracy
- JSON output for structured data
- Visual debugging capabilities

Requirements:
- opencv-python
- pytesseract
- easyocr
- pillow
- numpy
- imutils

Installation:
pip install opencv-python pytesseract easyocr pillow numpy imutils

Usage:
python enhanced_form_ocr.py -i filled_form.png -t blank_template.png --config roi_config.json
"""

import numpy as np
import cv2
import json
import argparse
import os
from pathlib import Path
from collections import namedtuple
from typing import List, Dict, Tuple, Optional
import imutils

# OCR imports
import pytesseract
import easyocr

# Initialize EasyOCR reader
try:
    easyocr_reader = easyocr.Reader(['en'])
except Exception as e:
    print(f"Warning: EasyOCR initialization failed: {e}")
    easyocr_reader = None

class OCRFormParser:
    def __init__(self, debug=False):
        self.debug = debug
        self.OCRLocation = namedtuple("OCRLocation", ["id", "bbox", "filter_keywords", "ocr_config", "poss_values"])
        self.roi_config = None
        
    def preprocess_image(self, image: np.ndarray, method: str = "threshold") -> np.ndarray:
        """
        Preprocess image for better OCR results
        
        Args:
            image: Input image as numpy array
            method: Preprocessing method ('threshold', 'adaptive', 'gaussian')
        
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if method == "threshold":
            # Simple binary threshold
            _, processed = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        elif method == "adaptive":
            # Adaptive threshold
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
        elif method == "gaussian":
            # Gaussian blur + threshold
            _, processed = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            processed = cv2.GaussianBlur(processed, (5, 5), 0)
            processed = 255 - processed
        else:
            processed = gray
        
        return processed

    def align_images(self, image: np.ndarray, template: np.ndarray, 
                    max_features: int = 500, keep_percent: float = 0.2) -> np.ndarray:
        """
        Align input image to template using ORB feature matching
        
        Args:
            image: Input image to align
            template: Reference template image
            max_features: Maximum number of features to detect
            keep_percent: Percentage of best matches to keep
        
        Returns:
            Aligned image
        """
        # Convert to grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Detect ORB keypoints and descriptors
        orb = cv2.ORB_create(max_features)
        kps_image, descs_image = orb.detectAndCompute(image_gray, None)
        kps_template, descs_template = orb.detectAndCompute(template_gray, None)
        
        if descs_image is None or descs_template is None:
            print("Warning: Could not find enough features for alignment")
            return image

        # Match features
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descs_image, descs_template, None)

        # Sort and keep best matches
        matches = sorted(matches, key=lambda x: x.distance)
        keep = int(len(matches) * keep_percent)
        matches = matches[:keep]

        if len(matches) < 4:
            print("Warning: Not enough matches for homography")
            return image

        # Extract matched points
        pts_image = np.zeros((len(matches), 2), dtype="float")
        pts_template = np.zeros((len(matches), 2), dtype="float")
        
        for i, match in enumerate(matches):
            pts_image[i] = kps_image[match.queryIdx].pt
            pts_template[i] = kps_template[match.trainIdx].pt

        # Compute homography and align
        H, _ = cv2.findHomography(pts_image, pts_template, method=cv2.RANSAC)
        if H is not None:
            h, w = template.shape[:2]
            aligned = cv2.warpPerspective(image, H, (w, h))
            return aligned
        else:
            print("Warning: Could not compute homography")
            return image

    def extract_text_tesseract(self, roi: np.ndarray, config: str = '') -> str:
        """Extract text using Tesseract OCR"""
        try:
            if config:
                text = pytesseract.image_to_string(roi, config=config)
            else:
                text = pytesseract.image_to_string(roi)
            return self.cleanup_text(text)
        except Exception as e:
            print(f"Tesseract OCR failed: {e}")
            return ""

    def extract_text_easyocr(self, roi: np.ndarray, allowlist: str = '') -> str:
        """Extract text using EasyOCR"""
        if easyocr_reader is None:
            return ""
        
        try:
            if allowlist:
                results = easyocr_reader.readtext(roi, allowlist=allowlist, detail=0)
            else:
                results = easyocr_reader.readtext(roi, detail=0)
            
            # Join multiple text results
            return ' '.join(results) if results else ""
        except Exception as e:
            print(f"EasyOCR failed: {e}")
            return ""

    def cleanup_text(self, text: str) -> str:
        """Clean up extracted text"""
        # Remove non-ASCII characters and extra whitespace
        cleaned = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        # Remove multiple spaces
        cleaned = ' '.join(cleaned.split())
        return cleaned

    def ocr_roi(self, roi: np.ndarray, ocr_config: Dict) -> Dict:
        """
        Perform OCR on a region of interest
        
        Args:
            roi: Region of interest image
            ocr_config: Configuration for OCR processing
        
        Returns:
            Dictionary with OCR results from different engines
        """
        results = {}
        
        # Preprocess the ROI
        preprocess_method = ocr_config.get('preprocess', 'threshold')
        processed_roi = self.preprocess_image(roi, preprocess_method)
        
        # Save debug image if enabled
        if self.debug:
            debug_filename = f"debug_roi_{hash(roi.tobytes()) % 10000}.png"
            cv2.imwrite(debug_filename, processed_roi)
            print(f"Debug: Saved processed ROI to {debug_filename}")
        
        # EasyOCR
        allowlist = ocr_config.get('easyocr_allowlist', '')
        results['easyocr'] = self.extract_text_easyocr(processed_roi, allowlist)
        if results['easyocr']:
            results['best'] = results['easyocr'].strip()
        else:
            # Fall back to Tesseract OCR if no result from EasyOCR
            tesseract_config = ocr_config.get('tesseract_config', '')
            results['tesseract'] = self.extract_text_tesseract(processed_roi, tesseract_config)
            results['best'] = results['tesseract'].strip() if results['tesseract'] else ""
        
        return results

    def load_roi_config(self, config_path: str) -> None:
        """Load ROI configuration from JSON file and store it on the instance"""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Extract top-level defaults
            default_filter_keywords = config_data.get('default_filter_keywords', [])
            default_ocr_config = config_data.get('default_ocr_config', {})
            
            locations = []
            for item in config_data['roi_locations']:
                # Start with defaults and layer on ROI-specific overrides
                roi_filter_keywords = item.get('filter_keywords', default_filter_keywords)
                
                # Merge OCR config: defaults first, then ROI-specific overrides
                roi_ocr_config = default_ocr_config.copy()
                roi_ocr_config.update(item.get('ocr_config', {}))
                
                locations.append(self.OCRLocation(
                    id=item['id'],
                    bbox=tuple(item['bbox']),
                    filter_keywords=roi_filter_keywords,
                    ocr_config=roi_ocr_config,
                    poss_values=item.get('poss_values', [])
                ))
            
            self.roi_config = locations
        except Exception as e:
            print(f"Error loading ROI config: {e}")
            self.roi_config = []

    def create_default_config(self, output_path: str):
        """Create a default ROI configuration file"""
        default_config = {
            "default_filter_keywords": [],
            "default_ocr_config": {
                "preprocess": "gaussian",
                "tesseract_config": "--psm 7 -c tessedit_char_whitelist=0123456789",
                "easyocr_allowlist": "0123456789"
            },
            "roi_locations": [
                {
                    "id": "car_number",
                    "bbox": [264, 291, 183, 172],
                    "filter_keywords": ["car", "number"],
                    "ocr_config": {
                        "preprocess": "threshold"
                    }
                },
                {
                    "id": "cm_A",
                    "bbox": [222, 892, 244, 154],
                    "filter_keywords": ["checkpoint", "A"]
                }
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"Default config created at {output_path}")

    def save_annotated_image(self, aligned_image, results, output_path):
        """
        Save a copy of the aligned image with ROIs and detected text drawn.
        Text is printed in red to the right of each ROI.
        """
        annotated = aligned_image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        font_thickness = 2
        text_color = (0, 0, 255)  # Red in BGR
        result_type_colors = {
            'expected': (0, 255, 0),  # Green in BGR
            'unexpected': (0, 140, 255),  # Orange in BGR
            'empty': (255, 0, 0)  # Blue in BGR
        }
        for field_id, field_data in results["fields"].items():
            x, y, w, h = field_data["bbox"]
            text = str(field_data["ocr_results"].get("best", ""))
            box_color = result_type_colors[field_data["result_type"]]
            # Draw rectangle
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
            # Put text to the right of the ROI
            text_x = x + w + 10
            text_y = y + h // 2 + 10
            cv2.putText(annotated, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.imwrite(output_path, annotated)
        print(f"Annotated image saved to {output_path}")

    def from_results(self, results: dict, image_path: str, template_path: str, annotate_path: str = None) -> dict:
        """
        Parse a form from a JSON result file
        """
        print("[INFO] Loading images...")
        image = cv2.imread(image_path)
        template = cv2.imread(template_path)

        if image is None or template is None:
            raise ValueError("Could not load images")
        
        print("[INFO] Aligning images...")
        aligned = self.align_images(image, template)
        
        if self.roi_config is None:
            raise ValueError("No ROI configuration loaded")
        
        config_rois = {roi.id: roi for roi in self.roi_config}  # map roi id to roi object

        loaded_results = {
            "image_path": image_path,
            "template_path": template_path,
            "fields": {}
        }
        for field_id, value in results.items():
            if field_id in ['Finished', 'Zodiac_1969', 'Date_1886', 'Passenger_Floyd', 'Passenger_Lick', 'Sunshine'] or field_id.startswith('CP_'):
                continue
            field_data = config_rois[field_id]
            result_type = None
            if value in field_data.poss_values:
                result_type = 'expected'
            elif value:
                result_type = 'unexpected'
            else:
                result_type = 'empty'
            loaded_results["fields"][field_id] = {
                "bbox": field_data.bbox,
                "filter_keywords": field_data.filter_keywords,
                "ocr_results": {"best": value},
                "result_type": result_type
            }

        if annotate_path:
            self.save_annotated_image(aligned, loaded_results, annotate_path)
        
        return loaded_results

    def parse_form(self, image_path: str, template_path: str, annotate_path: str = None) -> dict:
        """
        Main function to parse a filled form
        Optionally saves an annotated image if annotate_path is provided.
        """
        print("[INFO] Loading images...")
        image = cv2.imread(image_path)
        template = cv2.imread(template_path)
        
        if image is None or template is None:
            raise ValueError("Could not load images")
        
        print("[INFO] Aligning images...")
        aligned = self.align_images(image, template)
        
        if self.roi_config is None:
            # Use default ROI configuration
            self.roi_config = [
                self.OCRLocation("car_num", (264, 291, 183, 172), [], 
                               {"preprocess": "threshold", 
                                "tesseract_config": "--psm 7 -c tessedit_char_whitelist=0123456789",
                                "easyocr_allowlist": "0123456789"}),
            ]
        
        print("[INFO] Processing ROI regions...")
        results = {
            "image_path": image_path,
            "template_path": template_path,
            "fields": {}
        }
        
        for location in self.roi_config:
            print(f"[INFO] Processing field: {location.id}")
            x, y, w, h = location.bbox
            roi = aligned[y:y + h, x:x + w]
            
            if roi.size == 0:
                print(f"Warning: Empty ROI for field {location.id}")
                continue
            
            # Perform OCR on the ROI
            ocr_results = self.ocr_roi(roi, location.ocr_config)
            
            # check if the ocr result is in the poss_values
            result_type = None
            if ocr_results['best'] in location.poss_values:
                result_type = 'expected'
            elif ocr_results['best']:
                result_type = 'unexpected'
            else:
                result_type = 'empty'

            results["fields"][location.id] = {
                "bbox": location.bbox,
                "filter_keywords": location.filter_keywords,
                "ocr_results": ocr_results,
                "result_type": result_type
            }
            
            print(f"[INFO] {location.id}: {ocr_results['best']}")
        
        if annotate_path:
            self.save_annotated_image(aligned, results, annotate_path)
        
        return results

def collect_other_fields():
    """Collect other fields from user input"""
    other_fields = {}

    for i in range(1, 15):
        other_fields[f'CP_{i}'] = False
    signed_cps = input("Enter the signed CPS numbers (comma separated): ")
    signed_cps = [f'CP_{i}' for i in signed_cps.split(',') if i]
    for i in signed_cps:
        other_fields[i] = True

    other_fields['Zodiac_1969'] = input("Is 1969 Written next to the Z in? (y/n): ") == 'y'
    other_fields['Date_1886'] = input("Is the date December 27, 1886? (y/n): ") == 'y'
    other_fields['Passenger_Floyd'] = input("Is Richard Floyd on the passenger list? (y/n): ") == 'y'
    other_fields['Passenger_Lick'] = input("Is James Lick on the passenger list? (y/n): ") == 'y'
    other_fields['Sunshine'] = input("Is there a happy face next to the driver's name? (y/n): ") == 'y'

    return other_fields

def main():
    parser = argparse.ArgumentParser(description="Enhanced OCR Form Parser")
    parser.add_argument("-i", "--image", required=True, 
                       help="Path to the filled form image")
    parser.add_argument("-t", "--template", required=True,
                       help="Path to the blank template image")
    parser.add_argument("-c", "--config", 
                       help="Path to ROI configuration JSON file")
    parser.add_argument("--create-config", metavar="PATH",
                       help="Create a default configuration file at the specified path")
    parser.add_argument("-f", "--from-results", 
                       help="Annotate an image from a JSON result file")
    parser.add_argument("-o", "--output", 
                       help="Output JSON file for results")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--annotate", metavar="PATH",
                       help="Save an annotated image of the parsed form")
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        parser_instance = OCRFormParser()
        parser_instance.create_default_config(args.create_config)
        return
    
    # Initialize parser
    ocr_parser = OCRFormParser(debug=args.debug)
    
    # Load ROI configuration if provided
    if args.config:
        ocr_parser.load_roi_config(args.config)
    
    if args.from_results:
        with open(args.from_results, 'r') as f:
            results = json.load(f)
        ocr_parser.from_results(results, args.image, args.template, args.annotate)
        return
    
    try:
        # Parse the form
        results = ocr_parser.parse_form(args.image, args.template, args.annotate)
        
        # Output results
        if args.output:
            output_dict = {k: v['ocr_results']['best'] for k, v in results['fields'].items()}
            output_dict['Finished'] = True
            other_fields = collect_other_fields()
            output_dict.update(other_fields)
            with open(args.output, 'w') as f:
                json.dump(output_dict, f, indent=2)
            print(f"[INFO] Results saved to {args.output}")
        else:
            print("\n" + "="*50)
            print("PARSING RESULTS")
            print("="*50)
            for field_id, field_data in results["fields"].items():
                print(f"{field_id}: {field_data['ocr_results']['best']}")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 