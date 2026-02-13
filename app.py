from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import easyocr
import base64
import re
import os
from pathlib import Path
import traceback
import uvicorn
from datetime import datetime
from typing import Optional, Dict, Any, List
from pyzbar import pyzbar
from ultralytics import YOLO


app = FastAPI(
    title="PAN Card OCR API - Enhanced",
    version="2.1",
    description="Enhanced API for extracting comprehensive information from PAN card images"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

classes = ["NAME", "FATHER_NAME", "DOB", "PAN_NUMBER"]

try:
    model = YOLO("yolov8s.pt")
    print("✓ YOLO (Ultralytics) model loaded successfully")
    
    # Check if model has expected classes
    model_classes = [c.lower() for c in model.names.values()]
    expected_classes = ["name", "father_name", "father-s name", "dob", "pan_number", "pan number"]
    
    if not any(c in model_classes for c in expected_classes):
        print(f"⚠️  WARNING: Loaded model 'yolov8s.pt' does not appear to have PAN card classes.")
        print(f"   Model classes: {list(model.names.values())[:5]}... (Total: {len(model.names)})")
        print(f"   Expected classes like: {expected_classes}")
except Exception as e:
    print(f"✗ Error loading YOLO model: {e}")
    model = None

try:
    reader = easyocr.Reader(['en'], gpu=False)
    print("✓ EasyOCR reader initialized successfully")
except Exception as e:
    print(f"✗ Error initializing EasyOCR: {e}")
    reader = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_text(text: str) -> str:
    """Clean and normalize extracted text"""
    if not text:
        return ""
    # Remove common OCR artifacts and non-printable chars
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'[|\[\]{}_]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess_image_for_ocr(image: np.ndarray) -> List[np.ndarray]:
    """Generate multiple preprocessed versions of the image for better OCR"""
    processed_images = []
    
    # 1. Original
    processed_images.append(image)
    
    # 2. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_images.append(gray)
    
    # 3. CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    processed_images.append(enhanced)
    
    # 4. Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    processed_images.append(adaptive)
    
    # 5. Otsu threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(otsu)
    
    # 6. Bilateral filter + CLAHE
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    bilateral_clahe = clahe.apply(bilateral)
    processed_images.append(bilateral_clahe)
    
    # 7. Denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    processed_images.append(denoised)
    
    # 8. Sharpening
    kernel_sharpen = np.array([[-1,-1,-1], 
                               [-1, 9,-1], 
                               [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
    processed_images.append(sharpened)
    
    return processed_images


def extract_text_robust(roi: np.ndarray, reader) -> str:
    """Extract text from ROI using multiple preprocessing methods"""
    if roi is None or roi.size == 0:
        return ""
    
    # Upscale if too small
    h, w = roi.shape[:2]
    if h < 60 or w < 60:
        scale_factor = max(2.0, 80.0 / min(h, w))
        roi = cv2.resize(roi, None, fx=scale_factor, fy=scale_factor, 
                        interpolation=cv2.INTER_CUBIC)
    
    results = []
    
    # Get multiple preprocessed versions
    if len(roi.shape) == 2:  # Already grayscale
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    else:
        roi_bgr = roi
    
    processed_images = preprocess_image_for_ocr(roi_bgr)
    
    # Try OCR on each preprocessed version
    for idx, img in enumerate(processed_images):
        try:
            # Read with detail to get confidence scores
            detections = reader.readtext(img, detail=1, paragraph=False)
            
            if detections:
                # Filter by confidence and combine
                texts = [text for bbox, text, conf in detections if conf > 0.3]
                if texts:
                    combined = " ".join(texts)
                    results.append((combined, len(combined), idx))
        except Exception as e:
            continue
    
    if results:
        # Sort by length (longer is often better) and return best
        results.sort(key=lambda x: x[1], reverse=True)
        return results[0][0]
    
    return ""


def extract_pan_number(text_dict: Dict, full_text: str) -> str:
    """Extract PAN number with multiple pattern matching strategies"""
    # Regex patterns without word boundaries for flexibility
    pan_patterns = [
        r'[A-Z]{5}[0-9]{4}[A-Z]{1}',
        r'[A-Z]{3}P[A-Z]{1}[0-9]{4}[A-Z]{1}',
        r'[A-Z]{3}[PCHATFBLJG]{1}[A-Z]{1}[0-9]{4}[A-Z]{1}',
    ]
    
    def clean_candidate(text: str) -> str:
        """Heuristic cleaning for potential PAN strings"""
        # Specific PAN corrections: 
        # First 5 chars (letters)
        # Next 4 chars (digits): O->0, I->1, B->8, S->5
        # Last char (letter): 0->O, 1->I, 8->B, 5->S
        text = text.upper().replace(" ", "")
        text = re.sub(r'[^A-Z0-9]', '', text)
        return text

    # Strategy 1: From detected PAN_NUMBER region
    if "PAN_NUMBER" in text_dict:
        candidates = text_dict["PAN_NUMBER"]
        if isinstance(candidates, str):
            candidates = [candidates]
            
        for pan_text in candidates:
            pan_text = clean_candidate(pan_text)
            for pattern in pan_patterns:
                match = re.search(pattern, pan_text)
                if match:
                    return match.group(0)
    
    # Strategy 2: From all detected texts combined
    all_texts = []
    for key, val in text_dict.items():
        if isinstance(val, list):
            all_texts.extend(val)
        else:
            all_texts.append(str(val))
    
    combined = clean_candidate(" ".join(all_texts))
    for pattern in pan_patterns:
        match = re.search(pattern, combined)
        if match:
            return match.group(0)
    
    # Strategy 3: From full text (cleaned)
    # Using more aggressive replacements for full text search
    full_clean = full_text.upper().replace(" ", "")
    full_clean = full_clean.replace("O", "0").replace("I", "1") # Basic corrections
    full_clean = re.sub(r'[^A-Z0-9]', '', full_clean)
    
    for pattern in pan_patterns:
        match = re.search(pattern, full_clean)
        if match:
            return match.group(0)
            
    # Strategy 4: Search for PAN-like words in original text with specific char fixups
    # This helps when the PAN is isolated by spaces but has OCR errors
    words = full_text.split()
    for word in words:
        word = re.sub(r'[^A-Za-z0-9]', '', word.upper())
        if len(word) == 10:
            # Try to fix mixture of digits/letters
            # First 5 chars should be letters
            part1 = word[:5].replace('0', 'O').replace('1', 'I').replace('8', 'B').replace('5', 'S')
            # Next 4 chars should be digits
            part2 = word[5:9].replace('O', '0').replace('I', '1').replace('B', '8').replace('S', '5').replace('Z', '2')
            # Last char is a letter
            part3 = word[9].replace('0', 'O').replace('1', 'I').replace('8', 'B').replace('5', 'S')
            
            candidate = part1 + part2 + part3
            for pattern in pan_patterns:
                if re.match(pattern, candidate):
                    return candidate

    return ""

def extract_name(text: str, detected_data: Dict) -> str:
    """Extract cardholder's name with improved pattern matching and validation"""
    
    # Invalid keywords - removed "OF" to avoid blocking "OM" type names
    invalid_keywords = [
        'DOB', 'DATE', 'BIRTH', 'FATHER', 'INCOME', 'TAX', 'GOVT', 'INDIA',
        'PERMANENT', 'ACCOUNT', 'NUMBER', 'CARD', 'PAN', 'GOVERNMENT', 'DEPARTMENT',
        'IDENTITY', 'SIGNATURE', 'PHOTO', 'MUMBER', 'THE', 'HOLDER', 'INERS',
        'ASSESSEE', 'TAXPAYER', 'CARDHOLDER', 'FORM', 'APPLICATION', 'ISSUED'
    ]
    
    def is_valid_cardholder_name(name: str) -> bool:
        """Validate if extracted text is a valid person name"""
        if not name or len(name) < 2:
            return False
        
        if len(name) > 60:
            return False
        
        # Must be alphabetic with spaces and dots only
        if not re.match(r'^[A-Z\s\.]+$', name):
            return False
        
        # Should not contain invalid keywords as complete words
        name_words = name.split()
        for keyword in invalid_keywords:
            if keyword in name_words:  # Check for complete word match only
                return False
        
        # Count valid words (1+ letters, allowing single-letter initials with dots)
        words = []
        for w in name.split():
            # Remove dots and count letters
            letters_only = w.replace('.', '')
            if len(letters_only) >= 1 and re.match(r'^[A-Z]+$', letters_only):
                words.append(w)
        
        # Must have at least 1 valid word
        if len(words) < 1:
            return False
        
        # Accept single word if 3+ letters (excluding dots), or multi-word names
        if len(words) == 1:
            # Count actual letters (remove dots)
            word_letters = words[0].replace('.', '')
            if len(word_letters) >= 3:  # Like "AMIT" or initials won't work alone
                return True
            return False
        
        # Multi-word names are good (including "K. SRINIVASAN" type)
        if len(words) >= 2:
            return True
        
        return False
    
    # Strategy 1: Use detected NAME region with careful cleaning
    if "NAME" in detected_data:
        candidates = detected_data["NAME"]
        if isinstance(candidates, str):
            candidates = [candidates]
            
        for name in candidates:
            name = str(name).strip()
            
            # More precise label removal - only remove if it's clearly a label with separator
            name = re.sub(r'^\s*(?:NAME|Name|CARDHOLDER|Cardholder)\s*[:/-]\s*', '', name, flags=re.IGNORECASE)
            
            # Remove everything after stop words/phrases (with word boundaries)
            name = re.sub(r'\s+(?:DOB|DATE\s+OF\s+BIRTH|FATHER|INCOME|TAX|GOVT|INDIA|PERMANENT|ACCOUNT|SIGNATURE|PHOTO)\b.*$', '', name, flags=re.IGNORECASE)
            
            # Apply text cleaning
            name = clean_text(name)
            
            # Word-by-word validation
            words = name.split()
            clean_words = []
            
            for word in words:
                word = word.strip()
                word_upper = word.upper()
                
                # Stop at invalid keywords (complete word match)
                if word_upper in invalid_keywords:
                    break
                    
                # Keep ALL valid alphabetic words (including single letters with dots like "K.")
                # Accept words with 1+ letters (for initials) or 2+ letters
                if re.match(r'^[A-Za-z\.]+$', word) and len(word) >= 1:
                    # Count actual letters
                    letters = word.replace('.', '')
                    if len(letters) >= 1:
                        clean_words.append(word)
                        
                # Cap at 6 words to avoid capturing extra text
                if len(clean_words) >= 6:
                    break
            
            if len(clean_words) >= 1:
                result = ' '.join(clean_words).upper()
                if is_valid_cardholder_name(result):
                    return result
    
    # Strategy 2: Enhanced pattern matching
    patterns = [
        # Pattern for names with initials (K. SRINIVASAN, R. K. SHARMA, etc.)
        r'(?:NAME|Name)\s*[:/-]\s*([A-Z]\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(?:FATHER|Father|DOB|DATE)',
        
        # Pattern 1: NAME followed by name, then FATHER (most common)
        r'(?:NAME|Name)\s*[:/-]\s*([A-Z][A-Za-z]{1,}(?:\s+[A-Z][A-Za-z]{1,}){0,5})\s+(?:FATHER|Father)',
        
        # Pattern 2: NAME followed by name on same line, then newline
        r'(?:NAME|Name)\s*[:/-]\s*([A-Z][A-Za-z]{1,}(?:\s+[A-Za-z]{1,}){0,5})\s*(?:\n|$)',
        
        # Pattern 3: NAME on one line, name on next line
        r'(?:NAME|Name)\s*[:/-]?\s*\n+\s*([A-Z][A-Za-z]{1,}(?:\s+[A-Z][A-Za-z]{1,}){0,5})\s*\n',
        
        # Pattern 4: NAME followed by name, then DATE OF BIRTH
        r'(?:NAME|Name)\s*[:/-]\s*([A-Z][A-Za-z]{1,}(?:\s+[A-Z][A-Za-z]{1,}){0,5})\s+(?:DATE\s+OF\s+BIRTH|DOB)',
        
        # Pattern 5: After PAN number section
        r'(?:PERMANENT\s+ACCOUNT\s+NUMBER|PAN)\s*[:/-]?\s*[A-Z]{5}[0-9]{4}[A-Z]\s*\n+\s*(?:NAME|Name)?\s*[:/-]?\s*([A-Z][A-Za-z]{1,}(?:\s+[A-Z][A-Za-z]{1,}){0,5})',
        
        # Pattern 6: First capitalized name after header section
        r'(?:INCOME\s+TAX|GOVT|GOVERNMENT)\s+(?:DEPT|DEPARTMENT)?[^\n]*\n+[^\n]*?([A-Z][A-Za-z]{2,}(?:\s+[A-Z][A-Za-z]{2,}){1,4})',
        
        # Pattern 7: Standalone NAME with colon/dash
        r'NAME\s*[:/-]\s*([A-Z]\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)(?=\s|$|\n)',
        
        # Pattern 8: All-caps name pattern
        r'(?:NAME|Name)\s*[:/-]?\s*([A-Z]{2,}(?:\s+[A-Z]{2,}){0,5})\s+(?:FATHER|DOB|\n)',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            name = match.group(1).strip()
            
            # Only remove clear labels with separators
            name = re.sub(r'^(?:NAME|Name|CARDHOLDER)\s*[:/-]\s*', '', name, flags=re.IGNORECASE)
            
            # Word-by-word cleaning
            words = name.split()
            clean_words = []
            
            for word in words:
                word = word.strip()
                word_upper = word.upper()
                
                # Stop at invalid keywords
                if word_upper in invalid_keywords:
                    break
                    
                # Keep ALL valid alphabetic words (including single-letter initials)
                if re.match(r'^[A-Za-z\.]+$', word) and len(word) >= 1:
                    letters = word.replace('.', '')
                    if len(letters) >= 1:
                        clean_words.append(word)
                        
                # Max 6 words
                if len(clean_words) >= 6:
                    break
            
            if len(clean_words) >= 1:
                name_result = ' '.join(clean_words).upper()
                if is_valid_cardholder_name(name_result):
                    return name_result
    
    # Strategy 3: Find first valid proper name in document
    first_600_chars = text[:600]
    
    # Look for capitalized proper names (including 2-letter names)
    proper_names = re.findall(r'\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,}){1,5})\b', first_600_chars)
    
    for name in proper_names:
        name_upper = name.upper()
        if is_valid_cardholder_name(name_upper):
            return name_upper
    
    # Strategy 4: Try all-caps names
    allcaps_names = re.findall(r'\b([A-Z]{2,}(?:\s+[A-Z]{2,}){1,5})\b', first_600_chars)
    
    for name in allcaps_names:
        if is_valid_cardholder_name(name):
            return name
    
    return ""


def extract_father_name(text: str, detected_data: Dict, cardholder_name: str = "") -> str:
    """Extract father's name with improved pattern matching and strict validation"""
    
    # Invalid keywords - removed "OF" to avoid blocking names
    invalid_keywords = [
        'PERMANENT', 'ACCOUNT', 'NUMBER', 'CARD', 'INCOME', 'TAX', 
        'GOVERNMENT', 'INDIA', 'PAN', 'DEPT', 'DEPARTMENT', 'MUMBER',
        'SIGNATURE', 'PHOTO', 'IDENTITY', 'GOVT', 'THE',
        'NAME', 'FATHER', 'FATHERS', 'INERS', 'INER', 'HOLDER',
        'CARDHOLDER', 'TAXPAYER', 'ASSESSEE', 'DOB', 'DATE', 'BIRTH',
        'FORM', 'APPLICATION', 'ISSUED', 'VALID', 'TILL'
    ]
    
    def is_valid_father_name(name: str, cardholder: str = "") -> bool:
        """Check if the extracted text is a valid father's name"""
        if not name or len(name) < 2:
            return False
        
        if len(name) > 60:
            return False
        
        # Must contain only letters, spaces, and dots
        if not re.match(r'^[A-Z\s\.]+$', name):
            return False
        
        # Should not contain any invalid keywords as complete words
        name_words = name.split()
        for keyword in invalid_keywords:
            if keyword in name_words:  # Complete word match only
                return False
        
        # Should not be common labels
        if name in ['NAME', 'FATHER', 'FATHERS', 'DOB', 'DATE OF BIRTH', 
                    'FATHERS NAME', 'FATHER NAME', 'INERS NAME', 'FATHER S NAME',
                    'S NAME', 'FATHER S', 'S']:
            return False
        
        # Must have at least valid words (1+ letters for initials)
        words = []
        for w in name.split():
            letters_only = w.replace('.', '')
            if len(letters_only) >= 1 and re.match(r'^[A-Z]+$', letters_only):
                words.append(w)
        
        if len(words) < 1:
            return False
        
        # For single word, at least 3 characters (not including dots)
        if len(words) == 1:
            word_letters = words[0].replace('.', '')
            if len(word_letters) < 3:
                return False
        
        # Should not be same as cardholder name
        if cardholder and name == cardholder.upper():
            return False
        
        # Should not end with label fragments
        if name.endswith((' NAME', ' FATHER', ' FATHERS', ' DOB', ' DATE', ' S', ' INERS')):
            return False
        
        # Should not start with common prefixes that are not names
        if name.startswith(('THE ', 'AND ', 'OR ')):
            return False
        
        return True
    
    # Strategy 1: Use detected FATHER_NAME region with careful cleaning
    if "FATHER_NAME" in detected_data:
        candidates = detected_data["FATHER_NAME"]
        if isinstance(candidates, str):
            candidates = [candidates]
            
        for fname in candidates:
            fname = str(fname).strip()
            
            # More precise label removal
            fname = re.sub(r"^\s*(?:FATHER'?S?\s*NAME|Father'?s?\s*Name|INERS\s*NAME|FATHER|Father)\s*[:/-]\s*", '', fname, flags=re.IGNORECASE)
            
            # Remove everything after markers (with word boundaries)
            fname = re.sub(r'\s+(?:DOB|DATE\s+OF\s+BIRTH|BIRTH|INCOME|TAX|SIGNATURE|GOVT|INDIA|DEPARTMENT|PERMANENT|ACCOUNT|NUMBER|CARD|MUMBER|PHOTO)\b.*$', '', fname, flags=re.IGNORECASE)
            
            fname = clean_text(fname)
            
            # Clean word by word
            words = fname.split()
            clean_words = []
            
            for word in words:
                word = word.strip()
                word_upper = word.upper()
                
                # Skip invalid keywords
                if word_upper in invalid_keywords:
                    continue
                    
                # Keep ALL valid alphabetic words (including single-letter initials)
                if re.match(r'^[A-Za-z\.]+$', word) and len(word) >= 1:
                    letters = word.replace('.', '')
                    if len(letters) >= 1:
                        clean_words.append(word)
                    
                # Max 5 words for father's name
                if len(clean_words) >= 5:
                    break
            
            if len(clean_words) >= 1:
                result = ' '.join(clean_words).upper()
                if is_valid_father_name(result, cardholder_name):
                    return result
    
    # Strategy 2: Enhanced pattern matching
    patterns = [
        # Pattern for names with initials (R. KRISHNAN, etc.)
        r"(?:FATHER'?S?\s+NAME|Father'?s?\s+Name|FATHER|Father)\s*[:/-]\s*([A-Z]\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(?:DOB|DATE)",
        
        # Pattern 1: FATHER'S NAME label followed by actual name before DOB
        r"(?:FATHER'?S?\s+NAME|Father'?s?\s+Name)\s*[:/-]\s*([A-Z][A-Za-z]{1,}(?:\s+[A-Z][A-Za-z]{1,}){0,4})\s+(?:DATE\s+OF\s+BIRTH|DOB|\d{2}/\d{2}/\d{4})",
        
        # Pattern 2: Just FATHER label before name and DOB
        r"(?:FATHER|Father)(?:'s)?\s+(?:NAME|Name)?\s*[:/-]\s*([A-Z][A-Za-z]{1,}(?:\s+[A-Z][A-Za-z]{1,}){0,4})\s+(?:DATE\s+OF\s+BIRTH|DOB|\d{2}/)",
        
        # Pattern 3: FATHER on one line, name on next line
        r"(?:FATHER'?S?\s*NAME|Father'?s?\s*Name|FATHER|Father)\s*[:/-]?\s*\n+\s*([A-Z][A-Za-z]{1,}(?:\s+[A-Z][A-Za-z]{1,}){0,4})\s*\n",
        
        # Pattern 4: After cardholder's NAME, before DOB (sequential layout)
        r"(?:NAME)\s+[A-Z][A-Za-z\s\.]+?\s+(?:FATHER|Father)[^\n]*?\s+([A-Z][A-Za-z]{1,}(?:\s+[A-Z][A-Za-z]{1,}){0,4})\s+(?:DATE|DOB)",
        
        # Pattern 5: FATHER followed by colon and name
        r"(?:FATHER|Father)\s*:\s*([A-Z]\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)(?=\s+(?:Date|DOB|\d{2}/))",
        
        # Pattern 6: Capture second name in sequential layout (2 capture groups)
        r"(?:NAME)\s+([A-Z][A-Za-z]{1,}(?:\s+[A-Z][A-Za-z]{1,}){0,4})\s+(?:FATHER|Father)?\s*'?s?\s*(?:NAME|Name)?\s*[:/-]?\s*([A-Z]\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)",
        
        # Pattern 7: All-caps father name after FATHER label
        r"(?:FATHER'?S?\s+NAME|FATHER)\s*[:/-]?\s*([A-Z]{2,}(?:\s+[A-Z]{2,}){0,4})\s+(?:DOB|DATE)",
        
        # Pattern 8: Father name between cardholder name and DOB (multiline)
        r"(?:NAME)[^\n]+\n+[^\n]*?([A-Z][A-Za-z]{1,}(?:\s+[A-Z][A-Za-z]{1,}){0,4})\s*\n+[^\n]*?(?:DATE\s+OF\s+BIRTH|DOB)",
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            # For pattern 6, try the second capture group first (father's name)
            fname = None
            if match.lastindex == 2:  # Pattern has 2 groups
                fname = match.group(2).strip()
                # Validate it's not the cardholder's name
                if cardholder_name and fname.upper() == cardholder_name.upper():
                    continue
            else:
                fname = match.group(1).strip()
            
            if not fname:
                continue
            
            # Only remove clear labels with separators
            fname = re.sub(r"^(?:FATHER'?S?\s*NAME|Father'?s?\s*Name|INERS\s*NAME|FATHER|Father)\s*[:/-]\s*", '', fname, flags=re.IGNORECASE)
            
            # Clean word by word
            words = fname.split()
            clean_words = []
            
            for word in words:
                word = word.strip()
                word_upper = word.upper()
                
                # Skip invalid keywords
                if word_upper in invalid_keywords:
                    break
                    
                # Keep ALL alphabetic words (including single-letter initials)
                if re.match(r'^[A-Za-z\.]+$', word) and len(word) >= 1:
                    letters = word.replace('.', '')
                    if len(letters) >= 1:
                        clean_words.append(word)
                    
                # Max 5 words
                if len(clean_words) >= 5:
                    break
            
            if len(clean_words) >= 1:
                fname_result = ' '.join(clean_words).upper()
                if is_valid_father_name(fname_result, cardholder_name):
                    return fname_result
    
    # Strategy 3: Find second person name (assuming first is cardholder)
    if cardholder_name:
        # Find all proper names
        all_names = re.findall(r'\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,}){1,4})\b', text)
        
        seen_cardholder = False
        for name in all_names:
            name_upper = name.upper()
            
            # Track if we've seen the cardholder name
            if cardholder_name.upper() in name_upper or name_upper in cardholder_name.upper():
                seen_cardholder = True
                continue
            
            # After seeing cardholder, next valid name is likely father's
            if seen_cardholder and is_valid_father_name(name_upper, cardholder_name):
                return name_upper
    
    # Strategy 4: Try all-caps names
    if cardholder_name:
        allcaps_names = re.findall(r'\b([A-Z]{3,}(?:\s+[A-Z]{3,}){0,4})\b', text)
        
        for name in allcaps_names:
            if is_valid_father_name(name, cardholder_name):
                return name
    
    # Strategy 5: Find any valid person name that's not the cardholder
    all_names = re.findall(r'\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,}){1,4})\b', text)
    
    for name in all_names:
        name_upper = name.upper()
        
        # Skip if it matches cardholder
        if cardholder_name and (cardholder_name.upper() in name_upper or name_upper in cardholder_name.upper()):
            continue
            
        if is_valid_father_name(name_upper, cardholder_name):
            return name_upper
    
    return ""



def extract_date_of_birth(text: str, detected_data: Dict) -> str:
    """Extract date of birth with improved pattern matching"""
    
    # Strategy 1: From detected DOB region
    if "DOB" in detected_data:
        candidates = detected_data["DOB"]
        if isinstance(candidates, str):
            candidates = [candidates]
            
        date_patterns = [
            r'(\d{2}[/-]\d{2}[/-]\d{4})',
            r'(\d{2}\s+\d{2}\s+\d{4})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        ]
        
        for dob_text in candidates:
            dob_text = str(dob_text).replace('O', '0').replace('o', '0')
            for pattern in date_patterns:
                match = re.search(pattern, dob_text)
                if match:
                    date_str = match.group(1)
                    if validate_date(date_str):
                        return normalize_date(date_str)
    
    # Strategy 2: Pattern matching from full text
    patterns = [
        # After DOB label
        r"(?:DATE\s+OF\s+BIRTH|DOB|Date\s+of\s+Birth)\s*[:/-]?\s*(\d{2}[/-]\d{2}[/-]\d{4})",
        r"(?:DATE\s+OF\s+BIRTH|DOB|Date\s+of\s+Birth)\s*[:/-]?\s*(\d{2}\s+\d{2}\s+\d{4})",
        # Any date pattern (fallback)
        r'(\d{2}[/-]\d{2}[/-]\d{4})',
        # With spaces
        r'(\d{2}\s+\d{2}\s+\d{4})',
    ]
    
    text_clean = text.replace('O', '0').replace('o', '0')
    
    for pattern in patterns:
        matches = re.finditer(pattern, text_clean, re.IGNORECASE)
        for match in matches:
            date_str = match.group(1)
            if validate_date(date_str):
                return normalize_date(date_str)
    
    return ""


def validate_date(date_str: str) -> bool:
    """Validate if string is a valid date"""
    try:
        date_str = date_str.replace(" ", "").replace("-", "/")
        parts = date_str.split("/")
        
        if len(parts) != 3:
            return False
        
        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
        
        # Handle 2-digit years
        if year < 100:
            year = 1900 + year if year > 50 else 2000 + year
        
        # Basic validation
        if not (1 <= month <= 12):
            return False
        if not (1 <= day <= 31):
            return False
        if not (1900 <= year <= datetime.now().year):
            return False
        
        # Validate actual date
        datetime(year, month, day)
        return True
    except:
        return False


def normalize_date(date_str: str) -> str:
    """Normalize date to DD/MM/YYYY format"""
    try:
        date_str = date_str.replace(" ", "").replace("-", "/")
        parts = date_str.split("/")
        
        if len(parts) == 3:
            day, month, year = parts[0].zfill(2), parts[1].zfill(2), parts[2]
            
            # Handle 2-digit years
            if len(year) == 2:
                year = "19" + year if int(year) > 50 else "20" + year
            
            return f"{day}/{month}/{year}"
    except:
        pass
    return date_str


def extract_card_type(pan_number: str) -> str:
    """Determine card type from PAN number"""
    if len(pan_number) >= 4:
        fourth_char = pan_number[3]
        card_types = {
            'P': 'Individual/Person',
            'C': 'Company',
            'H': 'Hindu Undivided Family (HUF)',
            'F': 'Firm/Partnership',
            'A': 'Association of Persons (AOP)',
            'T': 'Trust',
            'B': 'Body of Individuals (BOI)',
            'L': 'Local Authority',
            'J': 'Artificial Juridical Person',
            'G': 'Government',
        }
        return card_types.get(fourth_char, 'Unknown')
    return ""


def extract_profile_image(image: np.ndarray) -> Optional[str]:
    """Extract profile photo from PAN card"""
    try:
        h, w = image.shape[:2]
        
        # Try face detection first
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) > 0:
                x, y, w_face, h_face = max(faces, key=lambda f: f[2] * f[3])
                
                padding_x = int(w_face * 0.35)
                padding_y = int(h_face * 0.40)
                
                x1 = max(0, x - padding_x)
                y1 = max(0, y - padding_y)
                x2 = min(w, x + w_face + padding_x)
                y2 = min(h, y + h_face + padding_y)
                
                photo = image[y1:y2, x1:x2]
                
                if photo.size > 0 and photo.shape[0] > 40 and photo.shape[1] > 40:
                    _, buffer = cv2.imencode('.png', photo)
                    photo_base64 = base64.b64encode(buffer).decode('utf-8')
                    return f"data:image/png;base64,{photo_base64}"
        except Exception as e:
            print(f"Face detection failed: {e}")
        
        # Fallback: extract from expected region
        x_start = int(w * 0.02)
        x_end = int(w * 0.35)
        y_start = int(h * 0.10)
        y_end = int(h * 0.75)
        
        photo_region = image[y_start:y_end, x_start:x_end]
        
        if photo_region.size > 0:
            _, buffer = cv2.imencode('.png', photo_region)
            photo_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/png;base64,{photo_base64}"
        
        return None
        
    except Exception as e:
        print(f"Error extracting profile image: {e}")
        return None


def extract_qr_code(image: np.ndarray) -> Optional[Dict[str, Any]]:
    """Extract QR code from PAN card with enhanced detection"""
    try:
        h, w = image.shape[:2]
        
        # Expanded search regions - QR can be in various positions
        search_regions = [
            # Right side regions (most common)
            (int(w * 0.60), w, 0, h),                    # Full right 40%
            (int(w * 0.65), w, 0, h),                    # Right 35%
            (int(w * 0.70), w, 0, int(h * 0.60)),       # Top-right
            (int(w * 0.70), w, int(h * 0.40), h),       # Bottom-right
            (int(w * 0.55), w, 0, h),                    # Wider right region
            # Center-right regions
            (int(w * 0.50), w, int(h * 0.20), int(h * 0.80)),  # Center-right vertical strip
        ]
        
        for x_start, x_end, y_start, y_end in search_regions:
            qr_region = image[y_start:y_end, x_start:x_end].copy()
            
            if qr_region.size == 0:
                continue
            
            # Method 1: Direct decode on original region
            try:
                decoded_objects = pyzbar.decode(qr_region)
                
                if decoded_objects:
                    qr = decoded_objects[0]
                    x, y, w_qr, h_qr = qr.rect
                    
                    # Extract with padding
                    padding = 15
                    y1 = max(0, y - padding)
                    x1 = max(0, x - padding)
                    y2 = min(qr_region.shape[0], y + h_qr + padding)
                    x2 = min(qr_region.shape[1], x + w_qr + padding)
                    
                    qr_image = qr_region[y1:y2, x1:x2]
                    
                    if qr_image.size > 0 and qr_image.shape[0] > 20 and qr_image.shape[1] > 20:
                        _, buffer = cv2.imencode('.png', qr_image)
                        qr_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        return {
                            "qr_code_image": f"data:image/png;base64,{qr_base64}",
                            "qr_data": qr.data.decode('utf-8') if qr.data else None,
                            "qr_type": qr.type,
                            "qr_found": True,
                            "qr_dimensions": {
                                "width": qr_image.shape[1],
                                "height": qr_image.shape[0]
                            }
                        }
            except Exception as e:
                pass
            
            # Method 2: Try with preprocessing
            try:
                gray = cv2.cvtColor(qr_region, cv2.COLOR_BGR2GRAY)
                
                # Multiple preprocessing approaches
                preprocessed_versions = []
                
                # 1. Original grayscale
                preprocessed_versions.append(gray)
                
                # 2. Gaussian blur
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                preprocessed_versions.append(blurred)
                
                # 3. Median blur
                median = cv2.medianBlur(gray, 5)
                preprocessed_versions.append(median)
                
                # 4. Binary threshold
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                preprocessed_versions.append(binary)
                
                # 5. Otsu threshold
                _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                preprocessed_versions.append(otsu)
                
                # 6. Adaptive threshold
                adaptive = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                preprocessed_versions.append(adaptive)
                
                # 7. Inverted binary
                _, binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                preprocessed_versions.append(binary_inv)
                
                # 8. CLAHE enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                preprocessed_versions.append(enhanced)
                
                # 9. Sharpened
                kernel_sharpen = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
                sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
                preprocessed_versions.append(sharpened)
                
                # Try decoding each preprocessed version
                for processed in preprocessed_versions:
                    decoded_objects = pyzbar.decode(processed)
                    
                    if decoded_objects:
                        qr = decoded_objects[0]
                        x, y, w_qr, h_qr = qr.rect
                        
                        padding = 15
                        y1 = max(0, y - padding)
                        x1 = max(0, x - padding)
                        y2 = min(qr_region.shape[0], y + h_qr + padding)
                        x2 = min(qr_region.shape[1], x + w_qr + padding)
                        
                        qr_image = qr_region[y1:y2, x1:x2]
                        
                        if qr_image.size > 0 and qr_image.shape[0] > 20 and qr_image.shape[1] > 20:
                            _, buffer = cv2.imencode('.png', qr_image)
                            qr_base64 = base64.b64encode(buffer).decode('utf-8')
                            
                            return {
                                "qr_code_image": f"data:image/png;base64,{qr_base64}",
                                "qr_data": qr.data.decode('utf-8') if qr.data else None,
                                "qr_type": qr.type,
                                "qr_found": True,
                                "qr_dimensions": {
                                    "width": qr_image.shape[1],
                                    "height": qr_image.shape[0]
                                }
                            }
            except Exception as e:
                pass
        
        # Method 3: Contour-based fallback (if QR not decoded but visually present)
        try:
            # Focus on right side
            fallback_region = image[0:h, int(w * 0.60):w].copy()
            
            if fallback_region.size == 0:
                return {"qr_code_image": None, "qr_found": False}
            
            gray = cv2.cvtColor(fallback_region, cv2.COLOR_BGR2GRAY)
            
            # Edge detection for QR patterns
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 100)
            
            # Dilate to connect edges
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_qr_candidates = []
            region_area = fallback_region.shape[0] * fallback_region.shape[1]
            
            for contour in contours:
                area = cv2.contourArea(contour)
                min_area = region_area * 0.03  # At least 3% of region
                max_area = region_area * 0.70  # 
                
                if min_area < area < max_area:
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    aspect_ratio = float(w_box) / h_box if h_box > 0 else 0
                    
                    # QR codes are square (aspect ratio close to 1.0)
                    if 0.7 <= aspect_ratio <= 1.4 and w_box > 40 and h_box > 40:
                        # Calculate squareness score
                        squareness = 1.0 - abs(1.0 - aspect_ratio)
                        
                        # Check if it has QR-like features (count corners)
                        peri = cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                        
                        valid_qr_candidates.append({
                            'contour': contour,
                            'area': area,
                            'bbox': (x, y, w_box, h_box),
                            'squareness': squareness,
                            'corners': len(approx)
                        })
            
            if valid_qr_candidates:
                valid_qr_candidates.sort(
                    key=lambda c: (c['squareness'], c['area']), 
                    reverse=True
                )
                
                # Take the best candidate
                best = valid_qr_candidates[0]
                x, y, w_box, h_box = best['bbox']
                
                # Extract with padding
                padding = 15
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(fallback_region.shape[1], x + w_box + padding)
                y2 = min(fallback_region.shape[0], y + h_box + padding)
                
                qr_image = fallback_region[y1:y2, x1:x2]
                
                if qr_image.size > 0 and qr_image.shape[0] > 30 and qr_image.shape[1] > 30:
                    _, buffer = cv2.imencode('.png', qr_image)
                    qr_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    return {
                        "qr_code_image": f"data:image/png;base64,{qr_base64}",
                    }
        except Exception as e:
            print(f"Contour-based QR detection failed: {e}")
        
        return {"qr_code_image": None, "qr_found": False}
    
    except Exception as e:
        print(f"Error extracting QR code: {e}")
        traceback.print_exc()
        return {"qr_code_image": None, "qr_found": False, "error": str(e)}

def extract_signature(image: np.ndarray) -> Optional[str]:
    """Extract signature from PAN card"""
    try:
        h, w = image.shape[:2]
        
        x_start = int(w * 0.35)
        x_end = int(w * 0.65)
        y_start = int(h * 0.70)
        y_end = h
        
        sig_region = image[y_start:y_end, x_start:x_end].copy()
        
        if sig_region.size > 0:
            _, buffer = cv2.imencode('.png', sig_region)
            sig_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/png;base64,{sig_base64}"
        
        return None
        
    except Exception as e:
        print(f"Error extracting signature: {e}")
        return None


def process_pan_card(image_path: str) -> Dict[str, Any]:
    """Main function to process PAN card and extract all information"""
    if model is None or reader is None:
        raise Exception("YOLO model or EasyOCR reader not initialized")
    
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Failed to read image")
    
    h, w, _ = img.shape
    
    # Extract visual elements
    profile_image_base64 = extract_profile_image(img.copy())
    qr_code_info = extract_qr_code(img.copy())
    signature_base64 = extract_signature(img.copy())
    
    # Run YOLO detection
    results = model(img)
    
    detected_regions = {}
    
    # Class name mapping
    class_mapping = {
        'dob': 'DOB',
        'father-s name': 'FATHER_NAME',
        'father_name': 'FATHER_NAME',
        "father's name": 'FATHER_NAME',
        'name': 'NAME',
        'pan number': 'PAN_NUMBER',
        'pan_number': 'PAN_NUMBER',
    }
    
    # Extract text from detected regions
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label_name = result.names[cls_id].lower()
            
            if conf > 0.25:  # Lower threshold to catch more detections
                # Map the label to standard keys
                mapped_label = class_mapping.get(label_name, label_name.upper())
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # Add padding to ROI for better OCR
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                
                roi = img[y1:y2, x1:x2]
                
                if roi.size == 0:
                    continue
                
                # Extract text using robust method
                extracted_text = extract_text_robust(roi, reader)
                
                if extracted_text:
                    if mapped_label not in detected_regions:
                        detected_regions[mapped_label] = []
                    detected_regions[mapped_label].append(extracted_text)
    
    # Full text extraction with multiple methods
    print("Extracting full text from image...")
    full_text_list = reader.readtext(img, detail=0, paragraph=False)
    full_text = " ".join(full_text_list)
    
    # Try enhanced version
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced_text_list = reader.readtext(enhanced, detail=0, paragraph=False)
        enhanced_text = " ".join(enhanced_text_list)
        
        # Use the longer text
        if len(enhanced_text) > len(full_text):
            full_text = enhanced_text
    except Exception as e:
        print(f"Enhanced text extraction failed: {e}")
    
    print(f"Full text extracted: {full_text[:200]}...")
    
    # Extract fields
    pan_number = clean_text(extract_pan_number(detected_regions, full_text))
    name = clean_text(extract_name(full_text, detected_regions))
    father_name = clean_text(extract_father_name(full_text, detected_regions, name))
    dob = extract_date_of_birth(full_text, detected_regions)
    card_type = extract_card_type(pan_number)
    
    is_valid = bool(pan_number) and len(pan_number) == 10
    
    # Prepare clean output
    clean_detected_regions = {}
    for key, val in detected_regions.items():
        if isinstance(val, list) and val:
            # Combine all detections for this field
            clean_detected_regions[key] = " | ".join(val)
        elif isinstance(val, str):
            clean_detected_regions[key] = val
        else:
            clean_detected_regions[key] = ""
    
    # Calculate completeness score
    fields_found = sum([
        bool(pan_number),
        bool(name),
        bool(father_name),
        bool(dob),
    ])
    completeness_score = (fields_found / 4) * 100
    
    result = {
        "pan_number": pan_number,
        "name": name,
        "father_name": father_name,
        "date_of_birth": dob,
        "card_type": card_type,
        "profile_image": profile_image_base64,
        "qr_code_info": qr_code_info,
        "signature": signature_base64,
        "is_valid_pan": is_valid,
        "completeness_score": round(completeness_score, 2),
        "confidence_level": "high" if completeness_score >= 75 else "medium" if completeness_score >= 50 else "low",
        "detected_regions": clean_detected_regions,
        "full_extracted_text": full_text[:500],  # First 500 chars for debugging
        "image_dimensions": {
            "width": w,
            "height": h
        },
        "total_detections": sum(len(r.boxes) for r in results) if results else 0,
    }
    
    return result


@app.post('/extract')
async def extract_pan_info(image: UploadFile = File(...)):
    """API endpoint to extract PAN card information"""
    try:
        if not image.filename:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error": "No file selected",
                    "message": "Please select a file to upload"
                }
            )
        
        if not allowed_file(image.filename):
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error": "Invalid file type",
                    "message": f"Allowed file types: {', '.join(ALLOWED_EXTENSIONS)}",
                    "uploaded_file": image.filename
                }
            )
        
        contents = await image.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail={
                    "success": False,
                    "error": "File too large",
                    "message": f"Maximum file size is {MAX_FILE_SIZE // (1024*1024)}MB",
                    "uploaded_size": f"{len(contents) // (1024*1024)}MB"
                }
            )
        
        filename = "".join(c for c in image.filename if c.isalnum() or c in ('_', '.', '-'))
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        with open(filepath, 'wb') as f:
            f.write(contents)
        
        result = process_pan_card(filepath)
        
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Warning: Could not remove temporary file {filepath}: {e}")
        
        return {
            "success": True,
            "data": result,
            "message": "PAN card processed successfully",
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        print(traceback.format_exc())
        
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "message": "An error occurred while processing the image. Please ensure the image is a valid PAN card."
            }
        )


@app.get('/')
async def root():
    return {
        "name": "PAN Card OCR API",
        "version": "2.1",
        "status": "active",
        "endpoints": {
            "extract": "/extract (POST)"
        }
    }


if __name__ == '__main__':
    print("\n" + "="*60)
    print(" "*15 + "ENHANCED PAN CARD OCR API v2.1")
    print("="*60)
    print("Starting FastAPI application...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📎 Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}")
    print(f"📦 Max file size: {MAX_FILE_SIZE // (1024*1024)}MB")
    print(f"🤖 YOLO model: {'✓ Loaded' if model else '✗ Not loaded'}")
    print(f"👁  OCR reader: {'✓ Initialized' if reader else '✗ Not initialized'}")
    print("="*60)
    print("\nAPI Endpoint:")
    print("  • POST /extract  - Extract PAN card data")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(app, host='0.0.0.0', port=5000)