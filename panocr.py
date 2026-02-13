import cv2
import numpy as np
import easyocr
import re

classes = [
    "NAME",
    "FATHER_NAME",
    "DOB",
    "PAN_NUMBER"
]

net = cv2.dnn.readNet(
    r"yolov3_custom_best_10000.weights",
    r"yolov3.cfg"
)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

img = cv2.imread(r"D:\RannLab Works\AdharOCR\PAN OCR\uploads\dummy pan.png")
h, w, _ = img.shape

blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

boxes, confidences, class_ids = [], [], []

for out in outs:
    for det in out:
        scores = det[5:]
        cid = np.argmax(scores)
        conf = scores[cid]
        if conf > 0.3:
            cx, cy, bw, bh = (det[:4] * np.array([w, h, w, h])).astype(int)
            x = int(cx - bw / 2)
            y = int(cy - bh / 2)
            boxes.append([x, y, bw, bh])
            confidences.append(float(conf))
            class_ids.append(cid)

idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

reader = easyocr.Reader(['en'])
data = {}

for i in idxs.flatten():
    x, y, bw, bh = boxes[i]
    label = classes[class_ids[i]]
    
    x = max(0, x)
    y = max(0, y)
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)
    
    roi = img[y:y2, x:x2]
    
    if roi.size == 0:
        continue
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Try both binary and original enhanced image
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )
    
    # Read from both versions
    txt_binary = reader.readtext(binary, detail=0, paragraph=True)
    txt_gray = reader.readtext(gray, detail=0, paragraph=True)
    
    # Choose the longer result (usually better)
    txt = txt_binary if len(" ".join(txt_binary)) > len(" ".join(txt_gray)) else txt_gray
    
    data[label] = " ".join(txt).strip()
    
    # Debug: print what was detected in each region
    print(f"{label}: {data[label]}")

# Read full image text
full_text = reader.readtext(
    img,
    detail=0,
    paragraph=True
)
full_text = " ".join(full_text).upper()

def extract_pan(text_dict, full_text):
    """Extract PAN from detected regions first, then from full text"""
    # Try from PAN_NUMBER region first
    if "PAN_NUMBER" in text_dict:
        pan_text = text_dict["PAN_NUMBER"].upper().replace(" ", "")
        match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', pan_text)
        if match:
            return match.group(0)
    
    # Try from all detected regions
    combined = " ".join(text_dict.values()).upper().replace(" ", "")
    match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', combined)
    if match:
        return match.group(0)
    
    # Finally try from full text
    full_clean = full_text.replace(" ", "")
    match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', full_clean)
    return match.group(0) if match else ""

def extract_name_from_fulltext(text, detected_data):
    if "NAME" in detected_data and detected_data["NAME"]:
        name = detected_data["NAME"].strip()
        name = re.sub(r'DOB.*', '', name, flags=re.IGNORECASE).strip()
        name = re.sub(r'DATE.*', '', name, flags=re.IGNORECASE).strip()
        name = re.sub(r'BIRTH.*', '', name, flags=re.IGNORECASE).strip()
        name = re.sub(r'OF.*', '', name, flags=re.IGNORECASE).strip()
        name = re.sub(r'\s+', ' ', name).strip()
        if len(name) >= 3 and re.match(r'^[A-Za-z\s]+$', name):
            return name.upper()

    pattern1 = r'NAME\s+(.+?)\s+FATHER'
    m = re.search(pattern1, text)
    if m:
        name = m.group(1).strip()
        name = re.sub(r'DOB|DATE|BIRTH|OF', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+', ' ', name).strip()
        if len(name) >= 3 and re.match(r'^[A-Za-z\s]+$', name):
            return name.upper()

    pattern2 = r'NAME\s+(.+?)\s+(?:DATE\s+OF\s+BIRTH|DOB)'
    m = re.search(pattern2, text)
    if m:
        name = m.group(1).strip()
        name = re.sub(r'\s+', ' ', name).strip()
        if len(name) >= 3 and re.match(r'^[A-Za-z\s]+$', name):
            return name.upper()

    pattern3 = r'NAME\s+([A-Z][A-Za-z\s]+)'
    m = re.search(pattern3, text)
    if m:
        name = m.group(1).strip()
        words = name.split()
        name_words = []
        for word in words:
            if word.upper() in ['DOB', 'DATE', 'BIRTH', 'OF', 'FATHER', 'FATHERS', "FATHER'S"]:
                break
            if re.match(r'^[A-Za-z]+$', word):
                name_words.append(word)
            if len(name_words) >= 4:
                break
        if name_words:
            name = ' '.join(name_words)
            if len(name) >= 3:
                return name.upper()

    return ""


def extract_father_from_fulltext(text, detected_data):
    if "FATHER_NAME" in detected_data and detected_data["FATHER_NAME"]:
        fname = detected_data["FATHER_NAME"].strip()
        fname = re.sub(r'DOB.*', '', fname, flags=re.IGNORECASE).strip()
        fname = re.sub(r'DATE.*', '', fname, flags=re.IGNORECASE).strip()
        fname = re.sub(r'BIRTH.*', '', fname, flags=re.IGNORECASE).strip()
        fname = re.sub(r'\s+', ' ', fname).strip()
        if len(fname) >= 3 and re.match(r'^[A-Za-z\s]+$', fname):
            return fname.upper()

    pattern1 = r"FATHER(?:'S|S)?\s+NAME\s+(.+?)\s+(?:DATE\s+OF\s+BIRTH|DOB)"
    m = re.search(pattern1, text, re.IGNORECASE)
    if m:
        fname = m.group(1).strip()
        fname = re.sub(r'\s+', ' ', fname).strip()
        if len(fname) >= 3 and re.match(r'^[A-Za-z\s]+$', fname):
            return fname.upper()

    pattern2 = r"FATHER(?:'S|S)?\s+NAME\s+(.+?)\s+(?:INCOME|TAX|DEPARTMENT|SIGNATURE|PHOTO|\d{2})"
    m = re.search(pattern2, text, re.IGNORECASE)
    if m:
        fname = m.group(1).strip()
        fname = re.sub(r'\s+', ' ', fname).strip()
        if len(fname) >= 3 and re.match(r'^[A-Za-z\s]+$', fname):
            return fname.upper()

    pattern3 = r"FATHER(?:'S|S)?\s+NAME\s+([A-Z][A-Za-z\s]+)"
    m = re.search(pattern3, text, re.IGNORECASE)
    if m:
        fname = m.group(1).strip()
        words = fname.split()
        fname_words = []
        for word in words:
            if word.upper() in ['DOB', 'DATE', 'BIRTH', 'OF', 'INCOME', 'TAX', 'DEPARTMENT']:
                break
            if re.match(r'^[A-Za-z]+$', word):
                fname_words.append(word)
            if len(fname_words) >= 4:
                break
        if fname_words:
            fname = ' '.join(fname_words)
            if len(fname) >= 3:
                return fname.upper()

    return ""

def extract_dob(text, detected_data):
    """Extract date of birth"""
    # Try from detected region first
    if "DOB" in detected_data:
        dob_text = detected_data["DOB"]
        # Look for date patterns
        match = re.search(r'\d{2}[/-]\d{2}[/-]\d{4}', dob_text)
        if match:
            return match.group(0)
    
    # Try from full text
    patterns = [
        r"(?:DOB|DATE OF BIRTH|BIRTH)\s*[:/-]?\s*(\d{2}[/-]\d{2}[/-]\d{4})",
        r"(\d{2}[/-]\d{2}[/-]\d{4})"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return detected_data.get("DOB", "")

# Extract all fields
pan = extract_pan(data, full_text)
name = extract_name_from_fulltext(full_text, data)
father = extract_father_from_fulltext(full_text, data)
dob = extract_dob(full_text, data)
valid = bool(pan)

print("=" * 50)
print("EXTRACTED INFORMATION:")
print("=" * 50)
print(f"PAN Number: {pan}")
print(f"Name: {name}")
print(f"Father's Name: {father}")
print(f"DOB: {dob}")
print(f"Valid PAN Card: {valid}")
print("=" * 50)