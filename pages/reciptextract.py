import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import json
import re
import io
from datetime import datetime
import os
import cv2
import numpy as np
import base64

# הגדרת נתיב Tesseract ל-Windows (אם לא ב-PATH)
if os.name == 'nt':  # Windows
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

# הגדרות עמוד
st.set_page_config(
    page_title="חילוץ מידע מחשבוניות",
    page_icon="📄",
    layout="wide"
)

def preprocess_image_for_ocr(image):
    """שיפור איכות התמונה לפני OCR"""
    try:
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        # הגדלת רזולוציה
        scale_percent = 200
        width = int(gray.shape[1] * scale_percent / 100)
        height = int(gray.shape[0] * scale_percent / 100)
        dim = (width, height)
        upscaled = cv2.resize(gray, dim, interpolation=cv2.INTER_CUBIC)

        # שיפור ניגודיות
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(upscaled)

        # הסרת רעשים
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

        # חידוד
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # Thresholding
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # תיקון הטיה
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            if abs(angle) > 0.5:
                (h, w) = binary.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                binary = cv2.warpAffine(binary, M, (w, h),
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)

        return Image.fromarray(binary)

    except Exception as e:
        st.warning(f"⚠️ לא ניתן לעבד מקדים את התמונה: {str(e)}")
        return image

def extract_text_from_pdf(pdf_file):
    """חילוץ טקסט מקובץ PDF"""
    try:
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)

        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

        text_extracted = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text_extracted += page.get_text() + "\n"

        clean_text = text_extracted.strip()
        if len(clean_text) > 50:
            st.info("✅ PDF טקסטואלי - חולץ ישירות")
            pdf_document.close()
            return clean_text

        st.warning("⚠️ PDF לא מכיל טקסט מובנה - עובר ל-OCR...")
        st.info("🔍 מבצע OCR על התמונות ב-PDF...")

        text = ""
        for page_num in range(pdf_document.page_count):
            st.info(f"📄 מעבד עמוד {page_num + 1}/{pdf_document.page_count}...")

            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            preprocessed_image = preprocess_image_for_ocr(image)
            custom_config = r'--oem 3 --psm 3 -c preserve_interword_spaces=1'
            text += pytesseract.image_to_string(preprocessed_image, lang='heb+eng', config=custom_config) + "\n"

        pdf_document.close()
        return text

    except Exception as e:
        st.error(f"❌ שגיאה בעיבוד PDF: {str(e)}")
        return None

def extract_text_from_image(image_file):
    """חילוץ טקסט מתמונה"""
    try:
        image = Image.open(image_file)
        preprocessed_image = preprocess_image_for_ocr(image)
        custom_config = r'--oem 3 --psm 3 -c preserve_interword_spaces=1'
        text = pytesseract.image_to_string(preprocessed_image, lang='heb+eng', config=custom_config)
        return text
    except Exception as e:
        st.error(f"שגיאה בעיבוד תמונה: {str(e)}")
        return None

def clean_number(num_str):
    """ניקוי מספר - הסרת פסיקים ותווים מיוחדים"""
    if not num_str:
        return None
    # הסרת פסיקים, רווחים וסימנים אחרים
    cleaned = re.sub(r'[,\s]', '', str(num_str))
    return cleaned

def extract_supplier_name(text):
    """חילוץ שם ספק - משופר"""
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # חפש בשורות הראשונות
    for i, line in enumerate(lines[:10]):
        # דלג על שורות עם רק מספרים או תאריכים
        if re.match(r'^[\d\s\/\-\.]+$', line):
            continue
        # חפש שורה עם טקסט משמעותי
        if len(line) > 3 and not line.isdigit():
            # אם יש בע"מ או בע״מ או ltd זה כנראה שם החברה
            if re.search(r'בע[״\'\"]מ|בע״מ|בע\'מ|ltd|inc|corp', line, re.IGNORECASE):
                return line
            # אם זו השורה הראשונה עם טקסט משמעותי
            if i < 5:
                return line

    return ""

def extract_company_number(text):
    """חילוץ ח.פ - מספר חברה - משופר משמעותית"""
    patterns = [
        # ח.פ עם רווחים ופיסוק שונים
        r'ח[״\'\.\s]*פ[\.:\s]*(\d{9})',  # ח.פ 514417104
        r'ח[״\'\.\s]*פ[\.:\s]+מס[׳\']?\s*(\d{9})',
        r'ח[״\'\.\s]*פ\s*(\d{9})',

        # חפ ללא נקודות
        r'חפ[\.:\s]*(\d{9})',
        r'חפ\s+(\d{9})',

        # company number
        r'company\s+no\.?\s*[:\s]*(\d{9})',
        r'c\.?p\.?\s*[:\s]*(\d{9})',

        # מספר חברה
        r'מספר\s+חברה\s*[:\s]*(\d{9})',
        r'מס[׳\']?\s*חברה\s*[:\s]*(\d{9})',

        # Pattern כללי - 9 ספרות אחרי ח או cp
        r'(?:ח\s*\.?\s*פ|חפ|cp)\s*[:\s]*(\d{9})',
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            number = match.group(1)
            # וודא ש-9 ספרות
            if len(number) == 9:
                return number

    # נסה למצוא 9 ספרות ליד המילים ח.פ (גם אם יש רווחים)
    lines = text.split('\n')
    for line in lines:
        if re.search(r'ח[\.״\'\s]*פ', line, re.IGNORECASE):
            # חפש 9 ספרות באותה שורה
            numbers = re.findall(r'\d{9}', line)
            if numbers:
                return numbers[0]

    return ""

def extract_vat_number(text):
    """חילוץ מספר עוסק מורשה - משופר"""
    patterns = [
        # מספר עוסק
        r'מס[׳\']?\s*[:\s]*(\d{9,10})',  # מס' : 510982325
        r'עוסק\s*מורשה\s*[:\s]*(\d{9,10})',
        r'ע[״\']מ\s*[:\s]*(\d{9,10})',

        # VAT number
        r'vat\s*no\.?\s*[:\s]*(\d{9,10})',
        r'tax\s*id\s*[:\s]*(\d{9,10})',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            number = match.group(1)
            # מספר עוסק יכול להיות 9-10 ספרות
            if 9 <= len(number) <= 10:
                return number

    # חפש בשורות
    lines = text.split('\n')
    for line in lines:
        if re.search(r'מס[׳\']|עוסק', line, re.IGNORECASE):
            numbers = re.findall(r'\d{9,10}', line)
            if numbers:
                return numbers[0]

    return ""

def extract_invoice_number(text):
    """חילוץ מספר חשבונית - משופר משמעותית"""
    patterns = [
        # חשבון עסקה - הכי נפוץ ב-Priority/חשבשבת
        r'חשבון\s+עסקה\s+מס[׳\']?\s*[:\s]*(\d+)',  # חשבון עסקה מס' : 80064
        r'חשבון\s+עסקה\s*[:\s]*(\d+)',  # חשבון עסקה 80064

        # חשבונית רגילה
        r'חשבונית\s+מס[׳\']?\s*[:\s]*(\d+)',
        r'מס[׳\']\s*חשבונית\s*[:\s]*(\d+)',
        r'חשבונית\s*מספר\s*[:\s]*(\d+)',

        # מקור/מסמך
        r'מקור\s*[:\s]*(\d+)',
        r'מסמך\s*מס[׳\']?\s*[:\s]*(\d+)',

        # אנגלית
        r'invoice\s*(?:no\.?|number|#)\s*[:\s]*(\d+)',
        r'doc(?:ument)?\s*(?:no\.?|#)\s*[:\s]*(\d+)',

        # Pattern כללי - מספר בודד בשורה עם "חשבון" או "חשבונית"
        r'(?:חשבונ[ית]+|חשבון)\s+[^\d]*?(\d{4,8})',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            number = match.group(1)
            # מספר חשבונית בדרך כלל 4-8 ספרות
            if 3 <= len(number) <= 10:
                return number

    # חפש בשורות הראשונות
    lines = text.split('\n')[:15]
    for line in lines:
        if re.search(r'חשבון|חשבונית|invoice|מקור', line, re.IGNORECASE):
            # חפש מספר באותה שורה (4-8 ספרות)
            numbers = re.findall(r'\b(\d{4,8})\b', line)
            if numbers:
                return numbers[0]

    return ""

def extract_total_amount(text):
    """חילוץ סכום כולל לתשלום - משופר"""
    patterns = [
        # סה"כ לתשלום - הכי חשוב
        r'סה[״\']כ\s+לתשלום\s*[:\s]*([\d,]+\.?\d*)',  # סה"כ לתשלום: 1,622.50
        r'לתשלום\s*[:\s]*([\d,]+\.?\d*)',
        r'סכום\s+לתשלום\s*[:\s]*([\d,]+\.?\d*)',

        # Total
        r'total\s+(?:amount|payment)\s*[:\s]*([\d,]+\.?\d*)',
        r'total\s*[:\s]*([\d,]+\.?\d*)',
        r'amount\s+due\s*[:\s]*([\d,]+\.?\d*)',

        # סה"כ כללי
        r'סה[״\']כ\s*[:\s]*([\d,]+\.?\d*)',
        r'סך\s*הכל\s*[:\s]*([\d,]+\.?\d*)',
        r'סכום\s+כולל\s*[:\s]*([\d,]+\.?\d*)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            amount = match.group(1)
            # נקה פסיקים
            amount_clean = clean_number(amount)
            # וודא שזה מספר תקין
            try:
                float_amount = float(amount_clean)
                if float_amount > 0:
                    return amount_clean
            except:
                continue

    # חפש בשורות האחרונות
    lines = text.split('\n')
    for line in reversed(lines[-10:]):
        if re.search(r'לתשלום|total|סה[״\']כ', line, re.IGNORECASE):
            # חפש מספר עם נקודה עשרונית
            numbers = re.findall(r'([\d,]+\.\d{2})', line)
            if numbers:
                # קח את המספר הגדול ביותר (בדרך כלל הסכום הכולל)
                amounts = [float(clean_number(n)) for n in numbers]
                if amounts:
                    return clean_number(numbers[amounts.index(max(amounts))])

    return ""

def extract_vat(text):
    """חילוץ מע"מ - משופר"""
    patterns = [
        r'מע[״\']מ\s*[:\s]*([\d,]+\.?\d*)',  # מע"ם: 247.50
        r'מס\s*ערך\s*מוסף\s*[:\s]*([\d,]+\.?\d*)',
        r'vat\s*[:\s]*([\d,]+\.?\d*)',
        r'מעמ\s*[:\s]*([\d,]+\.?\d*)',

        # Pattern עם אחוזים
        r'(\d+\.?\d*)\s*%?\s*מע[״\']מ',
        r'מע[״\']מ\s*(\d+\.?\d*)\s*%',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            vat = match.group(1)
            vat_clean = clean_number(vat)
            try:
                float(vat_clean)
                return vat_clean
            except:
                continue

    return ""

def extract_date(text):
    """חילוץ תאריך - משופר"""
    patterns = [
        r'תאריך\s*[:\s]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',  # תאריך : 25/09/2025
        r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})',  # 25/09/2025
        r'date\s*[:\s]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

    # חפש בשורות הראשונות
    lines = text.split('\n')[:10]
    for line in lines:
        dates = re.findall(r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4}', line)
        if dates:
            return dates[0]

    return ""

def extract_phone(text):
    """חילוץ מספר טלפון"""
    patterns = [
        r'טל[פון]*[\.:\s]*(\d{2,3}[-\s]?\d{7})',
        r'טלפון[:\s]*(\d{2,3}[-\s]?\d{7})',
        r'פקס[:\s]*(\d{2,3}[-\s]?\d{7})',
        r'phone[:\s]*(\d{2,3}[-\s]?\d{7})',
        r'tel[:\s]*(\d{2,3}[-\s]?\d{7})',
        r'(\d{2,3}[-\s]?\d{7})',  # Generic phone pattern
    ]

    phones = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        phones.extend(matches)

    return ', '.join(list(dict.fromkeys(phones))[:3]) if phones else ""  # Return up to 3 unique phones

def extract_email(text):
    """חילוץ כתובת אימייל"""
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(pattern, text)
    return ', '.join(list(dict.fromkeys(emails))[:2]) if emails else ""  # Return up to 2 unique emails

def extract_address(text):
    """חילוץ כתובת"""
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # Look for address patterns
    address_keywords = [r'כתובת', r'רחוב', r'רח[׳\']', r'address', r'street']

    for i, line in enumerate(lines[:20]):
        for keyword in address_keywords:
            if re.search(keyword, line, re.IGNORECASE):
                # Try to get the address from the same line or next line
                address = re.sub(r'כתובת[:\s]*', '', line, flags=re.IGNORECASE)
                if len(address) > 5:
                    return address
                elif i + 1 < len(lines):
                    return lines[i + 1]

    return ""

def extract_bank_details(text):
    """חילוץ פרטי בנק"""
    bank_info = {}

    # Bank number
    bank_patterns = [
        r'בנק[:\s]*(\d{2,3})',
        r'מס[׳\']?\s*בנק[:\s]*(\d{2,3})',
    ]
    for pattern in bank_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            bank_info['bank_number'] = match.group(1)
            break

    # Branch number
    branch_patterns = [
        r'סניף[:\s]*(\d{3,4})',
        r'מס[׳\']?\s*סניף[:\s]*(\d{3,4})',
    ]
    for pattern in branch_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            bank_info['branch_number'] = match.group(1)
            break

    # Account number
    account_patterns = [
        r'חשבון[:\s]*(\d{5,10})',
        r'מס[׳\']?\s*חשבון[:\s]*(\d{5,10})',
        r'account[:\s]*(\d{5,10})',
    ]
    for pattern in account_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            bank_info['account_number'] = match.group(1)
            break

    return bank_info

def extract_payment_terms(text):
    """חילוץ תנאי תשלום"""
    patterns = [
        r'תנאי\s+תשלום[:\s]*([^\n]+)',
        r'אמצעי\s+תשלום[:\s]*([^\n]+)',
        r'payment\s+terms[:\s]*([^\n]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return ""

def extract_due_date(text):
    """חילוץ תאריך פירעון"""
    patterns = [
        r'תאריך\s+פירעון[:\s]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
        r'פירעון[:\s]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
        r'due\s+date[:\s]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

    return ""

def extract_discount(text):
    """חילוץ הנחה"""
    patterns = [
        r'הנחה[:\s]*([\d,]+\.?\d*)',
        r'discount[:\s]*([\d,]+\.?\d*)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return clean_number(match.group(1))

    return ""

def extract_subtotal(text):
    """חילוץ סכום לפני מע\"מ"""
    patterns = [
        r'סה[״\']כ\s+לפני\s+מע[״\']מ[:\s]*([\d,]+\.?\d*)',
        r'לפני\s+מע[״\']מ[:\s]*([\d,]+\.?\d*)',
        r'subtotal[:\s]*([\d,]+\.?\d*)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return clean_number(match.group(1))

    return ""

def extract_currency(text):
    """חילוץ מטבע"""
    currencies = {
        r'₪': 'ILS',
        r'ש[״\']ח': 'ILS',
        r'שקל': 'ILS',
        r'\$': 'USD',
        r'dollar': 'USD',
        r'€': 'EUR',
        r'euro': 'EUR',
    }

    for pattern, currency in currencies.items():
        if re.search(pattern, text, re.IGNORECASE):
            return currency

    return "ILS"  # Default to ILS

def extract_reference_number(text):
    """חילוץ מספר אסמכתא / הזמנה"""
    patterns = [
        r'הזמנה[:\s]*(\d+)',
        r'אסמכתא[:\s]*(\d+)',
        r'reference[:\s]*(\d+)',
        r'ref[:\s]*(\d+)',
        r'po[:\s]*(\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

    return ""

def extract_customer_info(text):
    """חילוץ פרטי לקוח"""
    customer = {}
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # Look for customer section
    customer_keywords = [r'לקוח', r'קונה', r'customer', r'bill\s+to']

    for i, line in enumerate(lines):
        for keyword in customer_keywords:
            if re.search(keyword, line, re.IGNORECASE):
                # Extract next few lines as customer info
                if i + 1 < len(lines):
                    customer['name'] = lines[i + 1]
                if i + 2 < len(lines) and not re.search(r'\d{9}', lines[i + 2]):
                    customer['address'] = lines[i + 2]
                break

    return customer

def extract_line_items(text):
    """חילוץ פריטים מהחשבונית"""
    lines = text.split('\n')
    items = []

    # Look for table-like structures with numbers
    for line in lines:
        # Pattern: description, quantity, price, total
        # This is a simplified version - can be enhanced
        numbers = re.findall(r'[\d,]+\.?\d*', line)
        if len(numbers) >= 2 and len(line) > 10:
            # Potential line item
            items.append(line.strip())

    return items[:10]  # Return up to 10 items

def extract_notes(text):
    """חילוץ הערות"""
    patterns = [
        r'הערות[:\s]*([^\n]+)',
        r'notes[:\s]*([^\n]+)',
        r'הערה[:\s]*([^\n]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return ""

def extract_invoice_data(text):
    """חילוץ כל המידע מהחשבונית - מורחב לכל השדות האפשריים"""

    # Extract bank details
    bank_details = extract_bank_details(text)

    # Extract customer info
    customer_info = extract_customer_info(text)

    data = {
        # שדות חובה - קריטיים
        "company_number": extract_company_number(text),
        "vat_number": extract_vat_number(text),
        "invoice_number": extract_invoice_number(text),
        "invoice_date": extract_date(text),
        "total_amount": extract_total_amount(text),
        "vat_amount": extract_vat(text),

        # פרטי ספק
        "supplier_name": extract_supplier_name(text),
        "supplier_address": extract_address(text),
        "supplier_phone": extract_phone(text),
        "supplier_email": extract_email(text),

        # פרטי תשלום
        "subtotal": extract_subtotal(text),
        "discount": extract_discount(text),
        "currency": extract_currency(text),
        "payment_terms": extract_payment_terms(text),
        "due_date": extract_due_date(text),

        # פרטי בנק
        "bank_number": bank_details.get('bank_number', ''),
        "branch_number": bank_details.get('branch_number', ''),
        "account_number": bank_details.get('account_number', ''),

        # מידע נוסף
        "reference_number": extract_reference_number(text),
        "customer_name": customer_info.get('name', ''),
        "customer_address": customer_info.get('address', ''),
        "notes": extract_notes(text),

        # פריטים (מוגבל ל-5 פריטים ראשונים לתצוגה)
        "line_items": extract_line_items(text)[:5],
    }

    return data

def validate_required_fields(data):
    """בדיקת שדות חובה"""
    missing_fields = []

    if not data.get("company_number") and not data.get("vat_number"):
        missing_fields.append("מספר חברה / עוסק מורשה (ח.פ)")
    if not data.get("invoice_number"):
        missing_fields.append("מספר חשבונית")
    if not data.get("invoice_date"):
        missing_fields.append("תאריך חשבונית")
    if not data.get("total_amount"):
        missing_fields.append("סכום כולל")
    if not data.get("vat_amount"):
        missing_fields.append("מע\"מ")

    return missing_fields

def get_pdf_as_base64(pdf_file):
    """המרת PDF ל-Base64 להצגה"""
    try:
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        return base64_pdf
    except:
        return None

def create_html_form(data):
    """יצירת טופס HTML עם כל הנתונים שחולצו"""

    # בדיקת שדות חובה
    missing_fields = validate_required_fields(data)
    validation_class = "invalid" if missing_fields else "valid"

    # Prepare line items display
    line_items_html = ""
    if data.get('line_items'):
        line_items_html = "<br>".join([f"• {item}" for item in data.get('line_items', [])])

    html = f"""
    <style>
        .invoice-form {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            direction: rtl;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        .form-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .form-group {{
            margin-bottom: 20px;
        }}
        .form-label {{
            display: block;
            font-weight: bold;
            margin-bottom: 8px;
            color: #333;
            font-size: 14px;
        }}
        .form-label.required::after {{
            content: " *";
            color: #e74c3c;
        }}
        .form-input {{
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 16px;
            transition: border-color 0.3s;
            box-sizing: border-box;
        }}
        .form-textarea {{
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
            min-height: 100px;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        .form-input:focus {{
            outline: none;
            border-color: #3498db;
        }}
        .form-input.required-field {{
            border-color: #3498db;
            background-color: #f0f8ff;
        }}
        .form-input.missing-field {{
            border-color: #e74c3c;
            background-color: #fff5f5;
        }}
        .form-input.optional-field {{
            border-color: #27ae60;
            background-color: #f0fff4;
        }}
        .validation-status {{
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
            font-weight: bold;
        }}
        .validation-status.valid {{
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        .validation-status.invalid {{
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
        .missing-fields-list {{
            margin-top: 10px;
            padding-right: 20px;
        }}
        .form-title {{
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #2c3e50;
            text-align: center;
        }}
        .form-section {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #eee;
        }}
        .form-section-title {{
            font-size: 18px;
            font-weight: bold;
            color: #34495e;
            margin-bottom: 15px;
            background: linear-gradient(to right, #3498db, #2ecc71);
            color: white;
            padding: 10px;
            border-radius: 5px;
        }}
        .json-output {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin-top: 20px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            white-space: pre-wrap;
            direction: ltr;
            text-align: left;
        }}
        .line-items-display {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border: 2px solid #ddd;
            font-size: 13px;
            line-height: 1.6;
        }}
    </style>

    <div class="invoice-form">
        <div class="form-title">📄 טופס חשבונית מלא</div>

        <div class="validation-status {validation_class}">
            {'✅ כל השדות הנדרשים זוהו בהצלחה!' if not missing_fields else '⚠️ חסרים שדות חובה:'}
            {f'<ul class="missing-fields-list">{"".join([f"<li>{field}</li>" for field in missing_fields])}</ul>' if missing_fields else ''}
        </div>

        <!-- שדות חובה -->
        <div class="form-section">
            <div class="form-section-title">📋 מידע בסיסי (שדות חובה)</div>

            <div class="form-row">
                <div class="form-group">
                    <label class="form-label required">מספר חברה (ח.פ)</label>
                    <input type="text" class="form-input {'required-field' if data.get('company_number') else 'missing-field'}"
                           value="{data.get('company_number', '')}" placeholder="9 ספרות">
                </div>

                <div class="form-group">
                    <label class="form-label">מספר עוסק מורשה</label>
                    <input type="text" class="form-input {'optional-field' if data.get('vat_number') else ''}"
                           value="{data.get('vat_number', '')}" placeholder="9-10 ספרות">
                </div>
            </div>

            <div class="form-row">
                <div class="form-group">
                    <label class="form-label required">מספר חשבונית</label>
                    <input type="text" class="form-input {'required-field' if data.get('invoice_number') else 'missing-field'}"
                           value="{data.get('invoice_number', '')}" placeholder="מספר חשבונית">
                </div>

                <div class="form-group">
                    <label class="form-label required">תאריך חשבונית</label>
                    <input type="text" class="form-input {'required-field' if data.get('invoice_date') else 'missing-field'}"
                           value="{data.get('invoice_date', '')}" placeholder="DD/MM/YYYY">
                </div>
            </div>

            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">מספר אסמכתא / הזמנה</label>
                    <input type="text" class="form-input {'optional-field' if data.get('reference_number') else ''}"
                           value="{data.get('reference_number', '')}" placeholder="מספר הזמנה">
                </div>

                <div class="form-group">
                    <label class="form-label">תאריך פירעון</label>
                    <input type="text" class="form-input {'optional-field' if data.get('due_date') else ''}"
                           value="{data.get('due_date', '')}" placeholder="DD/MM/YYYY">
                </div>
            </div>
        </div>

        <!-- פרטי ספק -->
        <div class="form-section">
            <div class="form-section-title">🏢 פרטי ספק</div>

            <div class="form-group">
                <label class="form-label">שם ספק</label>
                <input type="text" class="form-input {'optional-field' if data.get('supplier_name') else ''}"
                       value="{data.get('supplier_name', '')}" placeholder="שם ספק">
            </div>

            <div class="form-group">
                <label class="form-label">כתובת</label>
                <input type="text" class="form-input {'optional-field' if data.get('supplier_address') else ''}"
                       value="{data.get('supplier_address', '')}" placeholder="כתובת ספק">
            </div>

            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">טלפון</label>
                    <input type="text" class="form-input {'optional-field' if data.get('supplier_phone') else ''}"
                           value="{data.get('supplier_phone', '')}" placeholder="מספר טלפון">
                </div>

                <div class="form-group">
                    <label class="form-label">אימייל</label>
                    <input type="text" class="form-input {'optional-field' if data.get('supplier_email') else ''}"
                           value="{data.get('supplier_email', '')}" placeholder="כתובת אימייל">
                </div>
            </div>
        </div>

        <!-- סכומים -->
        <div class="form-section">
            <div class="form-section-title">💰 פרטי תשלום</div>

            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">סכום לפני מע"מ</label>
                    <input type="text" class="form-input {'optional-field' if data.get('subtotal') else ''}"
                           value="{data.get('subtotal', '')}" placeholder="₪">
                </div>

                <div class="form-group">
                    <label class="form-label required">מע"מ</label>
                    <input type="text" class="form-input {'required-field' if data.get('vat_amount') else 'missing-field'}"
                           value="{data.get('vat_amount', '')}" placeholder="₪">
                </div>
            </div>

            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">הנחה</label>
                    <input type="text" class="form-input {'optional-field' if data.get('discount') else ''}"
                           value="{data.get('discount', '')}" placeholder="₪">
                </div>

                <div class="form-group">
                    <label class="form-label required">סכום כולל</label>
                    <input type="text" class="form-input {'required-field' if data.get('total_amount') else 'missing-field'}"
                           value="{data.get('total_amount', '')}" placeholder="₪">
                </div>
            </div>

            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">מטבע</label>
                    <input type="text" class="form-input {'optional-field' if data.get('currency') else ''}"
                           value="{data.get('currency', 'ILS')}" placeholder="ILS/USD/EUR">
                </div>

                <div class="form-group">
                    <label class="form-label">תנאי תשלום</label>
                    <input type="text" class="form-input {'optional-field' if data.get('payment_terms') else ''}"
                           value="{data.get('payment_terms', '')}" placeholder="תנאי תשלום">
                </div>
            </div>
        </div>

        <!-- פרטי בנק -->
        <div class="form-section">
            <div class="form-section-title">🏦 פרטי בנק</div>

            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">מספר בנק</label>
                    <input type="text" class="form-input {'optional-field' if data.get('bank_number') else ''}"
                           value="{data.get('bank_number', '')}" placeholder="2-3 ספרות">
                </div>

                <div class="form-group">
                    <label class="form-label">מספר סניף</label>
                    <input type="text" class="form-input {'optional-field' if data.get('branch_number') else ''}"
                           value="{data.get('branch_number', '')}" placeholder="3-4 ספרות">
                </div>
            </div>

            <div class="form-group">
                <label class="form-label">מספר חשבון</label>
                <input type="text" class="form-input {'optional-field' if data.get('account_number') else ''}"
                       value="{data.get('account_number', '')}" placeholder="מספר חשבון בנק">
            </div>
        </div>

        <!-- פרטי לקוח -->
        <div class="form-section">
            <div class="form-section-title">👤 פרטי לקוח</div>

            <div class="form-group">
                <label class="form-label">שם לקוח</label>
                <input type="text" class="form-input {'optional-field' if data.get('customer_name') else ''}"
                       value="{data.get('customer_name', '')}" placeholder="שם לקוח">
            </div>

            <div class="form-group">
                <label class="form-label">כתובת לקוח</label>
                <input type="text" class="form-input {'optional-field' if data.get('customer_address') else ''}"
                       value="{data.get('customer_address', '')}" placeholder="כתובת לקוח">
            </div>
        </div>

        <!-- פריטים -->
        {f'''<div class="form-section">
            <div class="form-section-title">📦 פריטים בחשבונית</div>
            <div class="line-items-display">
                {line_items_html if line_items_html else "לא זוהו פריטים"}
            </div>
        </div>''' if data.get('line_items') else ''}

        <!-- הערות -->
        <div class="form-section">
            <div class="form-section-title">📝 הערות</div>

            <div class="form-group">
                <textarea class="form-textarea" placeholder="הערות נוספות">{data.get('notes', '')}</textarea>
            </div>
        </div>

        <div class="json-output">
{json.dumps(data, ensure_ascii=False, indent=2)}
        </div>
    </div>
    """

    return html

# ממשק המשתמש
st.title("📄 מערכת חילוץ מידע מחשבוניות - תצוגה משופרת")
st.markdown("---")

with st.expander("ℹ️ הוראות שימוש"):
    st.markdown("""
    1. העלה קובץ PDF או תמונה של חשבונית
    2. המערכת תחלץ את כל המידע האפשרי באופן אוטומטי
    3. החשבונית תוצג לצד טופס מלא עם הנתונים שחולצו

    **שדות חובה (*):**
    - ✅ מספר חברה / עוסק מורשה (ח.פ) - 9-10 ספרות
    - ✅ מספר חשבונית
    - ✅ תאריך חשבונית
    - ✅ סכום כולל
    - ✅ מע"מ

    **שדות נוספים שמחולצים אוטומטית:**
    - 📋 פרטי ספק: שם, כתובת, טלפון, אימייל
    - 💰 פרטי תשלום: סכום לפני מע"מ, הנחה, מטבע, תנאי תשלום, תאריך פירעון
    - 🏦 פרטי בנק: מספר בנק, סניף, חשבון
    - 👤 פרטי לקוח: שם, כתובת
    - 📦 פריטים בחשבונית
    - 📝 הערות ומספרי אסמכתא

    **פורמטים נתמכים:** PDF, JPG, JPEG, PNG
    """)

uploaded_file = st.file_uploader(
    "בחר קובץ חשבונית",
    type=['pdf', 'jpg', 'jpeg', 'png'],
    help="גרור קובץ לכאן או לחץ לבחירה"
)

if uploaded_file is not None:
    st.info(f"📎 **שם הקובץ:** {uploaded_file.name} | 📏 **גודל:** {uploaded_file.size / 1024:.2f} KB")

    process_button = st.button("🚀 עבד חשבונית", type="primary", use_container_width=True)

    if process_button:
        with st.spinner("מעבד את הקובץ... אנא המתן"):
            # חילוץ טקסט
            if uploaded_file.type == 'application/pdf':
                text = extract_text_from_pdf(uploaded_file)
            else:
                text = extract_text_from_image(uploaded_file)

            if text:
                # חילוץ נתונים
                invoice_data = extract_invoice_data(text)

                # בדיקת תקינות
                missing_fields = validate_required_fields(invoice_data)

                if missing_fields:
                    st.warning(f"⚠️ חסרים שדות חובה: {', '.join(missing_fields)}")
                else:
                    st.success("✅ כל השדות החובה זוהו בהצלחה!")

                st.markdown("---")

                # תצוגה side-by-side
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("📄 חשבונית מקורית")

                    if uploaded_file.type == 'application/pdf':
                        # הצגת PDF
                        uploaded_file.seek(0)
                        base64_pdf = get_pdf_as_base64(uploaded_file)
                        if base64_pdf:
                            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
                            st.markdown(pdf_display, unsafe_allow_html=True)
                    else:
                        # הצגת תמונה
                        uploaded_file.seek(0)
                        image = Image.open(uploaded_file)
                        st.image(image, use_container_width=True)

                with col2:
                    st.subheader("📝 טופס נתונים")

                    # הצגת הטופס
                    html_form = create_html_form(invoice_data)
                    components.html(html_form, height=2000, scrolling=True)

                # כפתור הורדה
                st.markdown("---")
                json_str = json.dumps(invoice_data, ensure_ascii=False, indent=2)
                json_bytes = json_str.encode('utf-8')
                st.download_button(
                    label="💾 הורד נתונים כ-JSON",
                    data=json_bytes,
                    file_name=f"invoice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

                # טקסט מלא
                with st.expander("📝 הצג טקסט מלא שחולץ מהחשבונית"):
                    st.text_area("טקסט", text, height=300)

else:
    st.info("👆 העלה קובץ חשבונית כדי להתחיל")

    with st.expander("🎯 דוגמה לפלט JSON מלא"):
        example_json = {
            "company_number": "514417104",
            "vat_number": "510982325",
            "invoice_number": "80064",
            "invoice_date": "25/09/2025",
            "total_amount": "1622.50",
            "vat_amount": "247.50",
            "supplier_name": "מילגם בע\"מ",
            "supplier_address": "רחוב הרצל 123, תל אביב",
            "supplier_phone": "03-1234567",
            "supplier_email": "info@example.com",
            "subtotal": "1375.00",
            "discount": "0",
            "currency": "ILS",
            "payment_terms": "שוטף + 30",
            "due_date": "25/10/2025",
            "bank_number": "12",
            "branch_number": "345",
            "account_number": "678901",
            "reference_number": "PO-12345",
            "customer_name": "חברת XYZ בע\"מ",
            "customer_address": "רחוב ויצמן 45, חיפה",
            "notes": "חשבונית עבור שירותי ייעוץ",
            "line_items": [
                "שירות ייעוץ - 10 שעות - 1000.00",
                "מוצר A - 5 יח' - 375.00"
            ]
        }
        st.json(example_json)

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    גרסה 3.0 - Side-by-Side View | נוצר עם ❤️ | Tesseract OCR + Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
