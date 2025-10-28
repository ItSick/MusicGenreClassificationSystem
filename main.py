import streamlit as st
import pandas as pd
import numpy as np

# --- Configuration and Utility Functions ---

def set_page_config():
    """Sets the Streamlit page configuration."""
    st.set_page_config(
        page_title="סיפור הנתונים של אודיו ML", # Changed to Hebrew
        layout="wide",
        initial_sidebar_state="expanded"
    )

def profile_tab():
    """Content for the Profile Tab."""
    st.header("פרופיל הפרויקט: סיווג אודיו") # Changed to Hebrew
    st.markdown("""
        <div style='padding: 20px; border-radius: 10px; background-color: #f0f2f6; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);'>
        <p style='font-size: 1.1em; font-weight: bold;'>סקירה כללית</p>
        <p>יישום זה מתאר פרויקט למידת מכונה המתמקד ב**חילוץ וסיווג תכונות אודיו**. המטרה היא לעבד אותות אודיו גולמיים, לחלץ תכונות מספריות משמעותיות באמצעות כלים כמו Librosa, ולאמן מודלים של למידת מכונה מונחית (Random Forest ו-Neural Networks) לסיווג האודיו (למשל, סיווג ז'אנר, זיהוי אירועי קול, או זיהוי מצב דובר).</p>
        <p>לשונית פרופיל זו משמשת בדרך כלל להצגת הצוות, יעדי הפרויקט והיקף העבודה.</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("יעדים מרכזיים") # Changed to Hebrew
    st.markdown(
        """
        - הצלחה בהפיכת נתוני אודיו מסדרת זמן למערך תכונות חזק.
        - השוואת ביצועים בין שתי ארכיטקטורות מודלים נפרדות (מבוססות עץ ולמידה עמוקה).
        - מתן ניתוח מקיף של תוצאות המודל באמצעות מדדי סיווג סטנדרטיים.
        """
    )
    st.image("https://placehold.co/800x250/2563EB/ffffff?text=Project+Visual+Placeholder")

# --- Nested Tabs Content ---

def librosa_content():
    """Content for the Librosa Tab."""
    st.subheader("1. Librosa: ערכת הכלים לתכונות אודיו") # Changed to Hebrew
    st.markdown("""
    Librosa היא ספריית Python לניתוח מוזיקה ואודיו. היא מספקת את אבני הבניין ליצירת מערכות לאחזור מידע מוזיקלי (MIR). בלמידת מכונה, אודיו גולמי (צורת גל סדרת זמן) משמש לעתים רחוקות ישירות. במקום זאת, Librosa משמשת לחילוץ **תכונות מספריות רלוונטיות לתפיסה**.
    """)

    st.markdown("### פונקציות ותכונות מפתח") # Changed to Hebrew
    st.markdown(
        """
        - **טעינת אודיו:** טוענת בקלות קובצי אודיו למערך NumPy עם קצב דגימה מוגדר.
        - **MFCCs (מקדם צפסטרום תדר-מל):** התכונה הנפוצה ביותר. היא לוכדת את הצורה הספקטרלית של הצליל, שהיא חיונית להבחנה בין גוון לצבע קול, מה שהופך אותם לאידיאליים למשימות כמו סיווג ז'אנר או כלי נגינה.
        - **מרכז ספקטרלי (Spectral Centroid):** מציין היכן ממוקם 'מרכז המסה' של הספקטרום. צלילים עם מרכזים גבוהים יותר נוטים להיות בהירים יותר.
        - **קצב חציית אפס (ZCR):** הקצב שבו האות משנה סימן (מחיוב לשלילי או להיפך). ZCR גבוה מצביע לעיתים קרובות על צלילים רועשים או קצביים.
        """
    )
    st.code("import librosa\n# Load the audio file\ny, sr = librosa.load(audio_path, sr=44100)\n# Extract MFCCs\nmfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)", language='python')
    st.markdown("")


def data_content():
    """Content for the Data Tab."""
    st.subheader("2. הבנת מערך נתוני האודיו") # Changed to Hebrew
    st.markdown("""
    מערך הנתונים לפרויקט זה מורכב מקטעי אודיו מתויגים. האתגר העיקרי באודיו ML הוא גישור על הפער בין אות האודיו הגולמי לפורמט מספרי מובנה המתאים לאלגוריתמים סטנדרטיים של למידת מכונה.
    """)

    st.markdown("### מבנה ועיבוד נתונים") # Changed to Hebrew
    st.markdown(
        """
        1.  **נתונים גולמיים:** אוסף של קובצי אודיו בתחום הזמן (למשל, `.wav`) עם תוויות מחלקה מתאימות (למשל, 'רוק', 'ג'אז', 'דיבור', 'שקט').
        2.  **חילוץ תכונות:** עבור כל קובץ אודיו, Librosa משמשת לחישוב תכונות רבות (MFCCs, מאפיינים ספקטרליים, תכונות קצב).
        3.  **צבירה:** מכיוון שתכונות אודיו הן לרוב תלויות זמן (מטריצה של ערכים לאורך זמן), מדדים סטטיסטיים (ממוצע, שונות, מינימום, מקסימום) מחושבים לאורך ציר הזמן ליצירת וקטור תכונות יחיד עבור כל קובץ אודיו.
        4.  **מערך נתונים סופי:** מערך נתונים טבלאי שבו:
            - **שורות:** מייצגות דגימות אודיו בודדות (יחידת הסיווג).
            - **עמודות:** הן תכונות האודיו המצטברות (למשל, 'mean\_mfcc\_1', 'variance\_zcr', וכו').
            - **עמודת יעד:** התווית הקטגורית (המחלקה שיש לחזות).
        """
    )
    # Mock DataFrame visualization
    st.markdown("מבט חטוף על מערך התכונות הסופי:") # Changed to Hebrew
    data = {
        'mean_mfcc_1': np.random.rand(5) * 50,
        'variance_zcr': np.random.rand(5) * 0.1,
        'mean_chroma': np.random.rand(5),
        'target_class': ['ג\'אז', 'רוק', 'פופ', 'ג\'אז', 'קלאסי'] # Changed classes to Hebrew
    }
    df = pd.DataFrame(data)
    st.dataframe(df)


def models_training_content():
    """Content for the Models Training Tab."""
    st.subheader("3. תהליך אימון המודלים (Random Forest & Neural Network)") # Changed to Hebrew
    st.markdown("""
    שתי משפחות שונות של מודלים נבחרו כדי להשוות ביצועים ולחקור יכולות למידה שונות: אנסמבל חזק מבוסס עץ (Random Forest) וארכיטקטורת למידה עמוקה מורכבת (Neural Network).
    """)

    col_rf, col_nn = st.columns(2)

    with col_rf:
        st.markdown("#### Random Forest (RF)")
        st.markdown(
            """
            - **סוג:** למידת אנסמבל, במיוחד **Bagging** (צבירה באמצעות אתחול).
            - **איך זה עובד:** הוא בונה מספר רב של עצי החלטה במהלך האימון. לסיווג, הפלט הוא המחלקה שנבחרה על ידי רוב העצים (הצבעת רוב).
            - **יתרונות:** יעיל מאוד, חסין מפני התאמת יתר (overfitting), אימון מהיר, ומספק מנגנון ברור ל**חשיבות תכונות**.
            - **תהליך אימון:**
                1.  תתי-קבוצות אקראיות של נתוני האימון (עם החלפה) משמשות לאימון עצי החלטה בודדים.
                2.  בכל פיצול בעץ, רק תת-קבוצה אקראית של תכונות נשקלת.
                3.  החיזוי הסופי משלב את התוצאות של כל העצים הבודדים.
            """
        )

    with col_nn:
        st.markdown("#### Neural Network (NN - Multilayer Perceptron)")
        st.markdown(
            """
            - **סוג:** למידה עמוקה (במיוחד, רשת הזנה קדמית מחוברת במלואה).
            - **איך זה עובד:** מורכבת משכבות קלט, נסתרות ופלט. המידע זורם קדימה, וטעויות משמשות להתאמת משקולות באמצעות **Backpropagation** (הפצה לאחור).
            - **אקטיבציה:** משתמש בדרך כלל ב**ReLU** בשכבות נסתרות לאי-לינאריות וב**Softmax** בשכבת הפלט עבור ציוני הסתברות רב-מחלקיים.
            - **תהליך אימון:**
                1.  **אתחול:** המשקולות מוגדרות באופן אקראי.
                2.  **מעבר קדמי:** הנתונים עוברים דרך הרשת ליצירת תחזיות.
                3.  **חישוב הפסד:** ההפרש בין התחזיות לתוויות בפועל מכומת (למשל, אנטרופיה צולבת קטגורית).
                4.  **הפצה לאחור:** ההפסד נשלח אחורה דרך הרשת, מעדכן משקולות באמצעות **מייעל** (Optimizer) (למשל, Adam או SGD) כדי למזער טעויות עתידיות.
            """
        )
    st.markdown("")


def result_content():
    """Content for the Result Tab."""
    st.subheader("4. מדדי ביצועים ויכולת פרשנות") # Changed to Hebrew
    st.markdown("""
    כדי להעריך מודל סיווג במלואו, אנו מסתכלים מעבר לדיוק פשוט ובודקים מספר מדדים מיוחדים וכלי פרשנות.
    """)

    st.markdown("### מדדי סיווג (F1, Recall, Precision)") # Changed to Hebrew
    st.markdown(
        """
        - **דיוק (Precision):** $TP / (TP + FP)$. מכל הדגימות שהמודל **חזה כחיוביות**, כמה היו **באמת חיוביות**? (ממזער False Positives)
        - **רגישות (Recall / Sensitivity):** $TP / (TP + FN)$. מכל הדגימות שהיו **באמת חיוביות**, כמה זיהה המודל **בצורה נכונה**? (ממזער False Negatives)
        - **ציון F1:** $2 \times (Precision \times Recall) / (Precision + Recall)$. הממוצע ההרמוני של Precision ו-Recall. זהו מדד שימושי כאשר יש צורך באיזון בין מזעור False Positives ו-False Negatives.
        """
    )

    st.markdown("### מטריצת בלבול (Confusion Matrix)") # Changed to Hebrew
    st.markdown(
        """
        מטריצת הבלבול היא סיכום ויזואלי של תוצאות החיזוי. היא מציגה את מספר החיזויים הנכונים והלא נכונים המחולקים לפי כל מחלקה.
        - **אלכסון:** True Positives (סיווגים נכונים).
        - **מחוץ לאלכסון:** False Positives (שגיאה מסוג I) ו-False Negatives (שגיאה מסוג II).
        """
    )
    # Mock Confusion Matrix Display
    st.code("from sklearn.metrics import confusion_matrix\nconf_matrix = confusion_matrix(y_true, y_pred)", language='python')
    st.markdown("")

    st.markdown("### חשיבות תכונות (RF ספציפי)") # Changed to Hebrew
    st.markdown(
        """
        עבור מודל Random Forest, אנו יכולים לכמת את התרומה היחסית של כל תכונת אודיו שחולצה (MFCCs, ZCR, וכו') להחלטת הסיווג הסופית. זה קריטי ליכולת הפרשנות של המודל ולהבנה **אילו מאפיינים אקוסטיים** באמת מניעים את ההבדל בין מחלקות. תכונות בעלות חשיבות גבוהה יותר הן אלו שמובילות לפיצולים טובים יותר בעצי ההחלטה.
        """
    )
    # Mock Feature Importance Chart Placeholder
    st.bar_chart({'Feature_1': 0.45, 'Feature_2': 0.30, 'Feature_3': 0.15, 'Feature_4': 0.10})


def data_story_tab():
    """Content for the main Data Story Tab with nested tabs."""
    st.title("Data Story") # Changed to Hebrew

    # Nested tabs definition
    tab_librosa, tab_data, tab_models, tab_results = st.tabs([
        "Librosa",
        "Data",
        "Models", 
        "Results"
    ])

    with tab_librosa:
        librosa_content()
    with tab_data:
        data_content()
    with tab_models:
        models_training_content()
    with tab_results:
        result_content()


# --- Main Application Logic ---

def main():
    """Runs the main Streamlit application."""
    set_page_config()
    
    # Custom CSS for RTL (Right-to-Left) direction and text alignment for Hebrew
    # This targets the main content area of Streamlit and forces RTL direction
    st.markdown("""
        <style>
        /* Force RTL direction on the main container for proper layout flow */
        .stApp {
            direction: rtl;
        }
        /* Ensure all text within the main content area aligns to the right */
        section.main {
            text-align: right;
        }
        /* Specifically target Streamlit markdown and text elements */
        .stMarkdown, .stText, .stHeader, .stSubheader, p {
            text-align: right;
        }
        /* Fix the alignment of the DataFrame title/caption if present */
        .dataframe-container p {
            text-align: right;
        }
        /* Ensure specific elements (like code blocks, or LTR tables) remain LTR if required */
        div[data-testid="stCodeBlock"], table {
            direction: ltr; 
            text-align: left;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("ניווט") # Changed to Hebrew
    
    # Main Tabs definition
    tab_profile, tab_data_story = st.tabs(["Profile", "Data Story"]) # Changed to Hebrew

    with tab_profile:
        profile_tab()
    with tab_data_story:
        data_story_tab()

if __name__ == "__main__":
    main()
