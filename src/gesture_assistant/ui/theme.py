from src.gesture_assistant.domain.enums import QualityStatus

MAIN_WINDOW_TITLE = "GestureAssistant"

STATUS_TEXTS = {
    QualityStatus.VALID: "Śledzenie",
    QualityStatus.NO_PERSON: "Brak osoby",
    QualityStatus.LOW_VISIBILITY: "Słaba widoczność",
    QualityStatus.OUT_OF_FRAME: "Osoba poza kadrze",
    QualityStatus.MULTIPLE_PEOPLE: "Wiele osób w kadrze",
    QualityStatus.WRONG_VIEW: "Nieprawidłowy widok",
    QualityStatus.CAMERA_ERROR: "Błąd kamery",
}

BACKGROUND_COLOR = "#101418"
FOREGROUND_COLOR = "#e6edf3"
ACCENT_COLOR = "#4c8dff"
ERROR_COLOR = "#ff6b6b"

NO_PREVIEW_TEXT = "Brak podglądu"
MISSING_MODEL_TEXT = "Brak modelu MediaPipe.\nPobierz model zgodnie z instrukcją w README."


def get_stylesheet() -> str:
    return f"""
    QMainWindow {{
        background-color: {BACKGROUND_COLOR};
        color: {FOREGROUND_COLOR};
    }}

    QLabel {{
        color: {FOREGROUND_COLOR};
    }}

    QPushButton {{
        background-color: {ACCENT_COLOR};
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
    }}

    QPushButton:disabled {{
        background-color: #3b4451;
        color: #9aa4b1;
    }}

    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
        background-color: #1b2129;
        color: {FOREGROUND_COLOR};
        border: 1px solid #3b4451;
        border-radius: 4px;
        padding: 6px;
    }}

    QTabWidget::pane {{
        border: 1px solid #3b4451;
        background-color: {BACKGROUND_COLOR};
    }}

    QTabBar::tab {{
        background-color: #1b2129;
        color: {FOREGROUND_COLOR};
        padding: 8px 16px;
        margin-right: 2px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }}

    QTabBar::tab:selected {{
        background-color: {ACCENT_COLOR};
        color: #ffffff;
    }}

    QGroupBox {{
        border: 1px solid #3b4451;
        border-radius: 6px;
        margin-top: 12px;
        padding-top: 10px;
        color: {FOREGROUND_COLOR};
        font-weight: bold;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 5px;
    }}

    QCheckBox {{
        color: {FOREGROUND_COLOR};
    }}
    """
