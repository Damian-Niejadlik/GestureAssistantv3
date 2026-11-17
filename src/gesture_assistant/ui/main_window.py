import math
import time
from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.gesture_assistant.config.schema import validate_config
from src.gesture_assistant.core.action_runtime import ActionRuntime
from src.gesture_assistant.core.config_manager import ConfigManager, defaults
from src.gesture_assistant.core.context_detector import MODES
from src.gesture_assistant.integrations import ACTION_CATALOG
from src.gesture_assistant.integrations.browser_actions import (
    DEFAULT_SHORTCUTS as BROWSER_SHORTCUTS,
)
from src.gesture_assistant.integrations.media_actions import DEFAULT_SHORTCUTS as MEDIA_SHORTCUTS
from src.gesture_assistant.integrations.presentation_actions import (
    DEFAULT_SHORTCUTS as PRESENTATION_SHORTCUTS,
)
from src.gesture_assistant.integrations.teams import DEFAULT_SHORTCUTS as TEAMS_SHORTCUTS
from src.gesture_assistant.ui.camera_view import CameraView
from src.gesture_assistant.ui.theme import get_stylesheet
from src.gesture_assistant.vision.camera_worker import CameraWorker
from src.gesture_assistant.vision.hand_provider import MediaPipeHandProvider

ALL_DEFAULT_SHORTCUTS = {
    **TEAMS_SHORTCUTS,
    **BROWSER_SHORTCUTS,
    **PRESENTATION_SHORTCUTS,
    **MEDIA_SHORTCUTS,
}

SIGNALS_CATALOG = (
    ("OPEN_PALM", "Otwarta dłoń (OPEN_PALM)"),
    ("FIST", "Pięść (FIST)"),
    ("THUMBS_UP", "Kciuk w górę (THUMBS_UP)"),
    ("THUMBS_DOWN", "Kciuk w dół (THUMBS_DOWN)"),
    ("VICTORY", "Dwa palce / V (VICTORY)"),
    ("POINTING_UP", "Wskazanie w górę (POINTING_UP)"),
    ("PINCH", "Złączenie palców (PINCH)"),
    ("ROCK", "Gest rock (ROCK)"),
    ("SWIPE_LEFT", "Ruch w lewo (SWIPE_LEFT)"),
    ("SWIPE_RIGHT", "Ruch w prawo (SWIPE_RIGHT)"),
    ("SWIPE_UP", "Ruch w górę (SWIPE_UP)"),
    ("SWIPE_DOWN", "Ruch w dół (SWIPE_DOWN)"),
)

REASON_TRANSLATIONS = {
    "assistant_window_not_target": (
        "Kliknij docelowy program — skrótów nie wysyłamy do okna GestureAssistant"
    ),
    "stale_tracking": "Analiza klatki jest za stara — sprawdź czas inferencji/FPS kamery",
    "invalid_tracking_timestamp": "Niepoprawny znacznik czasu klatki",
    "future_tracking": "Niezgodny zegar klatki i wykonawcy",
    "non_monotonic_tracking": "Powtórzony lub cofnięty czas klatki",
    "gesture_unknown": "Gest niepewny — pokaż wyraźnie dłoń",
    "low_confidence": "Pewność gestu poniżej ustawionego progu",
    "tracking_quality_": "Jakość śledzenia nie pozwala wykonać akcji",
    "paused": "Pauza — akcje wstrzymane przez użytkownika",
    "actions_disabled": "Akcje wyłączone w konfiguracji (safety.enabled=false)",
    "action_not_allowed": "Akcja niedozwolona (brak w allowed_actions)",
    "sleep_disabled": "Uśpienie komputera zablokowane w ustawieniach",
    "unknown_or_stale_context": "Nieznana aplikacja docelowa (kliknij w docelowe okno)",
    "system_blocked_in_teams_browser_or_unknown": (
        "Akcje systemowe zablokowane w Teams / przeglądarce"
    ),
    "wrong_context": "Niewłaściwy profil/kontekst dla tej akcji",
    "wrong_target_application": "Akcja nie pasuje do aktywnego programu",
    "wrong_mode": "Niezgodny tryb profilu",
    "gesture_not_stable_700ms": "Gest w trakcie stabilizacji (przytrzymaj chwilę)",
    "busy": "Wykonywanie innej akcji w toku",
    "action_cooldown": "Cooldown po wykonanej akcji (odczekaj chwilę)",
    "motion_cooldown": "Cooldown po ruchu (odczekaj chwilę)",
    "motion_changed": "Zmieniono ruch przed wykonaniem",
    "gesture_changed": "Zmieniono gest przed wykonaniem",
    "gesture_latched": "Gest już wykonany — zwolnij dłoń przed ponowieniem",
    "expired_request": "Zdarzenie wygasło w kolejce",
    "explicit_confirmation_required": "Wymagane jednorazowe potwierdzenie w GUI",
    "destructive_not_enabled_in_settings": "Akcja destrukcyjna nie jest aktywowana w ustawieniach",
    "authorization_changed": "Autoryzacja wygasła (zmiana aktywnego okna lub utrata dłoni)",
    "dry_run_no_side_effect": "Dry-run — rozpoznano gest, brak wysyłania klawiszy",
    "not_implemented": "Brak implementacji akcji",
    "executed": "Akcja wykonana pomyślnie",
    "no_binding": "Brak przypisania dla tego gestu",
}


def format_reason_pl(reason: str, action_id: str = "") -> str:
    if action_id == "browser.open_url" and reason.startswith("executed"):
        return "Przekazano URL do przeglądarki domyślnej (bez potwierdzenia załadowania)"
    if not reason:
        return "Brak blokady"
    for key, text in REASON_TRANSLATIONS.items():
        if reason == key or reason.startswith(key):
            return text
    if reason.startswith("handler_error:"):
        return f"Błąd wykonania: {reason.split(':', 1)[1].strip()}"
    return reason


def default_model_path() -> Path:
    return Path(__file__).resolve().parents[3] / "models" / "hand_landmarker.task"


class MainWindow(QMainWindow):
    """Główne okno aplikacji GestureAssistant z podglądem, mapowaniem akcji i diagnostyką."""

    def __init__(self, config_path=None) -> None:
        super().__init__()
        self.setWindowTitle("GestureAssistant")
        self.setStyleSheet(get_stylesheet())
        self.resize(1120, 800)
        self._worker = None

        self._config_path = Path(config_path) if config_path is not None else Path("config.json")
        self._config_manager = ConfigManager(self._config_path)
        self._config = defaults()
        self._config_message = ""
        if config_path is not None or self._config_path.exists():
            try:
                loaded = self._config_manager.load()
                self._config = loaded.config
                self._config_message = loaded.message
            except OSError as exc:
                self._config_message = f"Błąd ustawień; użyto domyślnych: {exc}"

        # Always start PAUSED, even if a saved config requests otherwise.
        self._config["safety"]["paused"] = True
        self._runtime = ActionRuntime(self._config)
        self._closing = False
        self._failed = False

        # Kamera i kontrolki podglądu
        self._camera_index = QSpinBox()
        self._camera_index.setRange(0, 32)
        self._camera_index.setValue(self._config["camera"]["index"])
        self._start_button = QPushButton("Start")
        self._stop_button = QPushButton("Stop")
        self._stop_button.setEnabled(False)
        self._pause_button = QPushButton("Akcje: PAUSED")
        self._pause_button.clicked.connect(self._toggle_actions_pause)

        is_dry = self._config["execution"]["dry_run"]
        self._live_toggle_button = QPushButton("Tryb: DRY RUN" if is_dry else "Tryb: LIVE")
        self._live_toggle_button.clicked.connect(self._toggle_execution_mode)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(list(MODES))
        self._mode_combo.setCurrentText(self._config["mode"])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)

        self._camera_view = CameraView()
        self._camera_view.setVisible(self._config["preview"]["enabled"])
        self._status_label = QLabel("Gotowy — wystarczy widoczna dłoń.")
        self._diagnostic_label = QLabel("Gest: — | Ruch: — | Pewność: —")
        self._diagnostic_label.setWordWrap(True)
        self._diagnostic_label.setVisible(self._config["feedback"]["enabled"])
        self._motion_text = "NONE"
        self._motion_until = 0.0
        self._error_label = QLabel(self._config_message)
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #ff6b6b;")
        self._action_label = QLabel("Akcje: PAUSED")
        self._action_label.setWordWrap(True)

        self._context_label = QLabel("Profil: — | Aktywna aplikacja: —")
        self._context_label.setWordWrap(True)
        self._effective_action_label = QLabel("Efektywna akcja: — | Status: —")
        self._effective_action_label.setWordWrap(True)

        self._confirm_button = QPushButton("Potwierdź akcję…")
        self._confirm_button.setEnabled(False)
        self._confirm_button.clicked.connect(self._confirm_action)

        self._tabs = QTabWidget()
        self._build_preview_tab()
        self._build_bindings_tab()
        self._build_settings_tab()

        self.setCentralWidget(self._tabs)

        self._start_button.clicked.connect(self._start_analysis)
        self._stop_button.clicked.connect(self._stop_analysis)

        self._timer = QTimer(self)
        self._timer.setInterval(max(1, round(1000 / self._config["camera"]["fps_limit"])))
        self._timer.timeout.connect(self._refresh_preview)

        self._action_timer = QTimer(self)
        self._action_timer.setInterval(100)
        self._action_timer.timeout.connect(self._refresh_actions)
        self._action_timer.start()

    def _build_preview_tab(self) -> None:
        tab = QWidget()
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Numer kamery:"))
        controls.addWidget(self._camera_index)
        controls.addSpacing(10)
        controls.addWidget(self._start_button)
        controls.addWidget(self._stop_button)
        controls.addSpacing(10)
        controls.addWidget(self._pause_button)
        controls.addWidget(self._live_toggle_button)
        controls.addWidget(QLabel("Tryb:"))
        controls.addWidget(self._mode_combo)
        controls.addStretch()
        controls.addWidget(self._confirm_button)

        diag_group = QGroupBox("Panel diagnostyczny")
        diag_layout = QVBoxLayout()
        diag_layout.addWidget(self._status_label)
        diag_layout.addWidget(self._diagnostic_label)
        diag_layout.addWidget(self._context_label)
        diag_layout.addWidget(self._effective_action_label)
        diag_layout.addWidget(self._action_label)
        diag_layout.addWidget(self._error_label)
        diag_group.setLayout(diag_layout)

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(self._camera_view, 1)
        layout.addWidget(diag_group)
        tab.setLayout(layout)
        self._tabs.addTab(tab, "Podgląd i diagnostyka")

    def _build_bindings_tab(self) -> None:
        self._bindings_tab = QWidget()
        main_layout = QVBoxLayout()

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("<b>Wybierz profil do edycji:</b>"))
        self._profile_selector = QComboBox()
        self._profile_selector.addItems(
            [
                "Globalne (wszystkie aplikacje)",
                "Teams",
                "Browser",
                "Presentation",
                "Media",
                "System",
            ]
        )
        self._profile_selector.currentIndexChanged.connect(self._refresh_bindings_table)
        top_bar.addWidget(self._profile_selector)
        top_bar.addStretch()

        self._save_bindings_btn = QPushButton("Zapisz ustawienia")
        self._save_bindings_btn.clicked.connect(self._save_bindings_from_ui)
        self._reset_defaults_btn = QPushButton("Przywróć domyślne")
        self._reset_defaults_btn.clicked.connect(self._reset_defaults_with_confirm)
        self._test_dry_run_btn = QPushButton("Testuj przypisania w Dry-Run")
        self._test_dry_run_btn.clicked.connect(self._start_dry_run_test)

        top_bar.addWidget(self._test_dry_run_btn)
        top_bar.addWidget(self._reset_defaults_btn)
        top_bar.addWidget(self._save_bindings_btn)
        main_layout.addLayout(top_bar)

        info_lbl = QLabel(
            "<i>Hierarchia przypisań: Przypisania globalne nadpisują reguły profilu.<br>"
            "Ustawienie 'Brak akcji' w profilu lub globalnie definitywnie "
            "blokuje akcję dla danego gestu.</i>"
        )
        info_lbl.setStyleSheet("color: #9aa4b1;")
        main_layout.addWidget(info_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._bindings_container = QWidget()
        self._bindings_form = QVBoxLayout()
        self._bindings_container.setLayout(self._bindings_form)
        scroll.setWidget(self._bindings_container)
        main_layout.addWidget(scroll, 1)

        self._bindings_tab.setLayout(main_layout)
        self._tabs.addTab(self._bindings_tab, "Gesty i akcje")

        self._signal_row_widgets = {}
        self._refresh_bindings_table()

    def _build_settings_tab(self) -> None:
        self._settings_tab = QWidget()
        layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        form_layout = QVBoxLayout()

        # Grupa: Swipe
        swipe_group = QGroupBox("Czułość i detekcja ruchu (Swipe)")
        swipe_form = QFormLayout()
        self._swipe_dist_spin = QDoubleSpinBox()
        self._swipe_dist_spin.setRange(0.05, 0.80)
        self._swipe_dist_spin.setSingleStep(0.05)
        self._swipe_dist_spin.setValue(self._config["thresholds"].get("swipe_distance", 0.25))
        swipe_form.addRow("Minimalny dystans swipe (ułamek kadru):", self._swipe_dist_spin)

        self._swipe_dur_spin = QSpinBox()
        self._swipe_dur_spin.setRange(200, 3000)
        self._swipe_dur_spin.setSingleStep(50)
        self._swipe_dur_spin.setValue(self._config["thresholds"].get("swipe_max_duration_ms", 700))
        swipe_form.addRow("Maksymalny czas trwania swipe (ms):", self._swipe_dur_spin)

        self._motion_cd_spin = QSpinBox()
        self._motion_cd_spin.setRange(1500, 30000)
        self._motion_cd_spin.setSingleStep(100)
        self._motion_cd_spin.setValue(self._config["thresholds"]["motion_cooldown_ms"])
        swipe_form.addRow("Cooldown po ruchu swipe (ms):", self._motion_cd_spin)
        swipe_group.setLayout(swipe_form)
        form_layout.addWidget(swipe_group)

        # Grupa: Gesty statyczne
        static_group = QGroupBox("Stabilizacja gestów statycznych")
        static_form = QFormLayout()
        self._hold_ms_spin = QSpinBox()
        self._hold_ms_spin.setRange(700, 5000)
        self._hold_ms_spin.setSingleStep(50)
        self._hold_ms_spin.setValue(self._config["thresholds"]["gesture_hold_ms"])
        static_form.addRow("Czas wymaganej stabilizacji gestu (ms):", self._hold_ms_spin)

        self._action_cd_spin = QSpinBox()
        self._action_cd_spin.setRange(2000, 30000)
        self._action_cd_spin.setSingleStep(200)
        self._action_cd_spin.setValue(self._config["thresholds"]["action_cooldown_ms"])
        static_form.addRow("Cooldown po wykonaniu akcji (ms):", self._action_cd_spin)

        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.10, 1.00)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(self._config["thresholds"]["min_confidence"])
        static_form.addRow("Minimalny próg pewności (confidence):", self._conf_spin)
        static_group.setLayout(static_form)
        form_layout.addWidget(static_group)

        # Grupa: Bezpieczeństwo
        safety_group = QGroupBox("Polityka bezpieczeństwa i wykonanie")
        safety_vbox = QVBoxLayout()
        self._safety_enabled_check = QCheckBox("Włącz wykonywanie akcji (safety.enabled)")
        self._safety_enabled_check.setChecked(self._config["safety"]["enabled"])
        safety_vbox.addWidget(self._safety_enabled_check)

        self._dry_run_check = QCheckBox("Domyślny bezpieczny tryb Dry-Run (execution.dry_run)")
        self._dry_run_check.setChecked(self._config["execution"]["dry_run"])
        safety_vbox.addWidget(self._dry_run_check)

        self._sleep_check = QCheckBox("Zezwól na uśpienie komputera (safety.sleep_computer)")
        self._sleep_check.setChecked(self._config["safety"]["sleep_computer"])
        safety_vbox.addWidget(self._sleep_check)
        safety_group.setLayout(safety_vbox)
        form_layout.addWidget(safety_group)

        # Grupa: Opis konwencji
        conv_group = QGroupBox("Konwencja kierunków kamery i swipe")
        conv_layout = QVBoxLayout()
        conv_text = QLabel(
            "<b>Konwencja kierunków i podgląd kamery:</b><br>"
            "• <b>SWIPE_RIGHT:</b> ruch w prawo w układzie współrzędnych obrazu "
            "(wartość x rośnie).<br>"
            "• <b>SWIPE_LEFT:</b> ruch w lewo w układzie współrzędnych obrazu "
            "(wartość x maleje).<br>"
            "• <b>SWIPE_UP:</b> ruch w górę w układzie współrzędnych obrazu "
            "(wartość y maleje ku górze).<br>"
            "• <b>SWIPE_DOWN:</b> ruch w dół w układzie współrzędnych obrazu "
            "(wartość y rośnie ku dołowi).<br><br>"
            "<i>Uwaga dot. widoku lustrzanego:</i> Podgląd kamery wyświetla naturalny "
            "strumień kamery bez programowego odwracania. Przy siedzeniu naprzeciw obiektywu, "
            "przesunięcie dłoni w Twoją prawą stronę przesuwa ją w lewo na matrycy (SWIPE_LEFT). "
            "W przypadku włączenia widoku lustrzanego (mirror) kierunki osi poziomej "
            "byłyby wizualnie odwrócone."
        )
        conv_text.setWordWrap(True)
        conv_layout.addWidget(conv_text)
        conv_group.setLayout(conv_layout)
        form_layout.addWidget(conv_group)

        container.setLayout(form_layout)
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)

        btn_bar = QHBoxLayout()
        btn_bar.addStretch()
        self._reset_settings_btn = QPushButton("Przywróć zalecane wartości")
        self._reset_settings_btn.clicked.connect(self._reset_settings_to_recommended)
        self._save_settings_btn = QPushButton("Zapisz parametry")
        self._save_settings_btn.clicked.connect(self._save_settings_from_ui)
        btn_bar.addWidget(self._reset_settings_btn)
        btn_bar.addWidget(self._save_settings_btn)
        layout.addLayout(btn_bar)

        self._settings_tab.setLayout(layout)
        self._tabs.addTab(self._settings_tab, "Czułość i parametry")

    def _current_profile_key(self) -> str:
        idx = self._profile_selector.currentIndex()
        if idx == 0:
            return "bindings"
        names = ["Teams", "Browser", "Presentation", "Media", "System"]
        return names[idx - 1]

    def _refresh_bindings_table(self) -> None:
        # Wyczyść stary formularz
        while self._bindings_form.count():
            item = self._bindings_form.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self._signal_row_widgets.clear()
        profile_key = self._current_profile_key()
        is_global = profile_key == "bindings"

        current_mapping = (
            self._config["bindings"] if is_global else self._config["profiles"].get(profile_key, {})
        )
        allowed_set = set(self._config.get("allowed_actions", []))

        # Przygotuj listę dostępnych akcji z katalogu
        catalog_actions = sorted(ACTION_CATALOG.keys())

        for sig_id, sig_name in SIGNALS_CATALOG:
            box = QGroupBox(sig_name)
            box_layout = QVBoxLayout()
            row1 = QHBoxLayout()

            row1.addWidget(QLabel("Przypisana akcja:"))
            combo = QComboBox()
            if is_global:
                combo.addItem("(Dziedzicz z aktywnego profilu)", "inherit")
            combo.addItem("(Brak akcji — blokuj)", "none")

            selected_action = current_mapping.get(sig_id, "inherit" if is_global else "none")
            combo_idx = max(0, combo.findData(selected_action))
            for _idx, act_id in enumerate(catalog_actions):
                spec = ACTION_CATALOG[act_id]
                # Jeśli profil specyficzny, preferuj akcje tego profilu, ale pozwól wybrać
                if (
                    not is_global
                    and spec.context.value != profile_key
                    and act_id != selected_action
                ):
                    continue
                flags = []
                if spec.destructive:
                    flags.append("wymaga potwierdzenia")
                if not spec.implemented:
                    flags.append("brak impl.")
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                combo.addItem(f"{act_id}{flag_str}", act_id)
                if act_id == selected_action:
                    combo_idx = combo.count() - 1

            combo.setCurrentIndex(combo_idx)
            row1.addWidget(combo, 1)

            allowed_check = QCheckBox("Zezwól (allowed_actions)")
            allowed_check.setChecked(
                selected_action in allowed_set if selected_action != "none" else False
            )
            row1.addWidget(allowed_check)
            box_layout.addLayout(row1)

            # Informacja o efektywnym przypisaniu
            eff_label = QLabel()
            self._update_effective_label(eff_label, sig_id, selected_action, is_global, profile_key)
            box_layout.addWidget(eff_label)

            # Parametry akcji (URL lub skrót)
            param_row = QHBoxLayout()
            url_edit = QLineEdit()
            url_edit.setPlaceholderText("https://...")
            url_edit.setText(
                self._config["targets"]["urls"].get("browser.open_url", "https://example.com")
            )
            url_edit.setVisible(selected_action == "browser.open_url")

            shortcut_edit = QLineEdit()
            shortcut_edit.setPlaceholderText("np. ctrl+shift+m")
            current_sc = self._config["targets"]["shortcuts"].get(
                selected_action, ALL_DEFAULT_SHORTCUTS.get(selected_action, "")
            )
            shortcut_edit.setText(current_sc)
            has_sc = selected_action in ALL_DEFAULT_SHORTCUTS
            shortcut_edit.setVisible(has_sc and selected_action != "browser.open_url")

            param_row.addWidget(url_edit)
            param_row.addWidget(shortcut_edit)
            box_layout.addLayout(param_row)

            # Podłączenie aktualizacji
            def on_action_changed(
                idx,
                s=sig_id,
                c=combo,
                chk=allowed_check,
                ue=url_edit,
                se=shortcut_edit,
                el=eff_label,
            ):
                act = c.currentData()
                chk.setChecked(
                    act in self._config.get("allowed_actions", []) if act != "none" else False
                )
                ue.setVisible(act == "browser.open_url")
                has_shortcut = act in ALL_DEFAULT_SHORTCUTS
                se.setVisible(has_shortcut and act != "browser.open_url")
                if has_shortcut:
                    se.setText(
                        self._config["targets"]["shortcuts"].get(
                            act, ALL_DEFAULT_SHORTCUTS.get(act, "")
                        )
                    )
                self._update_effective_label(el, s, act, is_global, profile_key)

            combo.currentIndexChanged.connect(on_action_changed)

            box.setLayout(box_layout)
            self._bindings_form.addWidget(box)
            self._signal_row_widgets[sig_id] = {
                "combo": combo,
                "allowed_check": allowed_check,
                "url_edit": url_edit,
                "shortcut_edit": shortcut_edit,
                "eff_label": eff_label,
            }

        self._bindings_form.addStretch()

    def _update_effective_label(
        self, label: QLabel, sig: str, current_val: str, is_global: bool, profile: str
    ) -> None:
        if is_global:
            if current_val == "inherit":
                label.setText("<small>Efektywna: dziedzicz z aktywnego profilu</small>")
            elif current_val == "none":
                label.setText(
                    "<small style='color: #8a939c;'>"
                    "Efektywna: Brak akcji (zablokowano globalnie)</small>"
                )
            else:
                label.setText(
                    f"<small style='color: #4c8dff;'>"
                    f"Efektywna: <b>{current_val}</b> (Globalne)</small>"
                )
        else:
            global_val = self._config["bindings"].get(sig)
            if global_val is not None and global_val != "none":
                label.setText(
                    f"<small style='color: #ffaa00;'>"
                    f"Efektywna: <b>{global_val}</b> (Nadpisane przez Globalne!)</small>"
                )
            elif global_val == "none":
                label.setText(
                    "<small style='color: #8a939c;'>"
                    "Efektywna: Brak akcji (Globalnie zablokowano ten gest)</small>"
                )
            elif current_val != "none":
                label.setText(
                    f"<small style='color: #4c8dff;'>"
                    f"Efektywna: <b>{current_val}</b> (Profil: {profile})</small>"
                )
            else:
                label.setText("<small style='color: #8a939c;'>Efektywna: Brak akcji</small>")

    def _save_bindings_from_ui(self) -> None:
        new_config = deepcopy(self._config)
        profile_key = self._current_profile_key()
        is_global = profile_key == "bindings"

        allowed = set(new_config.get("allowed_actions", []))

        for sig_id, row in self._signal_row_widgets.items():
            act = row["combo"].currentData()
            chk = row["allowed_check"].isChecked()
            url = row["url_edit"].text().strip()
            sc = row["shortcut_edit"].text().strip()

            if is_global:
                if act == "inherit":
                    new_config["bindings"].pop(sig_id, None)
                elif act == "none":
                    new_config["bindings"][sig_id] = "none"
                else:
                    new_config["bindings"][sig_id] = act
            else:
                if act == "none":
                    new_config["profiles"][profile_key][sig_id] = "none"
                else:
                    new_config["profiles"][profile_key][sig_id] = act

            if act not in ("none", "inherit"):
                if chk:
                    allowed.add(act)
                if act == "browser.open_url" and url:
                    new_config["targets"]["urls"]["browser.open_url"] = url
                if sc and act in ALL_DEFAULT_SHORTCUTS:
                    new_config["targets"]["shortcuts"][act] = sc

        # Uaktualnij allowed_actions dla wszystkich zaznaczonych
        for _sig_id, row in self._signal_row_widgets.items():
            act = row["combo"].currentData()
            if act not in ("none", "inherit"):
                if row["allowed_check"].isChecked():
                    allowed.add(act)
                else:
                    allowed.discard(act)

        new_config["allowed_actions"] = sorted(list(allowed))
        self._apply_configuration(new_config)

    def _reset_defaults_with_confirm(self) -> None:
        choice = QMessageBox.question(
            self,
            "Potwierdzenie przywrócenia",
            "Czy na pewno chcesz przywrócić domyślne mapowania gestów i akcji?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if choice == QMessageBox.StandardButton.Yes:
            fresh = defaults()
            new_config = deepcopy(self._config)
            profile_key = self._current_profile_key()
            if profile_key == "bindings":
                new_config["bindings"] = fresh["bindings"]
            else:
                new_config["profiles"][profile_key] = fresh["profiles"].get(profile_key, {})
            new_config["allowed_actions"] = fresh["allowed_actions"]
            self._apply_configuration(new_config)

    def _start_dry_run_test(self) -> None:
        self._runtime.set_dry_run(True)
        self._config["execution"]["dry_run"] = True
        self._sync_execution_controls()
        self._pause_button.setText("Akcje: PAUSED")
        self._tabs.setCurrentIndex(0)
        self._error_label.setText(
            "Uruchomiono test przypisania w trybie DRY RUN. "
            "Wznów akcje przyciskiem PAUSED, potem wykonaj gest przed kamerą."
        )

    def _save_settings_from_ui(self) -> None:
        new_config = deepcopy(self._config)
        new_config["thresholds"]["swipe_distance"] = round(self._swipe_dist_spin.value(), 2)
        new_config["thresholds"]["swipe_max_duration_ms"] = int(self._swipe_dur_spin.value())
        new_config["thresholds"]["motion_cooldown_ms"] = int(self._motion_cd_spin.value())
        new_config["thresholds"]["gesture_hold_ms"] = int(self._hold_ms_spin.value())
        new_config["thresholds"]["action_cooldown_ms"] = int(self._action_cd_spin.value())
        new_config["thresholds"]["min_confidence"] = round(self._conf_spin.value(), 2)

        new_config["safety"]["enabled"] = self._safety_enabled_check.isChecked()
        new_config["execution"]["dry_run"] = self._dry_run_check.isChecked()
        new_config["safety"]["sleep_computer"] = self._sleep_check.isChecked()

        self._apply_configuration(new_config)

    def _reset_settings_to_recommended(self) -> None:
        self._swipe_dist_spin.setValue(0.25)
        self._swipe_dur_spin.setValue(700)
        self._motion_cd_spin.setValue(1500)
        self._hold_ms_spin.setValue(700)
        self._action_cd_spin.setValue(2000)
        self._conf_spin.setValue(0.70)
        self._error_label.setText(
            "Przywrócono zalecane parametry czułości (kliknij 'Zapisz parametry' aby utrwalić)."
        )

    def _on_mode_changed(self, mode: str) -> None:
        config = deepcopy(self._config)
        config["mode"] = mode
        self._apply_configuration(config)

    def _apply_configuration(self, new_config: dict) -> bool:
        try:
            validated = validate_config(new_config)
            self._config_manager.save(validated)
            self._config = validated
            self._runtime.apply_config(validated)
            if self._worker is not None:
                self._worker.apply_config(validated)
            self._pause_button.setText("Akcje: PAUSED")
            is_dry = self._config["execution"]["dry_run"]
            self._live_toggle_button.setText("Tryb: DRY RUN" if is_dry else "Tryb: LIVE")
            self._sync_execution_controls()
            self._error_label.setText(
                "Zapisano konfigurację. Akcje są PAUSED — świadomie wznów je przyciskiem."
            )
            self._refresh_bindings_table()
            return True
        except Exception as exc:
            self._error_label.setText(f"Błąd zapisu konfiguracji: {exc}")
            QMessageBox.critical(
                self, "Błąd konfiguracji", f"Niepoprawne parametry konfiguracji:\n{exc}"
            )
            return False

    def _sync_execution_controls(self) -> None:
        dry_run = self._config["execution"]["dry_run"]
        self._live_toggle_button.setText("Tryb: DRY RUN" if dry_run else "Tryb: LIVE")
        self._dry_run_check.setChecked(dry_run)
        self._safety_enabled_check.setChecked(self._config["safety"]["enabled"])
        self._sleep_check.setChecked(self._config["safety"]["sleep_computer"])

    def _toggle_execution_mode(self) -> None:
        currently_dry_run = self._config["execution"]["dry_run"]
        if currently_dry_run:
            choice = QMessageBox.warning(
                self,
                "Włączenie rzeczywistych akcji (LIVE)",
                "Czy na pewno chcesz włączyć tryb LIVE?\n\n"
                "W trybie LIVE gesty będą wysyłać prawdziwe wejścia (skróty klawiszowe, polecenia) "
                "do aktywnej aplikacji docelowej.\n\n"
                "Dla bezpieczeństwa po włączeniu możesz w każdej chwili zatrzymać działanie "
                "przyciskiem PAUSED lub Stop.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if choice == QMessageBox.StandardButton.Yes:
                self._runtime.set_dry_run(False)
                self._config["execution"]["dry_run"] = False
                self._sync_execution_controls()
                self._pause_button.setText("Akcje: PAUSED")
                self._error_label.setText("Włączono tryb LIVE. Pamiętaj o wznowieniu po PAUSED.")
        else:
            self._runtime.set_dry_run(True)
            self._config["execution"]["dry_run"] = True
            self._sync_execution_controls()
            self._pause_button.setText("Akcje: PAUSED")
            self._error_label.setText("Przełączono w bezpieczny tryb DRY RUN.")
        self._refresh_actions()

    def _start_analysis(self) -> None:
        if self._worker is not None or self._runtime.is_running:
            return
        try:
            provider = MediaPipeHandProvider(default_model_path())
        except (OSError, ValueError) as exc:
            self._error_label.setText(str(exc))
            self._status_label.setText("Nie można uruchomić detekcji dłoni.")
            return
        self._failed = False
        self._motion_text = "NONE"
        self._motion_until = 0.0
        self._diagnostic_label.setText("Oczekiwanie na analizę…")
        self._error_label.setText(self._config_message)
        self._camera_view.clear()
        self._runtime.pause()
        self._pause_button.setText("Akcje: PAUSED")
        worker = CameraWorker(
            self._camera_index.value(),
            provider,
            parent=self,
            runtime=self._runtime,
            camera_settings=self._config["camera"],
        )
        worker.apply_config(self._config)
        worker.error.connect(self._on_error)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        self._camera_index.setEnabled(False)
        self._start_button.setEnabled(False)
        self._stop_button.setEnabled(True)
        self._status_label.setText("Uruchamianie kamery i modelu dłoni…")
        self._timer.start()
        try:
            self._runtime.start()
            worker.start()
        except RuntimeError as exc:
            self._runtime.stop()
            provider.close()
            self._worker = None
            worker.deleteLater()
            self._timer.stop()
            self._start_button.setEnabled(True)
            self._stop_button.setEnabled(False)
            self._camera_index.setEnabled(True)
            self._error_label.setText(str(exc))

    def _toggle_actions_pause(self) -> None:
        if self._runtime.safety.paused:
            if self._worker is None or not self._runtime.is_running:
                self._error_label.setText("Najpierw uruchom kamerę.")
                return
            if not self._config["safety"]["enabled"]:
                choice = QMessageBox.question(
                    self,
                    "Akcje wyłączone",
                    "Wykonywanie akcji jest wyłączone w konfiguracji (safety.enabled=false).\n"
                    "Czy chcesz włączyć obsługę akcji teraz?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if choice == QMessageBox.StandardButton.Yes:
                    config = deepcopy(self._config)
                    config["safety"]["enabled"] = True
                    if not self._apply_configuration(config):
                        return
                else:
                    self._error_label.setText(
                        "Akcje wyłączone: ustaw safety.enabled w konfiguracji."
                    )
                    return
            live = not self._config["execution"]["dry_run"]
            if live:
                choice = QMessageBox.warning(
                    self,
                    "Włączenie rzeczywistych akcji",
                    "Gesty mogą wysyłać skróty i wykonywać dozwolone akcje Windows. "
                    "Czy włączyć wykonanie? PAUSED lub Stop je wstrzyma.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if choice != QMessageBox.StandardButton.Yes:
                    return
            self._runtime.resume()
            self._pause_button.setText("Akcje: LIVE" if live else "Akcje: DRY RUN")
        else:
            self._runtime.pause()
            self._pause_button.setText("Akcje: PAUSED")

    def _refresh_actions(self) -> None:
        pending = self._runtime.pending_confirmation()
        self._confirm_button.setEnabled(pending is not None and not self._runtime.safety.paused)
        outcome = self._runtime.status()
        state = (
            "PAUSED"
            if self._runtime.safety.paused
            else ("DRY RUN" if self._config["execution"]["dry_run"] else "LIVE")
        )
        detail = (
            ""
            if outcome is None
            else f" | {outcome.action_id}: {format_reason_pl(outcome.reason, outcome.action_id)}"
        )
        self._action_label.setText(f"Akcje: {state}{detail}")
        if outcome is not None:
            self._confirm_button.setToolTip(f"{outcome.action_id}: {outcome.reason}")

        # Aktualizacja rozszerzonego panelu diagnostycznego
        diag = self._runtime.diagnostics()
        if diag is not None:
            fg_name = (
                Path(diag["foreground_exe"]).name
                if diag["foreground_exe"]
                else "Brak / własne okno"
            )
            self._context_label.setText(
                f"Profil: <b>{diag['active_profile']}</b> (Tryb: {diag['mode']}) | "
                f"Aktywna aplikacja: <b>{fg_name}</b>"
            )
            act = diag["effective_action"]
            act_text = act if act is not None else "(brak przypisania)"
            source = diag["action_source"]
            reason_pl = format_reason_pl(diag["reason"], act or "")
            if not diag["reason"]:
                if not diag.get("consumer_running", True):
                    reason_pl = "Konsument akcji nie działa — zatrzymaj i uruchom kamerę"
                else:
                    reason_pl = "Gotowe do wykonania / oczekiwanie na wynik kolejki"
            progress = f" | Hold: {diag.get('stable_ms', 0)}/{diag.get('required_hold_ms', 700)} ms"
            self._effective_action_label.setText(
                f"Gest: <b>{diag['signal']}</b> ➔ Akcja: <b>{act_text}</b> [{source}] | "
                f"Status: <span style='color: #4c8dff;'>{reason_pl}</span>{progress}"
            )

        if self._closing and self._worker is None and not self._runtime.is_running:
            self.close()

    def _confirm_action(self) -> None:
        pending = self._runtime.pending_confirmation()
        if pending is None:
            return
        answer = QMessageBox.warning(
            self,
            "Potwierdzenie jednorazowe",
            f"Zezwolić na {pending[0]}? "
            "Po zatwierdzeniu wróć do poprzedniego okna i ponownie pokaż gest. "
            "Potwierdzenie wygaśnie po 5 sekundach. Nie wykonuje starego żądania.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            if self._runtime.confirm_pending():
                self._error_label.setText("Potwierdzono: wróć do celu i pokaż gest przez 700 ms.")
            else:
                self._error_label.setText("Potwierdzenie wygasło lub akcje są wstrzymane.")

    def _stop_analysis(self) -> None:
        self._runtime.stop()
        self._pause_button.setText("Akcje: PAUSED")
        if self._worker is not None:
            self._stop_button.setEnabled(False)
            self._status_label.setText("Zatrzymywanie…")
            self._worker.stop()

    def _refresh_preview(self) -> None:
        if self._worker is None:
            return
        packet = self._worker.take_latest_preview()
        if packet is None:
            return
        frame, result = packet
        drawable_hands = tuple(
            hand
            for hand in result.hands
            if len(hand.landmarks) == 21
            and all(
                math.isfinite(value)
                for point in hand.landmarks
                for value in (point.x, point.y, point.z)
            )
        )
        if self._config["preview"]["enabled"]:
            self._camera_view.set_frame(
                frame,
                drawable_hands if self._config["preview"]["landmarks"] else (),
            )
        analysis = self._worker.take_latest_analysis()
        if analysis is not None:
            if analysis.motion_event is not None:
                self._motion_text = analysis.motion.value
                self._motion_until = time.monotonic() + 1.5
            if time.monotonic() >= self._motion_until:
                self._motion_text = "NONE"
            self._diagnostic_label.setText(
                f"Gest: {analysis.gesture.value} | Ruch: {self._motion_text}\n"
                f"Pewność heurystyczna: {analysis.confidence:.2f} | "
                f"Jakość: {analysis.quality.value} | "
                f"Cooldown: {analysis.cooldown_ms / 1000:.1f} s\n"
                f"Feedback: {analysis.feedback}"
            )
        if self._failed:
            return
        if not result.hands:
            self._status_label.setText("Brak dłoni — pokaż dłoń do kamery.")
        elif any(
            p.x < 0 or p.x > 1 or p.y < 0 or p.y > 1
            for hand in result.hands
            for p in hand.landmarks
        ):
            self._status_label.setText("Dłoń częściowo poza kadrem.")
        else:
            self._status_label.setText(f"Śledzenie dłoni: {len(result.hands)}")

    def _on_error(self, message: str) -> None:
        self._runtime.stop()
        self._pause_button.setText("Akcje: PAUSED")
        self._failed = True
        self._status_label.setText("Błąd kamery lub detekcji dłoni.")
        self._error_label.setText(message)
        if self._worker is not None:
            self._worker.stop()

    def _on_worker_finished(self) -> None:
        self._runtime.stop()
        self._pause_button.setText("Akcje: PAUSED")
        self._timer.stop()
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self._start_button.setEnabled(True)
        self._stop_button.setEnabled(False)
        self._camera_index.setEnabled(True)
        self._camera_view.clear()
        if not self._failed:
            self._status_label.setText("Kamera zatrzymana.")
        if self._closing:
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event) -> None:
        if self._worker is not None or self._runtime.is_running:
            self._closing = True
            self._stop_analysis()
            event.ignore()
            return
        self._action_timer.stop()
        super().closeEvent(event)
