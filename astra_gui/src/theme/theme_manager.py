"""
ASTRA Theme Manager
Loads scientific dark theme configuration and provides Qt styling utilities.
"""

import os
import yaml
from typing import Dict, Any, Optional
from PySide6.QtGui import QColor, QFont


class ThemeManager:
    """Singleton/Provider for ASTRA UI theme tokens, palettes, and stylesheets."""

    _instance: Optional["ThemeManager"] = None

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "..", "..", "configs", "theme.yaml"
        )
        self.theme_data = self._load_theme()
        self.palette = self.theme_data.get("palette", {})
        self.fonts = self.theme_data.get("fonts", {})

    @classmethod
    def get_instance(cls) -> "ThemeManager":
        if cls._instance is None:
            cls._instance = ThemeManager()
        return cls._instance

    def _load_theme(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {
            "palette": {
                "background_deep": "#0a0c10",
                "background_panel": "#12161f",
                "background_surface": "#181e2b",
                "border_subtle": "#263043",
                "primary_cyan": "#00d2ff",
                "secondary_violet": "#9d4edd",
                "confirmed_green": "#00f59b",
                "estimated_cyan": "#00d2ff",
                "possible_amber": "#ffbe0b",
                "unknown_gray": "#6c757d",
                "error_red": "#ff0055",
                "text_primary": "#f0f4fc",
                "text_secondary": "#9bb0cf",
                "text_muted": "#5d6d88"
            }
        }

    def get_color(self, key: str, default: str = "#ffffff") -> QColor:
        hex_val = self.palette.get(key, default)
        return QColor(hex_val)

    def get_color_hex(self, key: str, default: str = "#ffffff") -> str:
        return self.palette.get(key, default)

    def get_status_color(self, status_str: str) -> str:
        s = str(status_str).upper()
        if "CONFIRMED" in s:
            return self.get_color_hex("confirmed_green")
        elif "ESTIMATED" in s:
            return self.get_color_hex("estimated_cyan")
        elif "POSSIBLE" in s:
            return self.get_color_hex("possible_amber")
        elif "ERROR" in s or "FAIL" in s:
            return self.get_color_hex("error_red")
        return self.get_color_hex("unknown_gray")

    def get_region_color(self, region_name: str) -> str:
        r = str(region_name).upper()
        if "SYNC" in r or "PREAMBLE" in r:
            return self.get_color_hex("region_sync", "#00b4d8")
        elif "HEADER" in r:
            return self.get_color_hex("region_header", "#7209b7")
        elif "PAYLOAD" in r:
            return self.get_color_hex("region_payload", "#00f59b")
        elif "CRC" in r:
            return self.get_color_hex("region_crc", "#f72585")
        elif "PADDING" in r:
            return self.get_color_hex("region_padding", "#4a4e69")
        return self.get_color_hex("region_unknown", "#6c757d")

    def build_stylesheet(self) -> str:
        """Generates comprehensive dark scientific Qt StyleSheet."""
        p = self.palette
        return f"""
        QMainWindow, QDialog, QWidget#CentralContainer {{
            background-color: {p.get('background_deep')};
            color: {p.get('text_primary')};
            font-family: 'Segoe UI', 'Inter', sans-serif;
            font-size: 12px;
        }}

        QFrame, QGroupBox {{
            border: 1px solid {p.get('border_subtle')};
            border-radius: 6px;
            background-color: {p.get('background_panel')};
        }}

        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 2px 8px;
            color: {p.get('primary_cyan')};
            font-weight: bold;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        /* Buttons */
        QPushButton {{
            background-color: {p.get('background_surface')};
            border: 1px solid {p.get('border_subtle')};
            border-radius: 4px;
            padding: 6px 14px;
            color: {p.get('text_primary')};
            font-weight: 600;
        }}

        QPushButton:hover {{
            background-color: {p.get('background_elevated')};
            border: 1px solid {p.get('primary_cyan')};
            color: {p.get('primary_glow')};
        }}

        QPushButton:pressed {{
            background-color: {p.get('background_deep')};
        }}

        QPushButton#PrimaryAction {{
            background-color: #004b66;
            border: 1px solid {p.get('primary_cyan')};
            color: #ffffff;
        }}

        QPushButton#PrimaryAction:hover {{
            background-color: #007799;
        }}

        /* Scrollbars */
        QScrollBar:vertical {{
            border: none;
            background: {p.get('background_deep')};
            width: 8px;
            margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background: {p.get('border_subtle')};
            min-height: 20px;
            border-radius: 4px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {p.get('primary_cyan')};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}

        /* Tabs */
        QTabWidget::pane {{
            border: 1px solid {p.get('border_subtle')};
            background-color: {p.get('background_panel')};
            border-radius: 4px;
        }}

        QTabBar::tab {{
            background: {p.get('background_deep')};
            color: {p.get('text_secondary')};
            padding: 6px 14px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            margin-right: 2px;
            font-weight: 600;
        }}

        QTabBar::tab:selected {{
            background: {p.get('background_panel')};
            color: {p.get('primary_cyan')};
            border-bottom: 2px solid {p.get('primary_cyan')};
        }}

        QTabBar::tab:hover:!selected {{
            color: #ffffff;
            background: {p.get('background_surface')};
        }}

        /* Tooltips */
        QToolTip {{
            background-color: {p.get('background_elevated')};
            color: {p.get('text_primary')};
            border: 1px solid {p.get('primary_cyan')};
            border-radius: 4px;
            padding: 6px;
            font-size: 11px;
        }}

        /* Headers and Tables */
        QHeaderView::section {{
            background-color: {p.get('background_surface')};
            color: {p.get('text_secondary')};
            padding: 4px 8px;
            border: 1px solid {p.get('border_subtle')};
            font-weight: bold;
            font-size: 11px;
        }}

        QTableView, QTreeView, QListView {{
            background-color: {p.get('background_panel')};
            gridline-color: {p.get('border_subtle')};
            border: 1px solid {p.get('border_subtle')};
            selection-background-color: {p.get('background_elevated')};
            selection-color: {p.get('primary_cyan')};
            outline: none;
        }}
        """
