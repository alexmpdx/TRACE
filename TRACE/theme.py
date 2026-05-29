"""Theme module — semantic color tokens + dark / light theme instances.

Every hardcoded `#hex` style decision in the TRACE GUI eventually resolves
through ``current_theme()`` so the same widget code paints correctly in
both dark and light mode. The user picks ``Dark`` / ``Light`` / ``System``
from the Settings tab; the preference persists in QSettings under the
``theme/preference`` key. ``System`` mode follows the OS color scheme on
Qt 6 (via Qt.ColorScheme) and falls back to a window-palette brightness
heuristic on Qt 5.

Live switching: ``ThemeManager.themeChanged`` is a pyqtSignal carrying the
new ``Theme`` instance. Long-lived widgets that use inline stylesheets
(QPushButton CSS, QLabel rich text) connect to it and re-apply their
styles in the slot. Short-lived widgets (modal dialogs, walkthrough
overlay popups) read ``current_theme()`` at construction; by the time
they're shown again the theme has already been applied at startup.

Adding a new color decision: don't hardcode a hex. Add a semantic token
(e.g. ``my_section_separator``) to ``Theme`` with values for both themes,
then reference ``current_theme().my_section_separator``. The audit pass
that landed this module assumed every hex was theme-significant; if you
genuinely need a fixed color (e.g. a logo brand color), use a module-level
constant + a comment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from PyQt5.QtCore import QObject, QSettings, pyqtSignal


class ThemePreference(str, Enum):
    """User-facing preference (what the Settings picker stores)."""

    SYSTEM = "system"
    LIGHT = "light"
    DARK = "dark"


@dataclass(frozen=True)
class Theme:
    """Semantic color tokens — one instance per theme.

    Token groups:
    - **Surfaces**: layered backgrounds (window → panel → button → pressed)
    - **Text**: primary, body, muted, placeholder, disabled
    - **Borders**: main + subtle
    - **Accent**: primary interactive (blue) + link
    - **Status**: success/error/warning/cancel/skip — adjusted for contrast
      on each theme so the bright primary form reads on the appropriate
      background.
    - **Walkthrough**: highlight accent + dim layer color
    - **Dialog**: explicit bg/border for the custom-styled modal dialogs
      (Update available, JPEG warning, etc.) that bypass the system
      palette via setStyleSheet.
    - **Pipeline map**: stage stroke + fill pairs (kept identical across
      themes — the pastel fills work on both backgrounds).
    """

    name: str  # "dark" | "light" — used by callers that need to branch

    # --- Surfaces ---
    bg: str  # window background
    surface: str  # raised panel / group box / button base
    surface_alt: str  # button hovered / row hovered
    surface_pressed: str  # button pressed

    # --- Text ---
    text: str  # primary text
    text_body: str  # body text (slightly less prominent than headings)
    text_muted: str  # secondary / dimmed text
    text_placeholder: str  # placeholder in line edits
    text_disabled: str  # disabled labels / button text

    # --- Borders ---
    border: str  # main border
    border_subtle: str  # softer border

    # --- Accent ---
    accent: str  # primary blue — focus, IN_PROGRESS row, dialog border
    accent_hover: str
    link: str  # hyperlink color

    # --- Status ---
    success: str  # SUCCEEDED row + green progress fill
    success_text: str  # "✓ up to date" label
    error: str  # FAILED row + cancel highlight
    error_text: str  # error label color
    warning: str  # pause progress fill
    warning_text: str  # "update available" label
    cancel_highlight: str  # cancel state progress fill
    skip_gray: str  # resume SKIPPED row
    user_skip: str  # USER_SKIPPED row (warm gray)

    # --- Walkthrough / dialog overlays ---
    walkthrough_accent: str  # orange highlight ring + popup border
    dialog_bg: str  # custom modal dialog background
    dialog_border: str  # custom modal dialog border (usually = accent)
    dim_overlay_rgba: tuple  # walkthrough dim layer — (r, g, b, a) for QColor(*tup)

    # --- Pipeline map (stage stroke + pastel fill pairs) ---
    pipe_ifeat_stroke: str
    pipe_ifeat_fill: str
    pipe_result_stroke: str
    pipe_result_fill: str
    pipe_seg_stroke: str
    pipe_seg_fill: str
    pipe_analysis_stroke: str
    pipe_analysis_fill: str
    pipe_isolation_stroke: str
    pipe_isolation_fill: str
    pipe_rotation_stroke: str
    pipe_rotation_fill: str


DARK_THEME = Theme(
    name="dark",
    bg="#2d2d2d",
    surface="#373737",
    surface_alt="#454545",
    surface_pressed="#2f2f2f",
    text="#d0d0d0",
    text_body="#c0c0c0",
    text_muted="#aaaaaa",
    text_placeholder="#888888",
    text_disabled="#7a7a7a",
    border="#555555",
    border_subtle="#5a5a5a",
    accent="#0d6efd",
    accent_hover="#3a8dff",
    link="#4aa3ff",
    success="#5cb85c",
    success_text="#6cc66c",
    error="#ff3333",
    error_text="#ff8888",
    warning="#f0ad4e",
    warning_text="#ffb05a",
    cancel_highlight="#d9534f",
    skip_gray="#808080",
    user_skip="#a08070",
    walkthrough_accent="#ff9d4a",
    dialog_bg="#2d2d2d",
    dialog_border="#0d6efd",
    dim_overlay_rgba=(0, 0, 0, 160),
    # Pipeline map: pastel fills + saturated strokes. These already
    # read fine on both dark and light window backgrounds because the
    # nodes are filled (not transparent), so they keep their own
    # internal contrast either way.
    pipe_ifeat_stroke="#8c3c8c",
    pipe_ifeat_fill="#f4e6f4",
    pipe_result_stroke="#b33c3c",
    pipe_result_fill="#fde7e7",
    pipe_seg_stroke="#c8862a",
    pipe_seg_fill="#fde6c4",
    pipe_analysis_stroke="#3c8c4f",
    pipe_analysis_fill="#e6f4ea",
    pipe_isolation_stroke="#b38a1b",
    pipe_isolation_fill="#fff7e0",
    pipe_rotation_stroke="#3078a0",
    pipe_rotation_fill="#e0f0f7",
)


LIGHT_THEME = Theme(
    name="light",
    bg="#f5f5f5",
    surface="#ffffff",
    surface_alt="#ebebeb",
    surface_pressed="#dcdcdc",
    text="#1a1a1a",
    text_body="#2a2a2a",
    text_muted="#666666",
    text_placeholder="#888888",
    text_disabled="#aaaaaa",
    border="#c8c8c8",
    border_subtle="#dcdcdc",
    accent="#0d6efd",  # same blue reads well on both
    accent_hover="#3a8dff",
    link="#0a58ca",  # darker blue for contrast on white
    success="#2d8a2d",
    success_text="#2d8a2d",
    error="#c92a2a",
    error_text="#c92a2a",
    warning="#d97706",
    warning_text="#c25e08",
    cancel_highlight="#c92a2a",
    skip_gray="#999999",
    user_skip="#a08070",
    walkthrough_accent="#e8852a",  # orange that still pops on white
    dialog_bg="#ffffff",
    dialog_border="#0d6efd",
    dim_overlay_rgba=(40, 40, 40, 120),
    # Same pipeline-map colors — pastel fills read on both backgrounds.
    pipe_ifeat_stroke="#8c3c8c",
    pipe_ifeat_fill="#f4e6f4",
    pipe_result_stroke="#b33c3c",
    pipe_result_fill="#fde7e7",
    pipe_seg_stroke="#c8862a",
    pipe_seg_fill="#fde6c4",
    pipe_analysis_stroke="#3c8c4f",
    pipe_analysis_fill="#e6f4ea",
    pipe_isolation_stroke="#b38a1b",
    pipe_isolation_fill="#fff7e0",
    pipe_rotation_stroke="#3078a0",
    pipe_rotation_fill="#e0f0f7",
)


_THEMES_BY_NAME = {"dark": DARK_THEME, "light": LIGHT_THEME}


class ThemeManager(QObject):
    """Singleton holding the active theme + emitting changes.

    Access via the module-level ``manager()`` accessor (created lazily so
    a QApplication isn't required at import time — important because
    ``theme`` is imported at module-load time and QApplication doesn't
    exist that early).

    Usage:
        mgr = manager()
        mgr.themeChanged.connect(self._apply_theme_styles)
        mgr.set_preference(ThemePreference.LIGHT)
    """

    themeChanged = pyqtSignal(object)  # emits Theme

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._settings = QSettings("TRACE", "WingAnalysisPipeline")
        raw = str(self._settings.value("theme/preference", ThemePreference.SYSTEM.value) or "")
        try:
            self._preference = ThemePreference(raw)
        except ValueError:
            self._preference = ThemePreference.SYSTEM
        self._active: Theme = self._resolve(self._preference)

    @property
    def preference(self) -> ThemePreference:
        """Saved preference: SYSTEM / LIGHT / DARK."""
        return self._preference

    @property
    def active(self) -> Theme:
        """The currently-applied Theme (after SYSTEM resolution)."""
        return self._active

    def set_preference(self, pref: ThemePreference) -> None:
        """Persist a new preference + apply the resolved theme.

        No-op if the resolved active theme isn't changing (e.g. user
        toggles SYSTEM → DARK while the OS was already dark) so connected
        widgets don't get spurious re-style fires.
        """
        if pref is self._preference:
            return
        self._preference = pref
        self._settings.setValue("theme/preference", pref.value)
        new_theme = self._resolve(pref)
        if new_theme is not self._active:
            self._active = new_theme
            self.themeChanged.emit(new_theme)

    def refresh_system(self) -> None:
        """Re-resolve when the OS color scheme might have changed.

        Called on macOS/Linux when we detect a system appearance change
        (Qt fires QApplication.paletteChanged). Only meaningful in
        SYSTEM mode — otherwise the user's explicit pick wins.
        """
        if self._preference is not ThemePreference.SYSTEM:
            return
        new_theme = self._resolve(ThemePreference.SYSTEM)
        if new_theme is not self._active:
            self._active = new_theme
            self.themeChanged.emit(new_theme)

    @staticmethod
    def _resolve(pref: ThemePreference) -> Theme:
        if pref is ThemePreference.LIGHT:
            return LIGHT_THEME
        if pref is ThemePreference.DARK:
            return DARK_THEME
        return _detect_system_theme()


_MANAGER: Optional[ThemeManager] = None


def manager() -> ThemeManager:
    """Module-level accessor for the ThemeManager singleton."""
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = ThemeManager()
    return _MANAGER


def current_theme() -> Theme:
    """Shortcut for ``manager().active`` — the most common call site."""
    return manager().active


def _detect_system_theme() -> Theme:
    """Best-effort system color-scheme detection.

    Qt 6.5+ exposes ``QStyleHints.colorScheme()``; we probe it first.
    Otherwise fall back to inspecting the default QPalette's window
    color: if its luminance is below 0.5 we treat the system as dark.

    Worst case (no QApplication yet, no style hints, palette uninit):
    return DARK_THEME — TRACE has always shipped dark, and that's the
    documented default. The user can switch from Settings either way.
    """
    try:
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QGuiApplication

        gapp = QGuiApplication.instance()
        if gapp is not None:
            hints = gapp.styleHints()
            color_scheme = getattr(hints, "colorScheme", None)
            if callable(color_scheme):
                scheme = color_scheme()
                # Qt.ColorScheme.Light = 1, Dark = 2, Unknown = 0
                if int(scheme) == 1:
                    return LIGHT_THEME
                if int(scheme) == 2:
                    return DARK_THEME
            # Fall through to palette heuristic when scheme is Unknown.
            palette = gapp.palette()
            bg = palette.color(palette.Window)
            r, g, b, _ = bg.getRgb()
            # Rec. 709 relative luminance, gamma-uncorrected.
            luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
            return LIGHT_THEME if luminance >= 0.5 else DARK_THEME
    except Exception:
        # Any import / probe failure — bail to dark since that's what
        # TRACE shipped before the theme module existed.
        pass
    return DARK_THEME
