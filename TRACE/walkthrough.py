"""First-launch walkthrough overlay for the TRACE main window.

A `WalkthroughOverlay` is a child of the main window's central widget that
dims everything except one highlighted target widget at a time and shows a
short instruction popup next to it. Users advance with a Next button and
finish (or skip) at any point — both write `walkthrough_completed = True`
to QSettings so it doesn't auto-show on the next launch.

Auto-show:
    On the first launch (no `walkthrough_completed` flag in QSettings), the
    main window calls `WalkthroughOverlay(...).start()` via a deferred
    `QTimer.singleShot(0, ...)` so widget geometries are valid before we
    try to map positions into the overlay.

Re-trigger:
    The main window's Help menu exposes "Show Walkthrough" which builds a
    fresh overlay and runs it again regardless of the persisted flag.

Highlight technique:
    The overlay covers the full central widget. `paintEvent` fills it with
    a semi-transparent dark color and draws a 2px accent border around the
    target widget's rect. `setMask` subtracts the target rect from the
    overlay's mouse region so clicks pass through to the highlighted
    control — the user can actually interact with it.

Position tracking:
    The main window forwards `resizeEvent` and `splitterMoved` to
    `reposition()`, which re-reads the target widget's geometry (mapped
    into the central widget's coordinate space) and updates both the
    overlay mask and the popup placement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

from PyQt5.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from PyQt5.QtCore import QSettings


# ---------------------------------------------------------------------------
# Step definition
# ---------------------------------------------------------------------------


@dataclass
class WalkthroughStep:
    """One step of the walkthrough.

    target_resolver:
        Callable that returns the QWidget to highlight. Called fresh on each
        step transition (in case the widget is recreated, e.g. after a tab
        rebuild). Receives the main window as its only argument.
    title, body:
        Strings shown on the instruction popup. Markdown not interpreted —
        plain text only.
    pre_action:
        Optional callable invoked before the highlight repositions, e.g.
        `lambda w: w.right_tabs.setCurrentIndex(1)` to switch tabs first.
        Receives the main window as its only argument.
    """

    target_resolver: Callable[[QWidget], QWidget]
    title: str
    body: str
    pre_action: Optional[Callable[[QWidget], None]] = field(default=None)


# ---------------------------------------------------------------------------
# Instruction popup
# ---------------------------------------------------------------------------


class _WalkthroughPopup(QFrame):
    """Frameless instruction popup with title, body, Skip, Next/Finish buttons."""

    next_clicked = pyqtSignal()
    prev_clicked = pyqtSignal()
    skip_clicked = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        # QFrame::Box gives us a thin border that we then re-style with a
        # stylesheet for the accent color. setObjectName is required so the
        # stylesheet selector matches the QFrame instance rather than every
        # QFrame in the parent's tree.
        self.setObjectName("WalkthroughPopup")
        self.setStyleSheet(
            "#WalkthroughPopup { "
            "background-color: #2d2d2d; "
            "border: 2px solid #4aa3ff; "
            "border-radius: 6px; "
            "}"
        )
        self.setMinimumWidth(280)
        self.setMaximumWidth(360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        self._title = QLabel("")
        title_font = QFont(self._title.font())
        title_font.setBold(True)
        title_font.setPointSize(title_font.pointSize() + 1)
        self._title.setFont(title_font)
        self._title.setStyleSheet("color: #d0d0d0;")
        layout.addWidget(self._title)

        self._body = QLabel("")
        self._body.setWordWrap(True)
        self._body.setStyleSheet("color: #c0c0c0;")
        layout.addWidget(self._body)

        # "Don't show again" — explicit opt-out from the auto-show on every
        # launch. Without this checkbox, Skip / Finish / Esc all dismiss the
        # walkthrough for the current session only.
        self._dont_show_chk = QCheckBox("Don't show this again on launch")
        self._dont_show_chk.setStyleSheet("color: #aaa;")
        layout.addWidget(self._dont_show_chk)

        # Footer row: step counter on the left, Skip + Next on the right.
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        self._counter = QLabel("")
        self._counter.setStyleSheet("color: #888;")
        footer.addWidget(self._counter)
        footer.addStretch(1)
        self._skip_btn = QPushButton("Skip tutorial")
        self._skip_btn.setFlat(True)
        self._skip_btn.setStyleSheet("color: #888;")
        self._skip_btn.clicked.connect(self.skip_clicked)
        footer.addWidget(self._skip_btn)
        self._prev_btn = QPushButton("Previous")
        self._prev_btn.clicked.connect(self.prev_clicked)
        footer.addWidget(self._prev_btn)
        self._next_btn = QPushButton("Next")
        self._next_btn.setDefault(True)
        self._next_btn.clicked.connect(self.next_clicked)
        footer.addWidget(self._next_btn)
        layout.addLayout(footer)

    def set_content(self, title: str, body: str, step_idx: int, step_total: int, is_last: bool) -> None:
        self._title.setText(title)
        self._body.setText(body)
        self._counter.setText(f"{step_idx + 1} / {step_total}")
        self._next_btn.setText("Finish" if is_last else "Next")
        # Previous is disabled on the first step — nothing to go back to.
        self._prev_btn.setEnabled(step_idx > 0)

    def dont_show_again(self) -> bool:
        return self._dont_show_chk.isChecked()


# ---------------------------------------------------------------------------
# Overlay widget
# ---------------------------------------------------------------------------


class WalkthroughOverlay(QWidget):
    """Translucent overlay + instruction popup that walks through `steps`.

    Public API:
        start()  — show overlay, jump to step 0.
        next_step() — advance one step (or finish at end).
        skip() / finish() — hide overlay, persist completion flag.
        reposition() — recompute highlight + popup position; call from the
            host window's resizeEvent and splitterMoved signal.
    """

    # Hole padding (px) around the target widget's geometry so the highlight
    # ring doesn't visually clip the widget's border.
    _HOLE_PADDING = 6
    # Gap (px) between the highlight rect and the popup.
    _POPUP_GAP = 12

    finished = pyqtSignal()

    def __init__(
        self,
        window: QWidget,
        steps: list[WalkthroughStep],
        settings: Optional["QSettings"] = None,
        settings_key: str = "walkthrough_completed",
    ):
        # Parent the overlay to the central widget so its rect == central
        # area (excluding menu/status bars). The popup is also parented to
        # the central widget — same coordinate space, easier positioning.
        central = window.centralWidget() if hasattr(window, "centralWidget") else window
        super().__init__(central)
        self._window = window
        self._central = central
        self._steps = steps
        self._settings = settings
        self._settings_key = settings_key
        self._idx = 0
        self._hole_rect = QRect()
        self._target_widget: Optional[QWidget] = None

        # Mouse + focus setup. Overlay catches every click outside the hole
        # so the user can't accidentally interact with grayed controls.
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        # The instruction popup is a sibling of the overlay, raised above it.
        self._popup = _WalkthroughPopup(central)
        self._popup.next_clicked.connect(self.next_step)
        self._popup.prev_clicked.connect(self.prev_step)
        self._popup.skip_clicked.connect(self.skip)
        # Hide both until start() is called.
        self.hide()
        self._popup.hide()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def start(self) -> None:
        if not self._steps:
            self.finish()
            return
        self._idx = 0
        self.setGeometry(self._central.rect())
        self.show()
        self.raise_()
        self._popup.show()
        self._popup.raise_()
        self._apply_current_step()
        self.setFocus(Qt.OtherFocusReason)

    def next_step(self) -> None:
        if self._idx + 1 >= len(self._steps):
            self.finish()
            return
        self._idx += 1
        self._apply_current_step()

    def prev_step(self) -> None:
        if self._idx <= 0:
            return
        self._idx -= 1
        self._apply_current_step()

    def skip(self) -> None:
        self.finish()

    def finish(self) -> None:
        # Only persist the "completed" flag if the user explicitly opted out
        # via the popup's checkbox. Otherwise the walkthrough re-fires on the
        # next launch — closing it (Skip / Finish / Esc) is a per-session
        # dismissal, not a permanent one.
        if self._settings is not None and self._popup.dont_show_again():
            self._settings.setValue(self._settings_key, True)
            self._settings.sync()
        self.hide()
        self._popup.hide()
        self.deleteLater()
        self._popup.deleteLater()
        self.finished.emit()

    def reposition(self) -> None:
        """Recompute the overlay rect + hole + popup position.

        Safe to call when no step is active (during construction or after
        finish()); it just returns early.
        """
        if not self.isVisible() or self._target_widget is None:
            return
        self.setGeometry(self._central.rect())
        self._update_hole_rect()
        self._update_mask()
        self._position_popup()
        self.update()

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------
    def _apply_current_step(self) -> None:
        step = self._steps[self._idx]
        if step.pre_action is not None:
            try:
                step.pre_action(self._window)
            except Exception:
                # Pre-actions are best-effort UX (e.g. switching tabs); never
                # let them break the walkthrough flow.
                pass
        try:
            self._target_widget = step.target_resolver(self._window)
        except Exception:
            self._target_widget = None
        is_last = self._idx + 1 >= len(self._steps)
        self._popup.set_content(step.title, step.body, self._idx, len(self._steps), is_last)
        self._popup.adjustSize()
        self.reposition()
        # Overlay first, then popup on top — otherwise the dim layer covers
        # the popup (which makes it look gray AND swallows its button clicks).
        self.raise_()
        self._popup.raise_()
        self.setFocus(Qt.OtherFocusReason)

    def _update_hole_rect(self) -> None:
        if self._target_widget is None:
            self._hole_rect = QRect()
            return
        target_origin = self._target_widget.mapTo(self._central, QPoint(0, 0))
        size = self._target_widget.size()
        rect = QRect(target_origin, size)
        # Pad the highlight ring outward so the border doesn't sit on top of
        # the widget's own pixel grid (looks cleaner around input fields).
        self._hole_rect = rect.adjusted(
            -self._HOLE_PADDING, -self._HOLE_PADDING, self._HOLE_PADDING, self._HOLE_PADDING
        )
        # Clamp to the overlay's bounds.
        self._hole_rect = self._hole_rect.intersected(self.rect())

    def _update_mask(self) -> None:
        # setMask drives BOTH painting AND mouse-event regions: areas outside
        # the mask are transparent AND click-through. We mask out the hole so
        # clicks on the highlighted widget pass through to it; the rest of
        # the overlay catches and absorbs clicks.
        from PyQt5.QtGui import QRegion

        full = QRegion(self.rect())
        if self._hole_rect.isEmpty():
            self.setMask(full)
            return
        mask = full.subtracted(QRegion(self._hole_rect))
        self.setMask(mask)

    def _position_popup(self) -> None:
        """Place popup beside the highlight: prefer right, fall back gracefully."""
        if self._hole_rect.isEmpty():
            # No target — center the popup in the overlay.
            popup_size = self._popup.sizeHint()
            x = (self.width() - popup_size.width()) // 2
            y = (self.height() - popup_size.height()) // 2
            self._popup.move(max(0, x), max(0, y))
            return
        popup_size = self._popup.sizeHint()
        hole = self._hole_rect
        central_w = self._central.width()
        central_h = self._central.height()
        gap = self._POPUP_GAP

        candidates = [
            # Right side — preferred.
            QRect(hole.right() + gap, hole.top(), popup_size.width(), popup_size.height()),
            # Left side.
            QRect(hole.left() - gap - popup_size.width(), hole.top(), popup_size.width(), popup_size.height()),
            # Below.
            QRect(hole.left(), hole.bottom() + gap, popup_size.width(), popup_size.height()),
            # Above.
            QRect(hole.left(), hole.top() - gap - popup_size.height(), popup_size.width(), popup_size.height()),
        ]
        for cand in candidates:
            if cand.left() >= 0 and cand.top() >= 0 and cand.right() <= central_w and cand.bottom() <= central_h:
                self._popup.setGeometry(cand)
                return
        # Last resort: clamp to the central rect even if it overlaps the hole.
        x = min(max(0, hole.right() + gap), central_w - popup_size.width())
        y = min(max(0, hole.top()), central_h - popup_size.height())
        self._popup.setGeometry(QRect(x, y, popup_size.width(), popup_size.height()))

    # -----------------------------------------------------------------------
    # Qt events
    # -----------------------------------------------------------------------
    # Visual parameters tuned to match the instruction popup's border
    # (`#WalkthroughPopup { border: 2px solid #4aa3ff; border-radius: 6px; }`)
    # so the highlight ring and the popup look like a matched pair.
    _ACCENT_RING_WIDTH = 2
    _ACCENT_RING_RADIUS = 6  # corner radius in px, matches popup
    # px from hole edge to the ring path; the 2 px pen centered on the
    # path then paints in [hole_edge+1, hole_edge+3] — fully outside the
    # hole, so the overlay's mask doesn't clip the stroke.
    _ACCENT_RING_OFFSET = 2

    def paintEvent(self, event):  # noqa: N802 — Qt API
        painter = QPainter(self)
        # Anti-alias the rounded corners so they read smooth, not jaggy.
        painter.setRenderHint(QPainter.Antialiasing, True)
        # Semi-transparent dark fill across the masked region (everything
        # except the hole).
        painter.fillRect(self.rect(), QColor(0, 0, 0, 150))
        # Accent ring fully outside the hole so all four sides of the
        # stroke land in the dim region and none of it gets mask-clipped.
        if not self._hole_rect.isEmpty():
            pen = QPen(QColor("#4aa3ff"))
            pen.setWidth(self._ACCENT_RING_WIDTH)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            offset = self._ACCENT_RING_OFFSET
            ring = self._hole_rect.adjusted(-offset, -offset, offset, offset)
            # Clamp to the overlay so we don't try to paint outside the
            # central widget's bounds when the highlighted target is at
            # the edge of the window.
            ring = ring.intersected(self.rect())
            painter.drawRoundedRect(ring, self._ACCENT_RING_RADIUS, self._ACCENT_RING_RADIUS)

    def keyPressEvent(self, event):  # noqa: N802 — Qt API
        if event.key() == Qt.Key_Escape:
            self.skip()
        else:
            super().keyPressEvent(event)
