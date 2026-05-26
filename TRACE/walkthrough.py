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
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QPixmap
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
    rect_resolver:
        Optional callable returning the QRect to highlight, in the central
        widget's coordinate space. When given, it overrides target_resolver's
        widget geometry — used to highlight something that is not a whole
        widget, e.g. a single tab within the tab bar. Receives the main
        window as its only argument.
    """

    target_resolver: Callable[[QWidget], QWidget]
    title: str
    body: str
    pre_action: Optional[Callable[[QWidget], None]] = field(default=None)
    rect_resolver: Optional[Callable[[QWidget], "QRect"]] = field(default=None)


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
        # Orange border so the popup reads as distinct from the blue
        # highlight ring drawn around the target widget.
        self.setStyleSheet(
            "#WalkthroughPopup { "
            "background-color: #2d2d2d; "
            "border: 2px solid #ff9d4a; "
            "border-radius: 6px; "
            "}"
        )
        # Min width has to cover the footer at its natural size (counter +
        # Skip tutorial + Previous + Next buttons) — at 280 the buttons
        # collapse below their text width and the "Skip tutorial" / "Don't
        # show again on launch" labels visibly clip. The max gives long
        # bodies room to render on fewer lines without dominating the window.
        self.setMinimumWidth(380)
        self.setMaximumWidth(480)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        self._title = QLabel("")
        title_font = QFont(self._title.font())
        title_font.setBold(True)
        title_font.setPointSize(title_font.pointSize() + 1)
        self._title.setFont(title_font)
        self._title.setStyleSheet("color: #d0d0d0;")
        # Wrap long headers onto a second line instead of clipping them at
        # the popup's max width.
        self._title.setWordWrap(True)
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
        # Outlined (not flat) so it clearly reads as a clickable button
        # rather than plain label text.
        self._skip_btn.setStyleSheet(
            "QPushButton { color: #999; background-color: transparent; "
            "border: 1px solid #5a5a5a; border-radius: 4px; padding: 4px 10px; } "
            "QPushButton:hover { border-color: #ff9d4a; color: #c0c0c0; } "
            "QPushButton:pressed { background-color: #3a3a3a; }"
        )
        self._skip_btn.clicked.connect(self.skip_clicked)
        footer.addWidget(self._skip_btn)
        self._prev_btn = QPushButton("Previous")
        self._prev_btn.clicked.connect(self.prev_clicked)
        footer.addWidget(self._prev_btn)
        self._next_btn = QPushButton("Next")
        self._next_btn.setDefault(True)
        # Orange outline to match the popup border — and to override the
        # Fusion default-button's blue highlight ring.
        self._next_btn.setStyleSheet(
            "QPushButton { color: #d0d0d0; background-color: #3a3a3a; "
            "border: 1px solid #ff9d4a; border-radius: 4px; padding: 4px 12px; } "
            "QPushButton:hover { background-color: #454545; } "
            "QPushButton:pressed { background-color: #2f2f2f; }"
        )
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
        # Clean snapshot of the UI (overlay hidden) that paintEvent dims.
        # Re-grabbed each step — see _capture_dim_snapshot.
        self._dim_pixmap: Optional[QPixmap] = None

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
        self._capture_dim_snapshot()
        self.setGeometry(self._central.rect())
        self._update_hole_rect()
        self._update_mask()
        prev_popup_geom = self._popup.geometry()
        self._position_popup()
        # If the popup moved, clear any ghost it left over the masked hole.
        if self._popup.geometry() != prev_popup_geom:
            self._clear_popup_ghost()
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
        prev_target = self._target_widget
        try:
            self._target_widget = step.target_resolver(self._window)
        except Exception:
            self._target_widget = None
        is_last = self._idx + 1 >= len(self._steps)
        self._popup.set_content(step.title, step.body, self._idx, len(self._steps), is_last)
        self._popup.adjustSize()
        self.reposition()
        # The popup is stacked above the overlay's masked-out highlight hole,
        # so when it moves between steps Qt won't reliably repaint what was
        # underneath — leaving a gray ghost of the old popup. Force the
        # affected widgets to redraw.
        self._clear_popup_ghost(prev_target)
        # Overlay first, then popup on top — otherwise the dim layer covers
        # the popup (which makes it look gray AND swallows its button clicks).
        self.raise_()
        self._popup.raise_()
        self.setFocus(Qt.OtherFocusReason)

    def _clear_popup_ghost(self, prev_target: Optional[QWidget] = None) -> None:
        """Repaint regions a moved popup may have left ghosted.

        Where the overlay's mask punches the highlight hole, Qt does not
        always repaint the widget revealed beneath a popup that moved away.
        A `central.update()` only recomposites cached backing stores, which
        isn't enough — so we also explicitly repaint the highlighted widget
        and its scroll viewport (QListWidget / QTextEdit paint their content
        on a separate `viewport()` child).
        """
        self._central.update()
        for wdg in (prev_target, self._target_widget):
            if wdg is None:
                continue
            wdg.update()
            viewport = getattr(wdg, "viewport", None)
            if callable(viewport):
                try:
                    viewport().update()
                except Exception:
                    pass

    def _capture_dim_snapshot(self) -> None:
        """Grab a clean snapshot of the UI for paintEvent to dim.

        The overlay is a translucent child widget, so Qt would otherwise
        render its dim by propagating the parent's content into its backing
        — and that content already includes this overlay, a feedback loop
        that leaves ghosts of previous steps' highlight rings. Instead we
        briefly hide the overlay and popup, render the parent to an
        off-screen pixmap (grab() does not touch the screen, so no flash),
        and dim that pixmap in paintEvent. Re-grabbed every reposition so the
        snapshot tracks tab switches, scrolling and resizes.
        """
        overlay_visible = self.isVisible()
        popup_visible = self._popup.isVisible()
        self.hide()
        self._popup.hide()
        self._dim_pixmap = self._central.grab()
        if overlay_visible:
            self.show()
        if popup_visible:
            self._popup.show()

    def _update_hole_rect(self) -> None:
        # A step may supply an explicit rect (in central-widget coordinates)
        # via rect_resolver — used to highlight something that isn't a whole
        # widget, e.g. a single tab within the tab bar. Otherwise the hole is
        # the target widget's full geometry.
        rect: Optional[QRect] = None
        step = self._steps[self._idx] if 0 <= self._idx < len(self._steps) else None
        if step is not None and step.rect_resolver is not None:
            try:
                rect = step.rect_resolver(self._window)
            except Exception:
                rect = None
        if rect is None:
            if self._target_widget is None:
                self._hole_rect = QRect()
                return
            target_origin = self._target_widget.mapTo(self._central, QPoint(0, 0))
            rect = QRect(target_origin, self._target_widget.size())
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
        """Place the popup beside the highlight without covering it.

        Each side (right, left, below, above) is tried in preference order:
        the candidate is first clamped fully onto the central widget, then
        accepted only if it still clears the highlight rect. Clamping means a
        candidate that would have run off-screen is pulled back on-screen
        rather than discarded — it is rejected only if pulling it back makes
        it overlap the highlight. If no side clears the highlight (it nearly
        fills the window), the least-overlapping placement is used.
        """
        popup_size = self._popup.sizeHint()
        pw = popup_size.width()
        ph = popup_size.height()
        central_w = self._central.width()
        central_h = self._central.height()

        if self._hole_rect.isEmpty():
            # No target — center the popup in the overlay.
            self._popup.move(max(0, (central_w - pw) // 2), max(0, (central_h - ph) // 2))
            return

        hole = self._hole_rect
        gap = self._POPUP_GAP

        # Top-left corner for each side, in preference order.
        side_origins = [
            (hole.right() + gap, hole.top()),  # right — preferred
            (hole.left() - gap - pw, hole.top()),  # left
            (hole.left(), hole.bottom() + gap),  # below
            (hole.left(), hole.top() - gap - ph),  # above
        ]
        best: Optional[QRect] = None
        best_overlap_area: Optional[int] = None
        for x, y in side_origins:
            # Pull the candidate fully onto the central widget.
            x = max(0, min(x, central_w - pw))
            y = max(0, min(y, central_h - ph))
            rect = QRect(x, y, pw, ph)
            if not rect.intersects(hole):
                # Clears the highlight — use the first such side.
                self._popup.setGeometry(rect)
                return
            # Otherwise remember whichever side overlaps the highlight least.
            overlap = rect.intersected(hole)
            area = overlap.width() * overlap.height()
            if best_overlap_area is None or area < best_overlap_area:
                best, best_overlap_area = rect, area
        # No side fully clears the highlight — use the least-bad placement.
        if best is not None:
            self._popup.setGeometry(best)

    # -----------------------------------------------------------------------
    # Qt events
    # -----------------------------------------------------------------------
    # Highlight ring: same 2 px width / 6 px corner radius as the instruction
    # popup's border, but blue (`#4aa3ff`) rather than the popup's orange so
    # the ring around the target reads as distinct from the popup itself.
    _ACCENT_RING_WIDTH = 2
    _ACCENT_RING_RADIUS = 6  # corner radius in px
    # px from hole edge to the ring path; the 2 px pen centered on the
    # path then paints in [hole_edge+1, hole_edge+3] — fully outside the
    # hole, so the overlay's mask doesn't clip the stroke.
    _ACCENT_RING_OFFSET = 2

    def paintEvent(self, event):  # noqa: N802 — Qt API
        painter = QPainter(self)
        # Anti-alias the rounded corners so they read smooth, not jaggy.
        painter.setRenderHint(QPainter.Antialiasing, True)
        # Paint the cached clean snapshot of the UI, then a translucent dark
        # layer over it — together these are the "dim". The opaque snapshot
        # also fully overwrites the backing each frame, so nothing from a
        # previous step (old dim, old ring) can survive. (See
        # _capture_dim_snapshot for why a snapshot is used at all.)
        if self._dim_pixmap is not None:
            painter.drawPixmap(0, 0, self._dim_pixmap)
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
