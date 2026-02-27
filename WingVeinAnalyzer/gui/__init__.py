"""WingVeinAnalyzer interactive step-by-step GUI."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def main() -> None:
    """Launch the WingVeinAnalyzer GUI."""
    # Log to both console and file
    log_path = Path(__file__).resolve().parent.parent / "gui.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(str(log_path), mode="w"),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("GUI log file: %s", log_path)

    from PyQt5.QtWidgets import QApplication

    from WingVeinAnalyzer.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("WingVeinAnalyzer")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
