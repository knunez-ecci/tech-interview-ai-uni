import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging():
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logs"))
    os.makedirs(log_dir, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    if not root.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        root.addHandler(ch)

        fh = RotatingFileHandler(
            os.path.join(log_dir, "app.log"),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Silenciar loggers ruidosos pero mantener uvicorn.access para ver requests en consola
    for name in ("sentence_transformers", "transformers", "torch"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # uvicorn.access muestra cada GET/POST en consola — usar formato estándar
    access_log = logging.getLogger("uvicorn.access")
    access_log.setLevel(logging.INFO)
    if not access_log.handlers:
        access_ch = logging.StreamHandler()
        access_ch.setFormatter(fmt)
        access_log.addHandler(access_ch)
    access_log.propagate = False
