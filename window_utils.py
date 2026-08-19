"""Helpers para janelas OpenCV."""
import cv2
import numpy as np

_screen = None


def _screen_size():
    global _screen
    if _screen is not None:
        return _screen
    _screen = (None, None)
    try:
        import sys as _sys
        if _sys.platform == "darwin":
            import ctypes
            CG = ctypes.CDLL(
                "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
            )
            CG.CGMainDisplayID.restype = ctypes.c_uint32

            class _CGPoint(ctypes.Structure):
                _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

            class _CGSize(ctypes.Structure):
                _fields_ = [("w", ctypes.c_double), ("h", ctypes.c_double)]

            class _CGRect(ctypes.Structure):
                _fields_ = [("origin", _CGPoint), ("size", _CGSize)]

            CG.CGDisplayBounds.restype = _CGRect
            b = CG.CGDisplayBounds(CG.CGMainDisplayID())
            _screen = (int(b.size.w), int(b.size.h))
        elif _sys.platform == "win32":
            import ctypes
            u = ctypes.windll.user32
            _screen = (u.GetSystemMetrics(0), u.GetSystemMetrics(1))
        else:
            import re
            import subprocess
            out = subprocess.check_output(["xrandr"], text=True)
            for line in out.splitlines():
                if " connected " not in line:
                    continue
                m = re.search(r"(\d+)x(\d+)\+", line)
                if m:
                    _screen = (int(m.group(1)), int(m.group(2)))
                    break
    except Exception:
        _screen = (None, None)
    return _screen


def show(name, frame):
    """Mostra o frame em tela cheia sem distorcao (letterbox)."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    sw, sh = _screen_size()
    if not sw or not sh:
        cv2.imshow(name, frame)
        return

    h, w = frame.shape[:2]
    scale = min(sw / w, sh / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    if (nw, nh) != (w, h):
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (nw, nh), interpolation=interp)
    else:
        resized = frame

    canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
    x = (sw - nw) // 2
    y = (sh - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized

    cv2.imshow(name, canvas)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
