# PyInstaller spec for the TRACE Windows build.
#
# Build with:  pyinstaller TRACE/build/trace.spec --noconfirm
# Produces:    dist/TRACE/  (onedir bundle with TRACE.exe at its root)
#
# Onedir (not onefile) because:
#   - Faster startup (no per-launch extraction to a temp folder)
#   - Easier to debug missing files
#   - Inno Setup wraps the whole folder cleanly
#
# Hidden imports cover the sibling packages that TRACE.gui adds to sys.path
# at runtime — PyInstaller's static analysis misses dynamic sys.path
# additions, so it would otherwise skip bundling those packages' modules.

import sys
from pathlib import Path

# Repo root: TRACE/build/trace.spec → TRACE/build → TRACE → mapThemVeins
ROOT = Path(SPECPATH).parent.parent
TRACE = ROOT / "TRACE"

block_cipher = None

# --- Sibling packages bundled alongside TRACE -----------------------------
# Each sibling exports a top-level module that TRACE.gui (or its imports)
# needs at runtime. PyInstaller pulls them in via the `pathex` search path
# plus explicit hiddenimports.
_SIBLINGS = [
    "HingeChopper",
    "modelTOjson",
    "identifyFeatures",
    "wingRotator",
    "measurementMaker",
    "scaleEstimator",
    "wingIsolator",
    "resolutionAdjust",
    "LandmarkLocator",
    "preprocessing",
    # CPU_RAM_tester hosts recommend_workers.py — the Calibrate Workers
    # button on the Settings tab adds this dir to sys.path at runtime.
    "CPU_RAM_tester",
]
_pathex = [str(ROOT / s) for s in _SIBLINGS]

# Modules whose imports are too dynamic for PyInstaller's static analysis.
# Add to this list if a fresh Windows launch raises ImportError.
hiddenimports = [
    # Sibling top-level packages
    "hinge_chopper",
    # Sibling top-level scripts (not packages). PyInstaller's static
    # analysis resolves these against the host's installed packages
    # rather than the sibling-dir that gets added to sys.path at runtime,
    # so they need to be pinned explicitly.
    "hinge_psd_loader",   # HingeChopper/hinge_psd_loader.py
    "recommend_workers",  # CPU_RAM_tester/recommend_workers.py (Calibrate Workers)
    "wing_rotator",       # wingRotator/wing_rotator.py (Rotate-wing preprocessing)
    "modeltojson",
    "identify_features",
    "wingrotator",
    "measurement_maker",
    "scale_estimator",
    "wingIsolator",
    "resolutionAdjust",
    "landmark_locator",
    "preprocessing",
    # Heavy native deps with sub-modules PyInstaller misses
    "torch",
    "torchvision",
    "cv2",
    "scipy.special.cython_special",
    "skimage.io._plugins",
    "sklearn.utils._cython_blas",
    "pymupdf",
    "rawpy",
    "tifffile",
    "psd_tools",
    "napari",
    # PyQt5 plugins
    "PyQt5.sip",
    # QtSvg is needed for QIcon to load logo_dark.svg as the window icon.
    # PyInstaller's default PyQt5 hook bundles the qsvgicon image plugin
    # most of the time, but pinning the import is belt-and-suspenders.
    "PyQt5.QtSvg",
]

# --- Data files (kept relative to TRACE.exe at runtime) -------------------
# Format: (source_glob, target_dir_inside_bundle)
datas = [
    (str(TRACE / "GUI_images"), "TRACE/GUI_images"),
    (str(TRACE / "presets"), "TRACE/presets"),
    (str(TRACE / "README.md"), "TRACE"),
]
# Initialized empty so the collect_all loop below can append to it before
# the Analysis call. The Analysis() invocation receives this list directly.
binaries = []

# napari + its Qt ecosystem rely heavily on .dist-info metadata at runtime
# (importlib.metadata.distribution(...) lookups for entry points, version,
# plugin discovery). PyInstaller's static analysis often skips those
# folders, which produces napari's "No package metadata was found for An
# error occurred when importing Qt dependencies. Cannot show napari
# window" error on first launch — the metadata lookup that should reveal
# the real underlying ImportError fails first.
#
# Use copy_metadata for everything napari touches at import time, and
# collect_all (data + binaries + hidden submodules) for the major
# packages so we don't have to chase down individual misses.
from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

_METADATA_PACKAGES = [
    "napari",
    "napari-svg",
    "napari-console",
    "napari-builtins",
    "qtpy",
    "magicgui",
    "superqt",
    "psygnal",
    "vispy",
    "PyQt5",
    "pint",
    "app_model",
    "in_n_out",
    "pydantic",
    "numpydoc",
    "freetype-py",
    # imageio — root cause from an earlier failed launch.
    # napari_builtins.io.__init__ imports imageio, which at module-init
    # time calls importlib.metadata.version("imageio") to set __version__.
    # Without the .dist-info that call raises PackageNotFoundError.
    "imageio",
    "imageio-ffmpeg",
    # ome_types ships a napari.yaml manifest that the napari plugin
    # discovery hits during viewer creation.
    "ome-types",
    # tifffile / dask occasionally get probed by napari io
    "tifffile",
    "dask",
    # torch's c10.dll fails to load on Windows when its sibling DLLs
    # (caffe2, fbgemm, etc.) and dist-info aren't fully bundled — surfaces
    # as "WinError 1114: A dynamic link library (DLL) initialization
    # routine failed" on first PyTorch import (e.g. landmark model load).
    "torch",
    "torchvision",
    # rasterio — modelTOjson uses it for georeferenced raster I/O and it
    # has many lazily-imported submodules (rasterio.serde and friends)
    # that PyInstaller misses without copy_metadata.
    "rasterio",
]
for pkg in _METADATA_PACKAGES:
    try:
        datas += copy_metadata(pkg)
    except Exception:
        # Some packages may not be installed (e.g. napari_console pulls in
        # qtconsole which we may not have); skip silently rather than
        # failing the whole build on a name that wasn't there.
        pass

# collect_all pulls in submodules, data files, AND binaries. Add the
# napari plugin packages here so their plugin manifest YAMLs land in the
# bundle — they're not Python imports so collect_submodules misses them.
# torch + torchvision are listed so all their bundled DLLs (caffe2,
# fbgemm, cudnn stubs, etc.) end up in the dist folder; without
# collect_all the static-analysis path can miss some of c10.dll's
# transitive native deps.
_COLLECT_ALL_PACKAGES = [
    "napari",
    "napari_builtins",
    "napari_console",
    "napari_svg",
    "vispy",
    "magicgui",
    "superqt",
    "qtpy",
    "imageio",
    "ome_types",
    "torch",
    "torchvision",
    # rasterio — see comment in _METADATA_PACKAGES above.
    "rasterio",
    # certifi bundles cacert.pem — fetch_assets.make_ssl_context() points
    # urllib at it so frozen Windows builds can verify GitHub's TLS cert.
    # Without collect_all, the .pem file doesn't make it into the bundle
    # and certifi.where() returns a path that doesn't exist at runtime.
    "certifi",
]
for pkg in _COLLECT_ALL_PACKAGES:
    try:
        _data, _bin, _hidden = collect_all(pkg)
        datas += _data
        binaries += _bin
        hiddenimports += _hidden
    except Exception:
        pass

# onnxruntime is imported lazily inside OnnxModelWrapper.__init__
# (modelTOjson/modeltojson.py) so PyInstaller's static analysis misses
# it. Without collect_all we also lose the native DLLs and the capi/
# subdir, so the wing-isolation ONNX model fails to load at runtime
# with the misleading "onnxruntime is required for ONNX models" error.
#
# Pulled out of _COLLECT_ALL_PACKAGES on purpose: this one is mandatory
# for wing isolation. The blanket try/except above silently masks
# collection failures, which previously shipped a broken installer with
# no onnxruntime bundled. Letting this raise loudly turns a silent
# runtime failure into an immediate build failure.
import onnxruntime  # noqa: F401  build-time presence check
_ort_data, _ort_bin, _ort_hidden = collect_all("onnxruntime")
datas += _ort_data
binaries += _ort_bin
hiddenimports += _ort_hidden
# collect_all should catch onnxruntime/capi/*.dll already, but pull it
# in explicitly as belt-and-suspenders — the native DLLs are what fail
# to load on machines without VC++ runtime / when bundling skips them.
binaries += collect_dynamic_libs("onnxruntime")

# Belt-and-suspenders: explicit submodule lists for the ones that
# matter most. collect_all should cover these but a duplicate entry is
# cheaper than a missed one.
hiddenimports += collect_submodules("napari")
hiddenimports += collect_submodules("vispy")
hiddenimports += collect_submodules("magicgui")
hiddenimports += collect_submodules("imageio")


a = Analysis(
    [str(TRACE / "run_gui.py")],
    pathex=[str(ROOT)] + _pathex,
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Test suites — bloat without value for end users.
        "pytest",
        "tornado",
        # CUDA libs — we ship CPU torch.
        "torch.cuda",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TRACE",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI app — no console window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Embeds the TRACE logo as the .exe's Windows shell icon, so Add/Remove
    # Programs, the Start Menu shortcut, the taskbar, and Explorer all show
    # the TRACE wireframe instead of the generic application icon. The same
    # .ico is also used by Inno Setup via SetupIconFile= in installer.iss.
    icon=str(TRACE / "build" / "trace_icon.ico"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="TRACE",
)
