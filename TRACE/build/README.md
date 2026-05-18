# TRACE Windows build

How the downloadable Windows installer (`TRACE-Setup.exe`) is produced.

## Quick path: trigger a build, get an installer

1. **Tag a release** locally (anywhere) and push it:
   ```bash
   git tag windows-v0.1
   git push origin windows-v0.1
   ```
2. GitHub Actions picks up the tag and runs `.github/workflows/build-windows.yml`
   on a `windows-latest` runner (~20–30 min cold).
3. When it finishes, the installer is attached to the GitHub Release for that tag.
   Anyone with repo access can download `TRACE-Setup.exe` from the Releases page.
4. Double-click → installs to `%LocalAppData%\Programs\TRACE\`, adds Start Menu shortcut.
5. First launch downloads the 1.6 GB model bundle from the `v1.0-assets` release
   (existing `TRACE/fetch_assets.py` flow — unchanged by the Windows packaging).

You can also build without tagging via the **Actions tab → "Build Windows installer" →
Run workflow** button. That uploads `TRACE-Setup.exe` as a 30-day artifact instead of
attaching it to a release.

## What's in this folder

| File                          | Role                                                       |
| ----------------------------- | ---------------------------------------------------------- |
| `requirements-windows.txt`    | Consolidated Python deps. CPU PyTorch (no CUDA).           |
| `trace.spec`                  | PyInstaller spec — builds `dist/TRACE/` (onedir bundle).   |
| `installer.iss`               | Inno Setup script — wraps `dist/TRACE/` into the .exe.     |
| `README.md`                   | This file.                                                 |

The corresponding CI workflow lives at `.github/workflows/build-windows.yml`.

## Local build (on a Windows machine)

If you have a Windows box and want to iterate without going through CI:

```powershell
# 1. Set up a fresh venv
py -3.11 -m venv .venv
.\.venv\Scripts\activate

# 2. Install deps + sibling packages
pip install -r TRACE\build\requirements-windows.txt
pip install -e identifyFeatures
pip install -e measurementMaker
pip install -e LandmarkLocator

# 3. Bundle Python + deps into dist\TRACE\
pyinstaller TRACE\build\trace.spec --noconfirm

# 4. (Optional) Wrap into an installer
#    Requires Inno Setup 6 from https://jrsoftware.org/isinfo.php
iscc /DSourceDir=..\..\dist\TRACE TRACE\build\installer.iss
# → TRACE\build\Output\TRACE-Setup.exe
```

## Trade-offs in the current setup

- **CPU-only PyTorch.** Inference runs everywhere but on a slower path
  than CUDA. To ship a CUDA build, replace the `--extra-index-url` line
  in `requirements-windows.txt` with the matching CUDA index from
  https://pytorch.org/get-started/locally/. Installer grows by ~1.5 GB.
- **No code signing.** Windows SmartScreen warns users on first launch
  ("Microsoft Defender SmartScreen prevented an unrecognized app from
  starting"). Users click *More info → Run anyway*. To remove the
  warning, buy an EV code-signing certificate (~$200/yr) and add a
  signing step before `iscc`.
- **Per-user install.** Default install dir is `%LocalAppData%\Programs\TRACE\`
  (no admin prompt). To install per-machine instead, change
  `DefaultDirName={autopf}\TRACE` to `{commonpf64}\TRACE` and set
  `PrivilegesRequired=admin` in `installer.iss`.

## Iterating on build failures

The first 1–3 builds will probably fail with `ImportError` for some
sub-module PyInstaller's static analysis missed. The fix pattern:

1. Open the build log in the Actions run.
2. Find the `ImportError: No module named …` line.
3. Add the missing module to `hiddenimports` in `trace.spec`.
4. Push and re-build.

Common culprits: dynamically-imported submodules of `scipy`, `sklearn`,
`skimage`, `napari`. Each fix is a one-line addition to the spec.
