; Inno Setup script — wraps the PyInstaller dist/TRACE/ folder into a
; single TRACE-Setup.exe installer with Start Menu shortcut, Desktop
; shortcut, and Add/Remove Programs entry.
;
; Compile with:  iscc installer.iss
;   (on Windows, after installing Inno Setup from https://jrsoftware.org/isinfo.php)
;   The GitHub Actions workflow uses the `Inno Setup Action` to run iscc
;   automatically.
;
; Output lands in TRACE/build/Output/TRACE-Setup.exe by default.

#define MyAppName "TRACE"
; MyAppVersion is normally overridden by CI via `iscc /DMyAppVersion=…`
; (single source of truth lives in TRACE/__init__.py — see
; .github/workflows/build-windows.yml). The literal below is the
; fallback for local builds where the override isn't passed.
#ifndef MyAppVersion
  #define MyAppVersion "0.1.42"
#endif
#define MyAppPublisher "Blair Lab"
#define MyAppURL "https://github.com/alexmpdx/TRACE"
#define MyAppExeName "TRACE.exe"
; Source folder produced by PyInstaller. The Actions workflow sets this
; with /DSourceDir=... so a custom path can be passed from CI.
#ifndef SourceDir
  #define SourceDir "..\..\dist\TRACE"
#endif

[Setup]
AppId={{A7C5D1F0-9D2B-4E3A-B6F8-5A1C2D3E4F50}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
; Per-user install (no admin rights required). Switch to {commonpf64}
; and set PrivilegesRequired=admin for a per-machine install.
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputBaseFilename=TRACE-Setup
; lzma2/normal cuts the compile step roughly in half vs lzma2/ultra64
; (the previous setting). The installer is ~50–100 MB larger as a
; result but the runtime model download dwarfs that, so the tradeoff
; is heavily in favor of faster builds.
Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
; Skip the "ready to install" page since we don't ask the user any questions.
DisableReadyPage=no
; trace_icon.ico is generated from TRACE/GUI_images/logo/logo_dark.svg —
; re-run `python TRACE/build/make_installer_icon.py` to regenerate after
; any logo edit. The .ico contains 16/24/32/48/64/128/256-px resolutions so
; Windows can pick the right size for taskbar, Explorer, and Start Menu.
SetupIconFile=trace_icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
  GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; Wipe the bundled-deps tree before laying down new files. Inno Setup's
; default install behavior only overwrites same-named files; it never
; deletes anything. Each pip-installed package's metadata lives in a
; folder whose name embeds the version (e.g. numpy-2.4.6.dist-info,
; shapely-2.1.2.dist-info), so updates that change package versions
; leave the OLD .dist-info dirs sitting alongside the new ones. Python's
; importlib.metadata then picks up whichever wins the dir listing, often
; producing an inconsistent install where shapely's .pyd asks the
; running numpy for a symbol that only exists in the version embedded in
; the stale metadata. The symptom is a startup ImportError pointing at
; numpy._core.umath (or a similar transitively-loaded native dep). Users
; have been working around this by uninstalling fully before reinstalling
; — the [InstallDelete] section does that for them automatically every
; update.
;
; The TRACE models cache at {app}\TRACE\models\ is NOT inside _internal/
; (it sits next to TRACE.exe at the install root), so the ~1.6 GB of
; downloaded weights are preserved across updates. Same for the desktop
; / Start Menu shortcuts (managed by [Icons], untouched by this).
Type: filesandordirs; Name: "{app}\_internal"

[Files]
; Pull in the entire PyInstaller bundle. recursesubdirs handles nested
; data files (GUI_images/, presets/, Qt plugins, etc.).
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
; {autodesktop} resolves to {userdesktop} for per-user installs (our default
; via PrivilegesRequired=lowest) and {commondesktop} when the user elevates to
; a per-machine install. The previous literal {commondesktop} unconditionally
; targeted C:\Users\Public\Desktop, which needs admin rights — non-admin
; installs hit "IPersistFile::Save failed; code 0x80070005. Access is denied."
; the moment the "Create desktop icon" task was checked.
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; \
  Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; \
  Flags: nowait postinstall skipifsilent

[UninstallDelete]
; The bundled-models cache + tooltip cache are added at runtime; clean
; them up so uninstall leaves no orphan files. (Models live under {app}
; so the [Files] uninstall handles them, but the tooltip cache is in
; %TEMP%\trace_tooltip_cache.)
Type: filesandordirs; Name: "{%TEMP}\trace_tooltip_cache"
