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
#define MyAppVersion "0.1.0"
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
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
; Skip the "ready to install" page since we don't ask the user any questions.
DisableReadyPage=no
SetupIconFile=
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
  GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Pull in the entire PyInstaller bundle. recursesubdirs handles nested
; data files (GUI_images/, presets/, Qt plugins, etc.).
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

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
