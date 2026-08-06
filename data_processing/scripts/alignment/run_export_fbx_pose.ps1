param(
    [string]$FbxPath = "C:\Users\hand\Desktop\Dataset\0714\002\SIK_Actor_01_20260714_121232.fbx",
    [string]$OutDir = "",
    [string]$BlenderExe = "",
    [int]$FrameStep = 1,
    [switch]$OnlyDeformBones
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $FbxPath)) {
    throw "FBX not found: $FbxPath"
}

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path (Split-Path -Parent $FbxPath) "fbx_pose_export"
}

if ([string]::IsNullOrWhiteSpace($BlenderExe)) {
    $cmd = Get-Command blender.exe -ErrorAction SilentlyContinue
    if ($cmd) {
        $BlenderExe = $cmd.Source
    }
}

if ([string]::IsNullOrWhiteSpace($BlenderExe)) {
    $candidates = @(
        "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
        "C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
        "C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
        "C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            $BlenderExe = $candidate
            break
        }
    }
}

if ([string]::IsNullOrWhiteSpace($BlenderExe) -or -not (Test-Path -LiteralPath $BlenderExe)) {
    throw @"
Cannot find blender.exe.

Install Blender or pass its path manually, for example:
powershell -ExecutionPolicy Bypass -File "C:\Users\hand\Desktop\Dataset\tools\run_export_fbx_pose.ps1" -BlenderExe "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
"@
}

$scriptPath = Join-Path $PSScriptRoot "export_fbx_pose_to_csv.py"
$argsList = @(
    "--background",
    "--python", $scriptPath,
    "--",
    "--fbx", $FbxPath,
    "--outdir", $OutDir,
    "--frame-step", "$FrameStep"
)

if ($OnlyDeformBones) {
    $argsList += "--only-deform-bones"
}

Write-Host "Blender: $BlenderExe"
Write-Host "FBX:     $FbxPath"
Write-Host "OutDir:  $OutDir"
& $BlenderExe @argsList

if ($LASTEXITCODE -ne 0) {
    throw "Blender export failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Done."
Write-Host "Pose CSV:     $(Join-Path $OutDir 'pose_frames.csv')"
Write-Host "Skeleton CSV: $(Join-Path $OutDir 'skeleton_bones.csv')"
Write-Host "Metadata:     $(Join-Path $OutDir 'metadata.json')"
