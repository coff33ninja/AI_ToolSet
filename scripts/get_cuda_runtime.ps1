<#
.SYNOPSIS
    Provisions the CUDA/cuDNN runtime this project needs - TensorFlow into a
    project-local cuda_runtime/ folder, or (for PyTorch) verifies the driver
    only, since PyTorch wheels bundle CUDA/cuDNN inside themselves.

.DESCRIPTION
    TensorFlow on Windows requires matching CUDA and cuDNN runtime DLLs but
    does NOT require a system-wide CUDA toolkit install. This script downloads
    the exact build needed for your TensorFlow version and extracts just the
    DLLs into <repo>/cuda_runtime/bin.

    Every source is NVIDIA-official:
      * CUDA runtime DLLs (cudart, cublas, cufft, curand, cusolver, cusparse)
        come from NVIDIA's own redistributable wheels on PyPI
        (nvidia-*-cu11), fetched with `uv pip download`.
      * cuDNN comes from NVIDIA's redist server
        (developer.download.nvidia.com/compute/redist).

    Nothing is guessed:
      * TensorFlow version -> CUDA/cuDNN mapping is a verified table.
      * Wheel versions are pinned to known-good releases.
      * cuDNN always uses the official NVIDIA redistributable - third-party
        cudnn builds crash TensorFlow with 0xC0000409 on the first GPU op.
      * Your NVIDIA driver version is checked against the minimum required
        and a warning is printed if it is too old.

    sitecustomize.py is installed into the venv so cuda_runtime/bin is
    prepended to PATH at every interpreter startup before TensorFlow loads.

    PyTorch mode (-Framework pytorch) downloads nothing: the CUDA 11.8 wheels
    from download.pytorch.org bundle cuDNN + CUDA runtime in torch/lib. The
    NVIDIA driver is the only system requirement. Everything stays inside the
    project venv - no global installs, ever.

.PARAMETER Framework
    Which runtime to provision: 'tensorflow' (downloads CUDA/cuDNN DLLs),
    'pytorch' (driver check only, torch bundles its own runtime), or
    'fasterwhisper' (driver check only - the CUDA 12 runtime comes from
    NVIDIA's own redistributable wheels in the 'stt' extra).

.PARAMETER TensorFlowVersion
    TensorFlow major.minor whose runtime this project needs. Default 2.10.
    Supported: 2.4 (CUDA 11.0), 2.5-2.10 (CUDA 11.2).

.PARAMETER CudaVersion
    Override the CUDA runtime version (e.g. "11.2").

.PARAMETER CudnnVersion
    Override the cuDNN full version (e.g. "8.1.0.77").

.PARAMETER TargetDir
    Where to put cuda_runtime/. Default <repo>/cuda_runtime.

.PARAMETER WorkDir
    Scratch dir for downloads. Default $env:TEMP\ai_toolset_cuda.

.PARAMETER SkipSitecustomize
    Do not install sitecustomize.py into the venv.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/get_cuda_runtime.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/get_cuda_runtime.ps1 -TensorFlowVersion 2.4

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/get_cuda_runtime.ps1 -Framework pytorch
#>

[CmdletBinding()]
param(
    [ValidateSet('tensorflow', 'pytorch', 'fasterwhisper')]
    [string]$Framework = 'tensorflow',
    [string]$TensorFlowVersion = '2.10',
    [string]$CudaVersion,
    [string]$CudnnVersion,
    [string]$TargetDir,
    [string]$WorkDir,
    [switch]$SkipSitecustomize
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
if (-not $TargetDir) { $TargetDir = Join-Path $root 'cuda_runtime' }
if (-not $WorkDir) { $WorkDir = Join-Path $env:TEMP 'ai_toolset_cuda' }
$binDir = Join-Path $TargetDir 'bin'

$tfMatrix = @{
    '2.4'  = @{ Cuda = '11.0'; Cudnn = '8.0.5.39'; DriverMin = 451.82 }
    '2.5'  = @{ Cuda = '11.2'; Cudnn = '8.1.0.77'; DriverMin = 460.89 }
    '2.6'  = @{ Cuda = '11.2'; Cudnn = '8.1.0.77'; DriverMin = 460.89 }
    '2.7'  = @{ Cuda = '11.2'; Cudnn = '8.1.0.77'; DriverMin = 460.89 }
    '2.8'  = @{ Cuda = '11.2'; Cudnn = '8.1.0.77'; DriverMin = 460.89 }
    '2.9'  = @{ Cuda = '11.2'; Cudnn = '8.1.0.77'; DriverMin = 460.89 }
    '2.10' = @{ Cuda = '11.2'; Cudnn = '8.1.0.77'; DriverMin = 460.89 }
}

# CUDA runtime DLLs from NVIDIA's official redistributable wheels on PyPI.
# These are CUDA 11.x wheels (the cu11 lineage); TensorFlow 2.x links them by
# name and CUDA-major, so a newer 11.x runtime is fully compatible with the
# 11.2 build TF 2.10 was validated against.
$cu11Wheels = @(
    'nvidia-cuda-runtime-cu11==11.8.89',
    'nvidia-cublas-cu11==11.11.3.6',
    'nvidia-cufft-cu11==10.9.0.58',
    'nvidia-curand-cu11==10.3.0.86',
    'nvidia-cusolver-cu11==11.4.1.48',
    'nvidia-cusparse-cu11==11.7.5.86'
)

if ($Framework -eq 'pytorch') {
    # PyTorch CUDA 11.8 wheels (download.pytorch.org/whl/cu118) bundle the
    # CUDA + cuDNN runtime inside the wheel (torch/lib). No system CUDA
    # toolkit and no DLL download - the NVIDIA driver is the only system
    # requirement. Minimum driver for CUDA 11.8 on Windows is 452.39.
    $CudaVersion = '11.8'
    $CudnnVersion = 'bundled-in-torch'
    $minDriver = 452.39
} elseif (-not $CudaVersion -or -not $CudnnVersion) {
    if (-not $tfMatrix.ContainsKey($TensorFlowVersion)) {
        throw "No verified CUDA/cuDNN mapping for TensorFlow $TensorFlowVersion. Supported: $($tfMatrix.Keys -join ', '). Pass -CudaVersion/-CudnnVersion explicitly."
    }
    $entry = $tfMatrix[$TensorFlowVersion]
    if (-not $CudaVersion) { $CudaVersion = $entry.Cuda }
    if (-not $CudnnVersion) { $CudnnVersion = $entry.Cudnn }
    $minDriver = $entry.DriverMin
} else {
    $minDriver = 451.82
}

$cudnnPatch = ($CudnnVersion -split '\.')[0..2] -join '.'
$cudnnUrl = "https://developer.download.nvidia.com/compute/redist/cudnn/v$cudnnPatch/cudnn-$CudaVersion-windows-x64-v$CudnnVersion.zip"
$cudnnFile = "cudnn-$CudaVersion-windows-x64-v$CudnnVersion.zip"

function Get-Python {
    $venvPy = Join-Path $root '.venv\Scripts\python.exe'
    if (Test-Path $venvPy) { return $venvPy }
    return 'python'
}

function Assert-DriverCompatible {
    param([string]$Tool = 'TensorFlow')
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) {
        Write-Warning "nvidia-smi not found on PATH. Install/update NVIDIA drivers - $Tool GPU needs a recent driver (>= $minDriver for CUDA $CudaVersion)."
        return
    }
    $line = (& nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>$null) | Select-Object -First 1
    if (-not $line) {
        Write-Warning "nvidia-smi ran but returned nothing - is an NVIDIA GPU present?"
        return
    }
    $parts = $line -split ','
    $driver = [double]($parts[0].Trim())
    $gpu = $parts[1].Trim()
    Write-Host "Detected GPU: $gpu"
    Write-Host "Detected driver: $driver"
    if ($driver -lt $minDriver) {
        Write-Warning "Driver $driver is older than the $minDriver minimum for CUDA $CudaVersion. Update your NVIDIA drivers or GPU acceleration will fail."
    } else {
        Write-Host "Driver $driver meets the minimum ($minDriver) for CUDA $CudaVersion."
    }
}

function Expand-Wheels {
    param([string]$SourceDir, [string]$DestBin)
    # `uv pip install --target` already unpacks each wheel; DLLs live under
    # nvidia/<pkg>/bin (or lib). Copy every DLL recursively into one flat dir.
    Get-ChildItem -LiteralPath $SourceDir -Recurse -Filter '*.dll' -File | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $DestBin -Force
    }
}

if ($Framework -eq 'pytorch') {
    Assert-DriverCompatible -Tool 'PyTorch'
    Write-Host ""
    Write-Host "PyTorch wheels bundle the CUDA $CudaVersion + cuDNN runtime inside torch/lib."
    Write-Host "Nothing to download. The driver check above is the only system requirement."
    if (-not $SkipSitecustomize) {
        Write-Host "sitecustomize.py is not needed for PyTorch - its runtime DLLs live inside the wheel, not on PATH."
    }
    $venvPy = Get-Python
    if ((Test-Path $venvPy) -and (Test-Path (Join-Path $root '.venv\Lib\site-packages\torch'))) {
        Write-Host "Verifying installed torch ..."
        & $venvPy -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
        if ($LASTEXITCODE -ne 0) { Write-Warning "torch import failed - run 'uv sync --extra voice' and retry." }
    } else {
        Write-Host ""
        Write-Host "Install the project-local voice stack with:"
        Write-Host "  uv sync --extra voice"
        Write-Host ""
        Write-Host "Then verify with:"
        $verifyCmd = '  uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''none'')"'
        Write-Host $verifyCmd
    }
    return
}

if ($Framework -eq 'fasterwhisper') {
    # faster-whisper (CTranslate2): the official ctranslate2 PyPI wheels bundle
    # GPU support AND cuDNN 9 (cudnn64_9.dll) inside the wheel. The only
    # missing Windows runtime pieces are the CUDA 12 cuBLAS + cudart DLLs,
    # which come from NVIDIA's own redistributable wheels (nvidia-cublas-cu12,
    # nvidia-cuda-runtime-cu12) in the 'stt' extra. sitecustomize.py adds their
    # site-packages/nvidia/*/bin (or lib) dirs to PATH at interpreter startup.
    # So, like PyTorch, nothing is downloaded here - just the driver check +
    # verify.
    $CudaVersion = '12.9'
    $CudnnVersion = 'bundled-in-ctranslate2'
    $minDriver = 570.86
    Assert-DriverCompatible -Tool 'faster-whisper (CTranslate2)'
    Write-Host ""
    Write-Host "CTranslate2 wheels bundle cuDNN 9; NVIDIA's cuBLAS/cudart 12 wheels"
    Write-Host "(nvidia-cublas-cu12, nvidia-cuda-runtime-cu12) ship via the 'stt' extra."
    Write-Host "Nothing to download. The driver check above is the only system requirement."
    $venvPy = Get-Python
    if ((Test-Path $venvPy) -and (Test-Path (Join-Path $root '.venv\Lib\site-packages\ctranslate2'))) {
        if (-not $SkipSitecustomize) {
            $siteDir = Join-Path $root '.venv\Lib\site-packages'
            Copy-Item -LiteralPath (Join-Path $root 'sitecustomize.py') -Destination (Join-Path $siteDir 'sitecustomize.py') -Force
            Write-Host "Installed sitecustomize.py into $siteDir (adds the NVIDIA wheel lib dirs to PATH)."
        }
        Write-Host "Verifying installed faster-whisper/CTranslate2 ..."
        & $venvPy -c "import ctranslate2; print('ctranslate2', ctranslate2.__version__, '| CUDA devices:', ctranslate2.get_cuda_device_count())"
        if ($LASTEXITCODE -ne 0) { Write-Warning "ctranslate2 import failed - run 'uv sync --extra stt' and retry." }
    } else {
        Write-Host ""
        Write-Host "Install the project-local STT stack with:"
        Write-Host "  uv sync --extra stt"
        Write-Host ""
        Write-Host "Then verify with:"
        Write-Host '  uv run python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"'
    }
    return
}

Assert-DriverCompatible -Tool 'TensorFlow'

$alreadyPopulated = (Test-Path $binDir) -and ((Get-ChildItem -LiteralPath $binDir -Filter '*.dll' -File | Measure-Object).Count -ge 8)
if ($alreadyPopulated) {
    Write-Host "cuda_runtime/bin already populated, nothing to do."
} else {
    New-Item -ItemType Directory -Path $TargetDir, $WorkDir -Force | Out-Null

    # --- CUDA runtime DLLs from NVIDIA's official PyPI redistributable wheels ---
    $wheelDir = Join-Path $WorkDir 'wheels'
    New-Item -ItemType Directory -Path $wheelDir, $binDir -Force | Out-Null
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uv) {
        throw "uv not found on PATH - required to download NVIDIA's redistributable wheels."
    }
    $wheelArgs = @('pip', 'install', '--target', $wheelDir, '--no-deps',
                   '--only-binary', ':all:') + $cu11Wheels
    Write-Host "Downloading NVIDIA CUDA runtime wheels (official PyPI redistributables) ..."
    & $uv.Source @wheelArgs
    if ($LASTEXITCODE -ne 0) { throw "uv pip install --target failed for NVIDIA cu11 wheels." }
    Expand-Wheels -SourceDir $wheelDir -DestBin $binDir

    # --- cuDNN (official NVIDIA redistributable; third-party builds crash TF) ---
    $cudnnArchive = Join-Path $WorkDir $cudnnFile
    if (-not (Test-Path $cudnnArchive)) {
        Write-Host "Downloading cuDNN $CudnnVersion (official NVIDIA build, ~665 MB) ..."
        Invoke-WebRequest -Uri $cudnnUrl -OutFile $cudnnArchive
    }
    $cudnnExtract = Join-Path $WorkDir 'cudnn'
    $cudnnDllDir = Join-Path $cudnnExtract 'cuda\bin'
    if (-not ((Test-Path $cudnnDllDir) -and (Get-ChildItem -LiteralPath $cudnnDllDir -Filter '*.dll' -File | Measure-Object).Count -gt 0)) {
        if (Test-Path $cudnnExtract) { Remove-Item -LiteralPath $cudnnExtract -Recurse -Force }
        New-Item -ItemType Directory -Path $cudnnExtract -Force | Out-Null
        Write-Host "Extracting cuDNN ..."
        Expand-Archive -LiteralPath $cudnnArchive -DestinationPath $cudnnExtract -Force
    }
    if (Test-Path $cudnnDllDir) {
        Get-ChildItem -LiteralPath $cudnnDllDir -Filter '*.dll' -File | ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $binDir -Force
        }
    }

    $count = (Get-ChildItem -LiteralPath $binDir -Filter '*.dll' -File | Measure-Object).Count
    Write-Host "Done. Copied $count DLLs into $binDir"
}

if (-not $SkipSitecustomize) {
    $siteDir = Join-Path $root '.venv\Lib\site-packages'
    $sitecustomize = Join-Path $root 'sitecustomize.py'
    if ((Test-Path $siteDir) -and (Test-Path $sitecustomize)) {
        Copy-Item -LiteralPath $sitecustomize -Destination (Join-Path $siteDir 'sitecustomize.py') -Force
        Write-Host "Installed sitecustomize.py into $siteDir"
    } else {
        Write-Warning "No venv at $siteDir - run 'uv sync' first, then re-run this script."
    }
}

Write-Host ""
Write-Host 'Verify with:'
Write-Host '  uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices(''GPU''))"'
