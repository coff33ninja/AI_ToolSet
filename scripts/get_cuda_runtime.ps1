<#
.SYNOPSIS
    Provisions the CUDA/cuDNN runtime this project needs - TensorFlow into a
    project-local cuda_runtime/ folder, or (for PyTorch) verifies the driver
    only, since PyTorch wheels bundle CUDA/cuDNN inside themselves.

.DESCRIPTION
    TensorFlow on Windows requires matching CUDA and cuDNN runtime DLLs but
    does NOT require a system-wide CUDA toolkit install. This script resolves
    the exact build needed for your TensorFlow version, downloads it, and
    extracts just the DLLs into <repo>/cuda_runtime/bin.

    Nothing is guessed:
      * TensorFlow version -> CUDA/cuDNN mapping is a verified table.
      * The cudatoolkit download URL is resolved live from the conda-forge
        API (no hardcoded build hashes that can rot).
      * cuDNN always uses the official NVIDIA redistributable - conda-forge
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
    Which runtime to provision: 'tensorflow' (downloads CUDA/cuDNN DLLs) or
    'pytorch' (driver check only, torch bundles its own runtime).

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
    [ValidateSet('tensorflow', 'pytorch')]
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
    '2.4'  = @{ Cuda = '11.0'; CudaPkg = '11.0.3'; Cudnn = '8.0.5.39'; DriverMin = 451.82 }
    '2.5'  = @{ Cuda = '11.2'; CudaPkg = '11.2.2'; Cudnn = '8.1.0.77'; DriverMin = 460.89 }
    '2.6'  = @{ Cuda = '11.2'; CudaPkg = '11.2.2'; Cudnn = '8.1.0.77'; DriverMin = 460.89 }
    '2.7'  = @{ Cuda = '11.2'; CudaPkg = '11.2.2'; Cudnn = '8.1.0.77'; DriverMin = 460.89 }
    '2.8'  = @{ Cuda = '11.2'; CudaPkg = '11.2.2'; Cudnn = '8.1.0.77'; DriverMin = 460.89 }
    '2.9'  = @{ Cuda = '11.2'; CudaPkg = '11.2.2'; Cudnn = '8.1.0.77'; DriverMin = 460.89 }
    '2.10' = @{ Cuda = '11.2'; CudaPkg = '11.2.2'; Cudnn = '8.1.0.77'; DriverMin = 460.89 }
}

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
    $CudaPkgVersion = $entry.CudaPkg
    $minDriver = $entry.DriverMin
} else {
    $CudaPkgVersion = $CudaVersion
    $minDriver = 451.82
}

$cudnnMajorMinor = ($CudnnVersion -split '\.')[0..2] -join '.'
$cudnnUrl = "https://developer.download.nvidia.com/compute/redist/cudnn/v$cudnnMajorMinor/cudnn-$CudaVersion-windows-x64-v$CudnnVersion.zip"
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

function Get-CudaToolkitUrl {
    param([string]$Version)
    Write-Host "Resolving cudatoolkit $Version from conda-forge API ..."
    $api = Invoke-RestMethod -Uri 'https://api.anaconda.org/package/conda-forge/cudatoolkit' -Method Get
    $candidates = @($api.files | Where-Object {
        ($_.version -eq $Version -or $_.version -like "$Version.*") -and $_.basename -match '^win-64/.*\.tar\.bz2$'
    })
    if ($Version -eq '11.2.2') {
        $knownGood = @($candidates | Where-Object { $_.basename -eq 'win-64/cudatoolkit-11.2.2-h933977f_8.tar.bz2' })
        if ($knownGood) { $candidates = $knownGood }
    }
    if (-not $candidates) {
        throw "No win-64 cudatoolkit $Version found on conda-forge."
    }
    $url = $candidates[0].download_url
    if ($url -like '//*') { $url = 'https:' + $url }
    return $url, $candidates[0].basename
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

$alreadyPopulated = (Test-Path $binDir) -and ((Get-ChildItem -LiteralPath $binDir -Filter '*.dll' -File | Measure-Object).Count -ge 8)
if ($alreadyPopulated) {
    Write-Host "cuda_runtime/bin already populated, nothing to do."
} else {
    New-Item -ItemType Directory -Path $TargetDir, $WorkDir -Force | Out-Null

    $cudaUrl, $cudaFile = Get-CudaToolkitUrl -Version $CudaVersion
    $cudaArchive = Join-Path $WorkDir $cudaFile
    if (-not (Test-Path $cudaArchive)) {
        Write-Host "Downloading cudatoolkit $CudaVersion ..."
        Invoke-WebRequest -Uri $cudaUrl -OutFile $cudaArchive
    }
    $cudaExtract = Join-Path $WorkDir 'cudatoolkit'
    $cudaDllDir = Join-Path $cudaExtract 'Library\bin'
    if (-not ((Test-Path $cudaDllDir) -and (Get-ChildItem -LiteralPath $cudaDllDir -Filter '*.dll' -File | Measure-Object).Count -gt 0)) {
        if (Test-Path $cudaExtract) { Remove-Item -LiteralPath $cudaExtract -Recurse -Force }
        New-Item -ItemType Directory -Path $cudaExtract -Force | Out-Null
        Write-Host "Extracting cudatoolkit ..."
        & (Get-Python) -c "import sys, tarfile; tarfile.open(sys.argv[1], 'r:bz2').extractall(sys.argv[2])" $cudaArchive $cudaExtract
        if ($LASTEXITCODE -ne 0) { throw "Failed to extract cudatoolkit" }
    }

    $cudnnArchive = Join-Path $WorkDir $cudnnFile
    if (-not (Test-Path $cudnnArchive)) {
        Write-Host "Downloading cuDNN $CudnnVersion (official NVIDIA build, ~665 MB) ..."
        Invoke-WebRequest -Uri $cudnnUrl -OutFile $cudnnArchive
    }
    $cudnnExtract = Join-Path $WorkDir 'cudnn'
    $cudnnDllDir = Join-Path $cudnnExtract 'cuda\bin'
    if (-not ((Test-Path $cudnnDllDir) -and (Get-ChildItem -LiteralPath $cudnnDllDir -Filter '*.dll' -File | Measure-Object).Count -gt 0)) {
        if (Test-Path $cudnnExtract) { Remove-Item -LiteralPath $cudnnExtract -Recurse -Force }
        Write-Host "Extracting cuDNN ..."
        Expand-Archive -LiteralPath $cudnnArchive -DestinationPath $cudnnExtract -Force
    }

    New-Item -ItemType Directory -Path $binDir -Force | Out-Null
    foreach ($srcDir in @($cudaDllDir, $cudnnDllDir)) {
        if (Test-Path $srcDir) {
            Get-ChildItem -LiteralPath $srcDir -Filter '*.dll' -File | ForEach-Object {
                Copy-Item -LiteralPath $_.FullName -Destination $binDir -Force
            }
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
Write-Host "Verify with:"
Write-Host "  uv run python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\""
