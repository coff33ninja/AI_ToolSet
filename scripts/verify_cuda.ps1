<#
.SYNOPSIS
    Reports the CUDA/GPU status of this machine and a project-local runtime
    without installing anything.

.DESCRIPTION
    Answers "does my system have what this project needs?":
      * GPU name + driver version (nvidia-smi)
      * minimum driver required for the CUDA version your TensorFlow needs
      * whether cuda_runtime/bin is populated (DLL count)
      * optionally imports TensorFlow and prints detected GPU devices
      * if PyTorch is installed in the venv, reports torch.cuda.is_available()

.PARAMETER TensorFlowVersion
    TensorFlow major.minor the project targets (default 2.10). Used to print
    the recommended CUDA/cuDNN pair.

.PARAMETER RunTensorFlowCheck
    Also run the real TensorFlow GPU detection inside the venv.

.PARAMETER CheckTorch
    Also run torch.cuda.is_available() inside the venv (auto-runs when torch
    is detected; this forces it when detection is skipped).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/verify_cuda.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/verify_cuda.ps1 -RunTensorFlowCheck

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/verify_cuda.ps1 -CheckTorch
#>

[CmdletBinding()]
param(
    [string]$TensorFlowVersion = '2.10',
    [switch]$RunTensorFlowCheck,
    [switch]$CheckTorch
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$binDir = Join-Path $root 'cuda_runtime\bin'

$tfMatrix = @{
    '2.4'  = @{ Cuda = '11.0'; Cudnn = '8.0.5.39'; DriverMin = 451.82 }
    '2.5'  = @{ Cuda = '11.2'; Cudnn = '8.1.0.77'; DriverMin = 460.89 }
    '2.6'  = @{ Cuda = '11.2'; Cudnn = '8.1.0.77'; DriverMin = 460.89 }
    '2.7'  = @{ Cuda = '11.2'; Cudnn = '8.1.0.77'; DriverMin = 460.89 }
    '2.8'  = @{ Cuda = '11.2'; Cudnn = '8.1.0.77'; DriverMin = 460.89 }
    '2.9'  = @{ Cuda = '11.2'; Cudnn = '8.1.0.77'; DriverMin = 460.89 }
    '2.10' = @{ Cuda = '11.2'; Cudnn = '8.1.0.77'; DriverMin = 460.89 }
}

Write-Host "== AI ToolSet CUDA Status =="

if ($tfMatrix.ContainsKey($TensorFlowVersion)) {
    $e = $tfMatrix[$TensorFlowVersion]
    Write-Host "Target: TensorFlow $TensorFlowVersion needs CUDA $($e.Cuda) + cuDNN $($e.Cudnn), driver >= $($e.DriverMin)"
} else {
    Write-Host "Target: TensorFlow $TensorFlowVersion (no verified CUDA mapping; use scripts/get_cuda_runtime.ps1 -TensorFlowVersion $TensorFlowVersion to override)"
}

$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    $lines = & nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader 2>$null
    if ($lines) {
        foreach ($l in $lines) {
            $p = $l -split ','
            Write-Host ("GPU : {0}" -f $p[1].Trim())
            Write-Host ("Driver : {0} | VRAM: {1}" -f $p[0].Trim(), $p[2].Trim())
        }
        if ($tfMatrix.ContainsKey($TensorFlowVersion)) {
            $drv = [double]((($lines | Select-Object -First 1) -split ',')[0].Trim())
            $min = $tfMatrix[$TensorFlowVersion].DriverMin
            if ($drv -lt $min) {
                Write-Warning "Driver $drv is BELOW the $min minimum. Update NVIDIA drivers."
            } else {
                Write-Host "Driver meets the minimum for the target CUDA version. OK."
            }
        }
    } else {
        Write-Warning "nvidia-smi returned no GPU rows - check drivers/hardware."
    }
} else {
    Write-Warning "nvidia-smi not found. Install an NVIDIA driver; the driver package ships nvidia-smi."
}

$dllCount = if (Test-Path $binDir) { (Get-ChildItem -LiteralPath $binDir -Filter '*.dll' -File | Measure-Object).Count } else { 0 }
if ($dllCount -ge 8) {
    Write-Host "cuda_runtime/bin: populated ($dllCount DLLs). OK"
} else {
    Write-Warning "cuda_runtime/bin: only $dllCount DLLs found. Run scripts/get_cuda_runtime.ps1."
}

if ($RunTensorFlowCheck) {
    $venvPy = Join-Path $root '.venv\Scripts\python.exe'
    if (Test-Path $venvPy) {
        Write-Host "Running TensorFlow GPU detection ..."
        & $venvPy -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    } else {
        Write-Warning "No venv found. Run 'uv sync' first."
    }
}

$venvPy = Join-Path $root '.venv\Scripts\python.exe'
$torchPresent = (Test-Path (Join-Path $root '.venv\Lib\site-packages\torch'))
if ($torchPresent -or $CheckTorch) {
    if (Test-Path $venvPy) {
        Write-Host "Running PyTorch GPU detection ..."
        & $venvPy -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
        if (-not $torchPresent) {
            Write-Warning "torch not found in venv - run 'uv sync --extra voice' to install the project-local voice stack."
        }
    } else {
        Write-Warning "No venv found. Run 'uv sync --extra voice' first."
    }
} elseif (-not $torchPresent -and -not $CheckTorch) {
    Write-Host "PyTorch: not installed in venv (skip torch check; use scripts/get_cuda_runtime.ps1 -Framework pytorch)"
}
