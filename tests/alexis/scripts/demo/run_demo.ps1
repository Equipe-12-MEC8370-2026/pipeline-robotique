<#
.SYNOPSIS
  CTA demo: a trained policy (ACT P0 by default, or SmolVLA P1) picks and places the cube live in Isaac Sim, in one command.

.DESCRIPTION
  1. Starts the policy server (.venv-learn; ACT replanning every 25 steps, or SmolVLA every 50) in the background.
  2. Runs the simulation with its window (isaacsim-env) on a few trials, then stops the server.
  Each trial's front|wrist video and a report.json go to outputs\demo\<tag>\.
  Backup if the live run fails: outputs\demo\montage_v1.mp4.

.EXAMPLE
  # 5 new random positions (a new seed each time)
  powershell -ExecutionPolicy Bypass -File scripts\demo\run_demo.ps1
.EXAMPLE
  # replay chosen trials of the evaluation (seed 7): 0-39 in range, 40-49 outside the training range
  powershell -ExecutionPolicy Bypass -File scripts\demo\run_demo.ps1 -Seed 7 -Trials "0,14,47"
.EXAMPLE
  # rehearsal without a window
  powershell -ExecutionPolicy Bypass -File scripts\demo\run_demo.ps1 -Headless -Count 2
.EXAMPLE
  # same demo with the SmolVLA policy (P1: replans every 50 steps)
  powershell -ExecutionPolicy Bypass -File scripts\demo\run_demo.ps1 -Policy smolvla -Seed 7 -Trials "0,41,44"
#>
param(
    [int]$Count = 5,                 # number of trials (ignored with -Trials)
    [int]$Ood = 1,                   # how many of them outside the training range
    [int]$Seed = -1,                 # -1 = random
    [string]$Trials = "",            # explicit trial ids, e.g. "0,14,47" (with -Seed 7: the evaluated ones)
    [ValidateSet("act", "smolvla")][string]$Policy = "act",  # preset: checkpoint + replanning of the evaluated model
    [int]$NActionSteps = 0,          # 0 = the preset's value (ACT 25, SmolVLA 50)
    [string]$Ckpt = "",              # "" = the preset's checkpoint
    [switch]$Headless
)

$presets = @{
    act     = @{ ckpt = "outputs/train/act_p0/checkpoints/last/pretrained_model"; steps = 25; name = "ACT (P0)" }
    smolvla = @{ ckpt = "outputs/train/smolvla_p1/checkpoints/last/pretrained_model"; steps = 50; name = "SmolVLA (P1)" }
}
if (-not $Ckpt) { $Ckpt = $presets[$Policy].ckpt }
if ($NActionSteps -le 0) { $NActionSteps = $presets[$Policy].steps }
$env:HF_HUB_OFFLINE = "1"            # everything needed is local (SmolVLA tokenizer from the HF cache)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Repo
$LearnPy = Join-Path $Repo ".venv-learn\Scripts\python.exe"
$SimPy = Join-Path $Repo "..\isaacsim-env\Scripts\python.exe"
foreach ($p in @($LearnPy, $SimPy, (Join-Path $Ckpt "model.safetensors"))) {
    if (-not (Test-Path $p)) { throw "missing: $p" }
}

# the server port must be free (a server left over from an earlier run would serve the wrong model)
$busy = Get-NetTCPConnection -LocalPort 50765 -State Listen -ErrorAction SilentlyContinue
if ($busy) { throw "port 50765 already in use (PID $($busy.OwningProcess)): stop that policy server first" }

if ($Seed -lt 0) { $Seed = Get-Random -Minimum 100 -Maximum 100000 }
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$tag = "live_$stamp"
$outRoot = Join-Path $Repo "outputs\demo"
New-Item -ItemType Directory -Force $outRoot | Out-Null
$srvOut = Join-Path $outRoot "server_$stamp.out.txt"
$srvErr = Join-Path $outRoot "server_$stamp.err.txt"

Write-Host "=== CTA demo: $($presets[$Policy].name) policy, SO-101, Isaac Sim ===" -ForegroundColor Cyan
Write-Host "[demo] starting the policy server ($Ckpt, n_action_steps $NActionSteps)..."
$server = Start-Process -FilePath $LearnPy -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $srvOut -RedirectStandardError $srvErr `
    -ArgumentList @("scripts\eval\policy_server.py", "--ckpt", $Ckpt, "--n_action_steps", "$NActionSteps")

try {
    $deadline = (Get-Date).AddSeconds(180)
    while (-not ((Test-Path $srvOut) -and (Select-String -Path $srvOut -Pattern "READY" -Quiet))) {
        if ($server.HasExited) { throw "policy server exited (see $srvErr)" }
        if ((Get-Date) -gt $deadline) { throw "policy server not ready after 180 s (see $srvErr)" }
        Start-Sleep -Seconds 2
    }
    Write-Host "[demo] policy server ready." -ForegroundColor Green

    $simArgs = @("scripts\sim\eval_policy.py", "--enable_cameras", "--agent", "policy", "--tag", $tag,
                 "--out", "outputs/demo", "--seed", "$Seed", "--keep_success_videos", "100", "--shutdown_server")
    $ids = @($Trials -split "[,; ]+" | Where-Object { $_ -ne "" })
    if ($ids | Where-Object { $_ -notmatch "^\d+$" }) { throw "-Trials must be numbers separated by commas, got '$Trials'" }
    if ($ids.Count -gt 0) {
        $simArgs += @("--trials") + $ids
        $simArgs += @("--n_train", "40", "--n_ood", "10")
        Write-Host "[demo] seed $Seed, trials $($ids -join ', ')"
    } else {
        $simArgs += @("--n_train", "$($Count - $Ood)", "--n_ood", "$Ood")
        Write-Host "[demo] seed $Seed, $Count trials ($Ood outside the training range)"
    }
    if ($Headless) { $simArgs += "--headless" } else { $simArgs += @("--viz", "kit") }

    $env:PYTHONPATH = Join-Path $Repo "src"
    Write-Host "[demo] launching Isaac Sim (about 1 min to load)..."
    $ErrorActionPreference = "Continue"  # Isaac Sim writes warnings to stderr (PowerShell 5.1 would stop)
    $simLog = Join-Path $outRoot "sim_$stamp.txt"
    & $SimPy @simArgs 2>&1 | ForEach-Object {
        $line = "$_"
        Add-Content -Path $simLog -Value $line -Encoding utf8
        if ($line -match "^\[trial") {
            $color = if ($line -match "\] success") { "Green" } else { "Red" }
            Write-Host $line -ForegroundColor $color
        } elseif ($line -match "^\[eval\]") { Write-Host $line }
    }

    $report = Join-Path $outRoot "$tag\report.json"
    if (Test-Path $report) {
        $s = (Get-Content $report -Raw | ConvertFrom-Json).summary.all
        Write-Host ("=== result: {0}/{1} successes - videos and report in outputs\demo\{2} ===" -f $s.successes, $s.n, $tag) -ForegroundColor Cyan
    } else {
        Write-Host "[demo] no report (window closed early, or an error). Last lines of $simLog :" -ForegroundColor Yellow
        Get-Content $simLog -Tail 15 | ForEach-Object { Write-Host "    $_" }
    }
}
finally {
    if (-not $server.HasExited) {
        Start-Sleep -Seconds 2
        if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force; Write-Host "[demo] policy server stopped." }
    }
}
