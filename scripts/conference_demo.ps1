<#
.SYNOPSIS
    Fast SmartGC conference demo: prints the final measured results.

.DESCRIPTION
    Reads only the CSVs and JSON that a completed pipeline run already wrote.
    It trains nothing, simulates nothing and computes no result of its own, so
    it finishes in about a second and cannot fail on stage.

    Every number printed is read from a file in results/. If a file is missing
    the script says which command produces it rather than inventing a value.

    For the live version that re-runs the three placement policies in front of
    the audience, use scripts/conference_demo.py instead.

.EXAMPLE
    pwsh scripts/conference_demo.ps1
    pwsh scripts/conference_demo.ps1 -Target msr_src2_0 -Utilization 85
#>

[CmdletBinding()]
param(
    [string]$Target = "",
    [double]$Utilization = 80
)

$ErrorActionPreference = "Stop"

# Format numbers the same way on every machine. The host culture here groups
# digits Indian-style (8,13,24,630 for 81,324,630), which would make the printed
# figures disagree with the CSVs they came from.
[System.Threading.Thread]::CurrentThread.CurrentCulture =
    [System.Globalization.CultureInfo]::GetCultureInfo("en-US")

$repo = Split-Path -Parent $PSScriptRoot
$rule = "=" * 74

function Write-Section([string]$Title) {
    Write-Host ""
    Write-Host $rule
    Write-Host "  $Title"
    Write-Host $rule
}

function Require-File([string]$RelativePath, [string]$HowToMake) {
    $full = Join-Path $repo $RelativePath
    if (-not (Test-Path $full)) {
        Write-Host ""
        Write-Host "MISSING: $RelativePath" -ForegroundColor Yellow
        Write-Host "  Produce it with: $HowToMake"
        exit 2
    }
    return $full
}

$transferPath = Require-File "results/final/transfer_learning_comparison.csv" `
    "python experiments/run_all.py --stages evaluate"
$ablationPath = Require-File "results/final/ablation.csv" `
    "python experiments/run_all.py --stages simulate analyze"
$manifestPath = Require-File "results/final/experiment_manifest.json" `
    "python experiments/run_all.py"

$transfer = Import-Csv $transferPath
$ablation = Import-Csv $ablationPath
$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json

# Default to the first held-out target the frozen role assignment produced.
if (-not $Target) {
    $Target = $manifest.workload_roles.targets[0]
}
$targetRows = $transfer | Where-Object { $_.target_trace -eq $Target }
if (-not $targetRows) {
    Write-Host "No results for target '$Target'." -ForegroundColor Yellow
    Write-Host ("Available: " + (($transfer.target_trace | Sort-Object -Unique) -join ", "))
    exit 2
}

$datasetName = switch -Wildcard ($Target) {
    "msr_*"    { "MSR Cambridge (Microsoft Research, 2007)" }
    "systor_*" { "SYSTOR '17 enterprise VDI (Fujitsu, 2016)" }
    default    { $Target }
}

$finetuned = $targetRows | Where-Object { $_.model_type -eq "pretrained_finetuned" }

Write-Section "SMARTGC CONFERENCE DEMO"
Write-Host "  Dataset            : $datasetName"
Write-Host "  Target trace       : $Target"
Write-Host "  Model              : LSTM, sequence length $($finetuned.sequence_length), $($finetuned.parameter_count) parameters"
Write-Host "  Pretraining        : Yes - $($finetuned.pretrain_trace_count) MSR workloads, target held out"
Write-Host "  Fine-tuning        : Yes - $($finetuned.finetune_portion)"
Write-Host "  Test split         : $($finetuned.test_portion)"
Write-Host "  Test samples       : $('{0:N0}' -f [int]$finetuned.test_samples)"
Write-Host "  Seed               : $($manifest.seed)"
Write-Host "  Git commit         : $($manifest.environment.git_commit)"

Write-Section "PREDICTION - rewrite interval, held-out test split"
$fmt = "  {0,-40} {1,16} {2,17} {3,10} {4,8}"
Write-Host ($fmt -f "Approach", "MAE (us)", "RMSE (us)", "MAE log1p", "F1")
Write-Host ("  " + ("-" * 94))

$bestBaseline = $targetRows |
    Where-Object { $_.family -eq "baseline" } |
    Sort-Object { [double]$_.test_mae_log1p } |
    Select-Object -First 1
if ($bestBaseline) {
    Write-Host ($fmt -f "Best baseline ($($bestBaseline.model_type))",
        ('{0:N0}' -f [double]$bestBaseline.test_mae_us),
        ('{0:N0}' -f [double]$bestBaseline.test_rmse_us),
        ('{0:N4}' -f [double]$bestBaseline.test_mae_log1p),
        ('{0:N4}' -f [double]$bestBaseline.f1))
}
foreach ($stage in @("scratch", "pretrained", "pretrained_finetuned")) {
    $row = $targetRows | Where-Object { $_.model_type -eq $stage }
    if ($row) {
        $label = switch ($stage) {
            "scratch"              { "LSTM from scratch" }
            "pretrained"           { "Pretrained (no fine-tuning)" }
            "pretrained_finetuned" { "Pretrained + fine-tuned" }
        }
        Write-Host ($fmt -f $label,
            ('{0:N0}' -f [double]$row.test_mae_us),
            ('{0:N0}' -f [double]$row.test_rmse_us),
            ('{0:N4}' -f [double]$row.test_mae_log1p),
            ('{0:N4}' -f [double]$row.f1))
    }
}

Write-Section "LOG-STRUCTURED FILE SYSTEM - write amplification at $Utilization% utilization"
$runs = $ablation | Where-Object {
    $_.trace -eq $Target -and [math]::Round([double]$_.utilization_pct) -eq [math]::Round($Utilization)
}
if (-not $runs) {
    $available = ($ablation | Where-Object { $_.trace -eq $Target } |
        ForEach-Object { [math]::Round([double]$_.utilization_pct) } | Sort-Object -Unique) -join "%, "
    Write-Host "  No simulation at $Utilization% for $Target. Available: $available%"
} else {
    $sfmt = "  {0,-30} {1,8} {2,10} {3,14} {4,10} {5,16}"
    Write-Host ($sfmt -f "Policy", "WAF", "vs MIXED", "Migrations", "GC count", "GC bytes")
    Write-Host ("  " + ("-" * 92))
    $order = @("MIXED", "RULE_BASED", "LSTM scratch", "LSTM pretrained", "LSTM pretrained+finetuned")
    foreach ($name in $order) {
        $row = $runs | Where-Object { $_.configuration -eq $name }
        if ($row) {
            $change = if ($name -eq "MIXED") { "baseline" }
                      else { '{0:+0.00;-0.00}%' -f [double]$row.waf_vs_mixed_pct }
            Write-Host ($sfmt -f $name,
                ('{0:N4}' -f [double]$row.waf),
                $change,
                ('{0:N0}' -f [double]$row.valid_blocks_migrated),
                ('{0:N0}' -f [double]$row.gc_count),
                ('{0:N0}' -f [double]$row.gc_bytes_copied))
        }
    }
    $identical = @($runs | ForEach-Object { $_.total_write_requests } | Sort-Object -Unique)
    Write-Host ""
    Write-Host ("  Identical workload for every policy: {0:N0} write requests, {1} segments" -f `
        [int64]$identical[0], $runs[0].total_segments)
    if ($identical.Count -ne 1) {
        Write-Host "  WARNING: policies saw different workloads" -ForegroundColor Red
    }
}

Write-Section "FIGURES"
$plots = @(
    @{ File = "results/plots/final_waf_comparison.png";       What = "WAF by placement policy" },
    @{ File = "results/plots/final_gc_migrations.png";        What = "Valid-block migrations" },
    @{ File = "results/plots/final_prediction_comparison.png"; What = "Prediction error vs baselines" },
    @{ File = "results/plots/final_transfer_learning.png";    What = "Scratch vs pretrained vs fine-tuned" }
)
foreach ($plot in $plots) {
    $mark = if (Test-Path (Join-Path $repo $plot.File)) { "ok " } else { "-- " }
    Write-Host ("  [{0}] {1,-42} {2}" -f $mark, $plot.File, $plot.What)
}

Write-Host ""
Write-Host "  Tables: results/final/  |  Methodology: docs/methodology.md"
Write-Host "  Live version that re-runs the simulator: python scripts/conference_demo.py"
Write-Host $rule
