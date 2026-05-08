$ErrorActionPreference = 'Stop'

$repoRoot = 'C:\Users\cyl\Desktop\iTransformer-phasec-clean'
$dataRoot = 'C:\Users\cyl\Desktop\data'
$resultsRoot = Join-Path $repoRoot 'results'
$logDir = Join-Path $dataRoot 'run_logs'
$statusPath = Join-Path $logDir 'solar192_pipeline.status.txt'
$pythonExe = 'C:\Users\cyl\.conda\envs\itr\python.exe'

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Write-Status {
    param([string]$Message)
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $line = "$timestamp`t$Message"
    $line | Tee-Object -FilePath $statusPath -Append
}

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$CommandArgs,
        [string]$LogPath,
        [string]$WorkingDirectory
    )
    Write-Status "START $Name"
    Push-Location $WorkingDirectory
    try {
        & $pythonExe @CommandArgs *> $LogPath
        $exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($exitCode -ne 0) {
        Write-Status "FAIL $Name exit=$exitCode"
        exit $exitCode
    }
    Write-Status "DONE $Name"
}

function Get-ProjectionCount {
    param([string]$Prefix)
    $dirs = Get-ChildItem $resultsRoot -Directory | Where-Object { $_.Name -like "$Prefix*projection_*" }
    return @($dirs).Count
}

function Get-BaseTrainArgs {
    param(
        [string]$ModelId,
        [string]$Des,
        [int]$ItrStart,
        [int]$ItrCount
    )
    return @(
        'run.py',
        '--is_training', '1',
        '--model_id', $ModelId,
        '--model', 'iTransformer',
        '--data', 'Solar',
        '--root_path', './dataset/Solar/',
        '--data_path', 'solar_AL.txt',
        '--features', 'M',
        '--target', 'OT',
        '--freq', 'h',
        '--seq_len', '96',
        '--label_len', '48',
        '--pred_len', '192',
        '--enc_in', '137',
        '--dec_in', '137',
        '--c_out', '137',
        '--d_model', '512',
        '--n_heads', '8',
        '--e_layers', '2',
        '--d_layers', '1',
        '--d_ff', '512',
        '--factor', '1',
        '--embed', 'timeF',
        '--learning_rate', '0.0005',
        '--batch_size', '32',
        '--train_epochs', '10',
        '--patience', '3',
        '--itr_start', "$ItrStart",
        '--itr', "$ItrCount",
        '--des', $Des,
        '--loss', 'MSE',
        '--lradj', 'type1',
        '--seed', '2023',
        '--use_gpu', 'true',
        '--gpu', '0'
    )
}

Write-Status 'PIPELINE solar192_static begin'

$baselinePrefix = 'solar_96_192_baseline_itr3'
$baselineDone = Get-ProjectionCount $baselinePrefix
if ($baselineDone -lt 3) {
    $baselineArgs = Get-BaseTrainArgs -ModelId $baselinePrefix -Des 'Exp_baseline_itr3' -ItrStart $baselineDone -ItrCount (3 - $baselineDone)
    Invoke-Step -Name "baseline_train projections=$baselineDone/3" -CommandArgs $baselineArgs -LogPath (Join-Path $logDir 'solar192_baseline_train.log') -WorkingDirectory $repoRoot
} else {
    Write-Status 'SKIP baseline_train projections=3/3'
}

$staticPrefix = 'solar_96_192_staticcausal_softmax_itr3'
$staticDone = Get-ProjectionCount $staticPrefix
if ($staticDone -lt 3) {
    $staticArgs = Get-BaseTrainArgs -ModelId $staticPrefix -Des 'Exp_staticcausal_softmax_itr3' -ItrStart $staticDone -ItrCount (3 - $staticDone)
    $staticArgs += @(
        '--graph_enable', 'true',
        '--graph_mode', 'static_causal_residual',
        '--graph_interface_dir', 'C:\Users\cyl\Desktop\data\interfaces\Solar_graph_interface_parcorr',
        '--graph_use_static_bias', 'false',
        '--graph_use_dynamic_bias', 'false',
        '--graph_use_lambda_gate', 'false',
        '--graph_eval_use_static_bias', 'false',
        '--graph_residual_alpha', '0.10',
        '--graph_residual_scale_mode', 'fixed',
        '--graph_static_mix_mode', 'softmax',
        '--graph_lambda_logit_bias', 'false',
        '--graph_causal_pool_mode', 'auto',
        '--graph_causal_pool_budget_mb', '512',
        '--graph_support_topk', '32',
        '--graph_support_topk_metric', 'abs_a_base',
        '--graph_pool_dim', '64'
    )
    Invoke-Step -Name "static_train projections=$staticDone/3" -CommandArgs $staticArgs -LogPath (Join-Path $logDir 'solar192_static_train.log') -WorkingDirectory $repoRoot
} else {
    Write-Status 'SKIP static_train projections=3/3'
}

Invoke-Step -Name 'backfill_val_baseline' -CommandArgs @(
    (Join-Path $dataRoot 'backfill_posthoc_profile_preds.py'),
    '--profile', 'solar192_static',
    '--variant', 'baseline',
    '--split', 'val'
) -LogPath (Join-Path $logDir 'solar192_backfill_val_baseline.log') -WorkingDirectory $dataRoot

Invoke-Step -Name 'backfill_val_static' -CommandArgs @(
    (Join-Path $dataRoot 'backfill_posthoc_profile_preds.py'),
    '--profile', 'solar192_static',
    '--variant', 'static',
    '--split', 'val'
) -LogPath (Join-Path $logDir 'solar192_backfill_val_static.log') -WorkingDirectory $dataRoot

Invoke-Step -Name 'lambda_sweep' -CommandArgs @(
    (Join-Path $dataRoot 'diagnose_real_lambda_feature_sweep.py'),
    '--profile', 'solar192_static',
    '--pred-len', '192'
) -LogPath (Join-Path $logDir 'solar192_lambda_sweep.log') -WorkingDirectory $dataRoot

Invoke-Step -Name 'closed_loop' -CommandArgs @(
    (Join-Path $dataRoot 'posthoc_selected_lambda_closed_loop.py'),
    '--profile', 'solar192_static',
    '--pred-len', '192',
    '--lambda-transform', 'rank',
    '--lambda-quality-guard',
    '--out-dir', (Join-Path $dataRoot 'deltaA_signal_audit\solar192_closed_loop_rank_quality_guard'),
    '--tag', 'rank_quality_guard'
) -LogPath (Join-Path $logDir 'solar192_closed_loop.log') -WorkingDirectory $dataRoot

Write-Status 'PIPELINE solar192_static done'
