$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$common = @(
    "--dataset", "gefcom",
    "--max_sources", "2",
    "--device", "cuda",
    "--batch_size", "128",
    "--num_workers", "8",
    "--prefetch_factor", "4",
    "--matmul_precision", "high",
    "--pretrain_epochs", "16",
    "--finetune_epochs", "14",
    "--safe_epochs", "14",
    "--nf_max_steps", "180",
    "--nf_val_check_steps", "10",
    "--nf_early_stop_patience_steps", "30",
    "--nf_batch_size", "64",
    "--nf_windows_batch_size", "512",
    "--nf_inference_windows_batch_size", "1024"
)

$methods = @(
    "persistence", "ridge", "lightgbm",
    "lstm", "gru", "safe_gru",
    "tcn", "safe_tcn",
    "patchtst", "safe_patchtst"
)

$benchmarkArgs = @(
    ".\safe_tcn_lab\run_benchmark.py"
) + $common + @(
    "--methods"
) + $methods + @(
    "--seeds", "42", "43", "44",
    "--baseline_method", "gru",
    "--primary_method", "safe_gru",
    "--benchmark_root", ".\safe_tcn_lab\final_runs\benchmark_gefcom"
)

$sweepArgs = @(
    ".\safe_tcn_lab\run_transfer_sweep.py"
) + $common + @(
    "--methods"
) + $methods + @(
    "--seeds", "42", "43", "44",
    "--baseline_method", "gru",
    "--primary_method", "safe_gru",
    "--target_train_days_list", "14", "30", "60",
    "--sweep_root", ".\safe_tcn_lab\final_runs\sweep_gefcom"
)

& python @benchmarkArgs
& python @sweepArgs
