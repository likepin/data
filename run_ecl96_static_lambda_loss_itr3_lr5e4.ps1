$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\cyl\Desktop\iTransformer-phasec-clean'
$logDir = Join-Path $repo 'logs\ecl96_static_lambda_loss_itr3_lr5e4_20260421'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Set-Location $repo

$common = @(
  '--is_training','1',
  '--root_path','C:\Users\cyl\Desktop\iTransformer-phasec-clean\dataset',
  '--data_path','ECL.csv',
  '--model','iTransformer',
  '--data','custom',
  '--features','M',
  '--target','MT_320',
  '--seq_len','96',
  '--label_len','48',
  '--pred_len','96',
  '--enc_in','321',
  '--dec_in','321',
  '--c_out','321',
  '--d_model','512',
  '--n_heads','8',
  '--e_layers','3',
  '--d_layers','1',
  '--d_ff','512',
  '--factor','1',
  '--embed','timeF',
  '--itr','3',
  '--batch_size','16',
  '--learning_rate','0.0005',
  '--train_epochs','10',
  '--patience','3',
  '--use_gpu','True',
  '--gpu','0',
  '--num_workers','0'
)

$graphCommon = @(
  '--graph_enable','True',
  '--graph_mode','static_causal_residual',
  '--graph_interface_dir','C:\Users\cyl\Desktop\data\interfaces\ECL_graph_interface_parcorr',
  '--graph_use_static_bias','False',
  '--graph_use_dynamic_bias','False',
  '--graph_use_lambda_gate','False',
  '--graph_eval_use_static_bias','False',
  '--graph_static_mix_mode','softmax',
  '--graph_causal_pool_mode','auto',
  '--graph_causal_pool_budget_mb','512',
  '--graph_support_topk','32',
  '--graph_pool_dim','64'
)

$lambdaLossCommon = @(
  '--graph_lambda_loss_weighting','True',
  '--graph_lambda_loss_alpha','1.0'
)

$experiments = @(
  @{
    name='01_static_lambda_loss_direct_itr3'
    args=@(
      '--model_id','ecl96_static_lambda_loss_direct_itr3_lr5e4',
      '--des','ECL96StaticLambdaLossDirectLR5e4',
      '--graph_lambda_loss_polarity','direct'
    ) + $graphCommon + $lambdaLossCommon
  },
  @{
    name='02_static_lambda_loss_inverse_itr3'
    args=@(
      '--model_id','ecl96_static_lambda_loss_inverse_itr3_lr5e4',
      '--des','ECL96StaticLambdaLossInverseLR5e4',
      '--graph_lambda_loss_polarity','inverse'
    ) + $graphCommon + $lambdaLossCommon
  }
)

$summary = Join-Path $logDir 'run_summary.log'
"started $(Get-Date -Format o)" | Set-Content -Path $summary
foreach ($exp in $experiments) {
  $log = Join-Path $logDir ($exp.name + '.log')
  "[$(Get-Date -Format o)] START $($exp.name)" | Tee-Object -FilePath $summary -Append
  & conda run -n itr python -u run.py @common @($exp.args) *> $log
  $code = $LASTEXITCODE
  "[$(Get-Date -Format o)] END $($exp.name) exit=$code" | Tee-Object -FilePath $summary -Append
  if ($code -ne 0) { exit $code }
}
"completed $(Get-Date -Format o)" | Tee-Object -FilePath $summary -Append
