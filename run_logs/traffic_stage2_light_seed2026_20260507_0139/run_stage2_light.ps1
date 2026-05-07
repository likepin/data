$ErrorActionPreference = 'Stop'
Set-Location 'C:\Users\cyl\Desktop\iTransformer-phasec-clean'
$env:CUDA_VISIBLE_DEVICES='0'
$conda='D:\Anaconda\Scripts\conda.exe'
$baselineArgs=@('run','--no-capture-output','-n','itr','python','-u','run.py','--is_training','1','--root_path','./dataset/traffic/','--data_path','traffic.csv','--model','iTransformer','--data','custom','--features','M','--seq_len','96','--pred_len','96','--e_layers','4','--enc_in','862','--dec_in','862','--c_out','862','--d_model','512','--d_ff','512','--batch_size','16','--learning_rate','0.001','--itr','1','--itr_start','3','--seed','2026','--use_gpu','True','--gpu','0','--num_workers','0','--train_epochs','10','--patience','3','--model_id','traffic_96_96_baseline_itr3','--des','Exp_baseline_itr3')
$staticArgs=@('run','--no-capture-output','-n','itr','python','-u','run.py','--is_training','1','--root_path','./dataset/traffic/','--data_path','traffic.csv','--model','iTransformer','--data','custom','--features','M','--seq_len','96','--pred_len','96','--e_layers','4','--enc_in','862','--dec_in','862','--c_out','862','--d_model','512','--d_ff','512','--batch_size','16','--learning_rate','0.001','--itr','1','--itr_start','3','--seed','2026','--use_gpu','True','--gpu','0','--num_workers','0','--train_epochs','10','--patience','3','--model_id','traffic_96_96_staticcausal_softmax_itr3','--des','Exp_staticcausal_softmax_itr3','--graph_enable','True','--graph_mode','static_causal_residual','--graph_interface_dir','C:\Users\cyl\Desktop\data\interfaces\Traffic_graph_interface_parcorr','--graph_use_static_bias','False','--graph_use_dynamic_bias','False','--graph_use_lambda_gate','False','--graph_eval_use_static_bias','False','--graph_static_mix_mode','softmax','--graph_causal_pool_mode','auto','--graph_causal_pool_budget_mb','512')
$logRoot='C:\Users\cyl\Desktop\data\run_logs\traffic_stage2_light_seed2026_20260507_0139'
"START baseline seed=2026 $(Get-Date -Format o)" | Tee-Object -FilePath (Join-Path $logRoot 'stage2_status.log') -Append
& $conda @baselineArgs *> (Join-Path $logRoot 'baseline_projection3.log')
$baseExit=$LASTEXITCODE
"END baseline exit=$baseExit $(Get-Date -Format o)" | Tee-Object -FilePath (Join-Path $logRoot 'stage2_status.log') -Append
if ($baseExit -ne 0) { exit $baseExit }
"START staticcausal seed=2026 $(Get-Date -Format o)" | Tee-Object -FilePath (Join-Path $logRoot 'stage2_status.log') -Append
& $conda @staticArgs *> (Join-Path $logRoot 'staticcausal_projection3.log')
$staticExit=$LASTEXITCODE
"END staticcausal exit=$staticExit $(Get-Date -Format o)" | Tee-Object -FilePath (Join-Path $logRoot 'stage2_status.log') -Append
exit $staticExit
