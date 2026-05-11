@echo off
setlocal
cd /d C:\Users\cyl\Desktop\iTransformer-phasec-clean

echo [Stage] ETTh1-96 baseline itr=3
D:\Anaconda\Scripts\conda.exe run -n itr python -u run.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id etth196_validate_baseline_itr3 --model iTransformer --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --enc_in 7 --dec_in 7 --c_out 7 --des ETTh1ValidateBase --d_model 256 --d_ff 256 --n_heads 8 --factor 1 --embed timeF --itr 3 --batch_size 32 --learning_rate 0.0001 --train_epochs 10 --patience 3 --use_gpu True --gpu 0 --num_workers 0 --lradj type1
if errorlevel 1 exit /b %errorlevel%

echo [Stage] ETTh1-96 static anchor itr=3
D:\Anaconda\Scripts\conda.exe run -n itr python -u run.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id etth196_validate_static_anchor_itr3 --model iTransformer --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --enc_in 7 --dec_in 7 --c_out 7 --des ETTh1ValidateStaticAnchor --d_model 256 --d_ff 256 --n_heads 8 --factor 1 --embed timeF --itr 3 --batch_size 32 --learning_rate 0.0001 --train_epochs 10 --patience 3 --use_gpu True --gpu 0 --num_workers 0 --lradj type1 --graph_enable True --graph_mode static_causal_residual --graph_interface_dir C:\Users\cyl\Desktop\data\interfaces\ETTh1_graph_interface_cmiknn_ridgebase_sparse --graph_use_static_bias False --graph_use_dynamic_bias False --graph_use_lambda_gate False --graph_eval_use_static_bias False --graph_static_mix_mode softmax --graph_causal_pool_mode auto
if errorlevel 1 exit /b %errorlevel%

echo [Stage] ETTh1-96 static val/test backfill
D:\Anaconda\Scripts\conda.exe run -n itr python -u C:\Users\cyl\Desktop\data\backfill_etth1_staticcausal_preds.py
if errorlevel 1 exit /b %errorlevel%

echo [Stage] ETTh1-96 validation-calibrated dynamic correction
D:\Anaconda\Scripts\conda.exe run -n itr python -u C:\Users\cyl\Desktop\data\etth1_late_ramp_validation_grid.py
if errorlevel 1 exit /b %errorlevel%

echo [Done] ETTh1-96 validation pipeline completed
