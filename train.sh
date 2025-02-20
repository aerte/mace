# Arguments for slurm here


python mace/cli/run_train.py \
    --model="MACE" \
    --num_channels=16 \
    --max_L=0 \
    --r_max=4.0 \
    --name="mace01" \
    --E0s='average' \
    --model_dir="MACE_models" \
    --log_dir="MACE_models" \
    --checkpoints_dir="MACE_models" \
    --results_dir="MACE_models" \
    --train_file="/Users/faerte/Desktop/mace/data/train_ethanol.xyz" \
    --valid_fraction=0.10 \
    --test_file="/Users/faerte/Desktop/mace/data/test_ethanol.xyz" \
    --energy_key="energy" \
    --forces_key="forces" \
    --device="cpu" \
    --batch_size=2 \
    --max_num_epochs=100 \
    --swa \
    --start_swa=60 \
    --ema \
    --ema_decay=0.99 \
    --seed=123 \
    --wandb \
    --wandb_entity="aertebjerg-felix" \
    --wandb_project="Thesis" \
    --wandb_name="test"

python mace/cli/run_train.py \
     --name="MACE_ethanol" \
    --train_file="/Users/faerte/Desktop/mace/data/train_ethanol.xyz" \
    --valid_fraction=0.05 \
    --test_file="/Users/faerte/Desktop/mace/data/test_ethanol.xyz" \
    --energy_weight=8.0 \
    --forces_weight=1000.0 \
    --config_type_weights='{"Default":1.0}' \
    --E0s='average' \
    --model="ScaleShiftMACE" \
    --interaction_first="RealAgnosticResidualInteractionBlock" \
    --interaction="RealAgnosticResidualInteractionBlock" \
    --num_interactions=2 \
    --error_table="TotalMAE" \
    --max_ell=3 \
    --hidden_irreps='256x0e + 256x1o + 256x2e' \
    --num_cutoff_basis=5 \
    --correlation=3 \
    --r_max=6.0 \
    --scaling='rms_forces_scaling' \
    --batch_size=5 \
    --max_num_epochs=3500 \
    --lr=0.01 \
    --patience=200 \
    --swa_forces_weight=1000.0 \
    --swa_energy_weight=8.0 \
    --swa_lr=0.001 \
    --weight_decay=5e-7 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --default_dtype="float32"\
    --swa \
    --start_swa=2000 \
    --clip_grad=10 \
    --device=cpu \
    --seed=3 \
    --distributed=True \
    --wandb \
    --wandb_entity="aertebjerg-felix" \
    --wandb_project="Thesis" \
    --wandb_name="test" \
    --hpo \
    --hpo_type='grid' \
    --hpo_grid='[2, 2.5, 3, 3.5, 4, 4.5, 5]'


