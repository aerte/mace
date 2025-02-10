# Arguments for slurm here


python mace/cli/run_train.py \
    --model="MACE" \
    --num_channels=16 \
    --max_L=0 \
    --r_max=4.0 \
    --name="mace01" \
    --model_dir="MACE_models" \
    --log_dir="MACE_models" \
    --checkpoints_dir="MACE_models" \
    --results_dir="MACE_models" \
    --train_file="data/solvent_xtb_train_200.xyz" \
    --valid_fraction=0.10 \
    --test_file="data/solvent_xtb_test.xyz" \
    --energy_key="energy_xtb" \
    --forces_key="forces_xtb" \
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


