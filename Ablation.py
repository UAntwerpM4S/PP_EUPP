import os

#check added value of m_mulp (as suggested by reviewers)

base_command = "python3 Train.py --loss {loss} --ens-num 11 --target-var {target_var} --lr 0.001 --epochs 50 --batch-size 2 --nheads {nheads} --num_blocks {num_blocks} --projection_channels {projection_channels} --mlp_mult {mlp_mult} --num_predictors={num_predictors}"

configs = [
    (8, 4, 64, 1),
    (8, 4, 64, 2),
    (8, 4, 64, 4),
    (8, 4, 64, 8),
    (8, 5, 128, 4),
    (16, 4, 128, 4),
]


target_vars = {
    "t2m": ("CRPS", 12),
    "w10": ("CRPSTRUNC", 15),
    "w100": ("CRPSTRUNC", 15),
}


for target_var, (loss, num_predictors) in target_vars.items():
    for nheads, num_blocks, projection_channels, mlp_mult in configs:
        command = base_command.format(
            loss=loss,
            target_var=target_var,
            nheads=nheads,
            num_blocks=num_blocks,
            projection_channels=projection_channels,
            mlp_mult=mlp_mult,
            num_predictors=num_predictors
        )
        print(f"Executing: {command}")
        os.system(command)


