#!/bin/bash

SEEDS=( '20220118' 
        '2022011811' '2022011822' '2022011833' '2022011844'
      )

# for seed in ${SEEDS[@]}; do
#     python train.py --config=configs/toy_mog2/compare/mis50/vae_z5_encm_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
#     python train.py --config=configs/toy_mog2/compare/mis50/vae_z15_encm_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
#     python train.py --config=configs/toy_mog2/compare/mis50/vae_z25_encm_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
# done

# for seed in ${SEEDS[@]}; do
#     python train.py --config=configs/toy_mog2/compare/mis50/iwae_i5_encm_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
#     python train.py --config=configs/toy_mog2/compare/mis50/iwae_i15_encm_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
#     python train.py --config=configs/toy_mog2/compare/mis50/iwae_i25_encm_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
# done

# for seed in ${SEEDS[@]}; do
#     python train.py --config=configs/toy_mog2/compare/mis50/mvb2vae_lairdmisr0_k5_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
#     python train.py --config=configs/toy_mog2/compare/mis50/mvb2vae_lairdmisr0_k15_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
#     python train.py --config=configs/toy_mog2/compare/mis50/mvb2vae_lairdmisr0_k25_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
# done

for seed in ${SEEDS[@]}; do
    python train.py --config=configs/toy_mog2/compare/mis50/cvivae_lairdmisr0_k5_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
    python train.py --config=configs/toy_mog2/compare/mis50/cvivae_lairdmisr0_k15_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
    python train.py --config=configs/toy_mog2/compare/mis50/cvivae_lairdmisr0_k25_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
done

# for seed in ${SEEDS[@]}; do
#     python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decall_encissmis_decisdmmis_lairdmisr0_i1_k5_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed

#     python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decall_encissmis_decisdmmis_lairdmisr0_i1_k15_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
#     python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decall_encissmis_decisdmmis_lairdmisr0_i5_k3_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
#     python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decall_encissmis_decisdmmis_lairdmisr0_i3_k5_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed

#     python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decall_encissmis_decisdmmis_lairdmisr0_i1_k25_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
#     python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decall_encissmis_decisdmmis_lairdmisr0_i5_k5_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed

#     # # # LAIR sampler use more importance samples
#     # # # python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decall_encissmis_decisdmmis_lairdmisr0i5_i5_k3_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
#     # python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decall_encissmis_decisdmmis_lairdmisr0i3_i3_k5_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
#     # python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decall_encissmis_decisdmmis_lairdmisr0i5_i5_k5_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed

#     # # Enc-IS-DMMIS
#     # # python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decall_encisdmmis_decisdmmis_lairdmisr0_i1_k5_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
#     # python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decall_encisdmmis_decisdmmis_lairdmisr0_i3_k5_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
#     # python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decall_encisdmmis_decisdmmis_lairdmisr0_i5_k5_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed

# done

# for seed in ${SEEDS[@]}; do
#     python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decindep_encissmis_decisdmmis_lairdmisr0_i1_k5_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed

#     python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decindep_encissmis_decisdmmis_lairdmisr0_i1_k15_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
#     python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decindep_encissmis_decisdmmis_lairdmisr0_i5_k3_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
#     python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decindep_encissmis_decisdmmis_lairdmisr0_i3_k5_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed

#     python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decindep_encissmis_decisdmmis_lairdmisr0_i1_k25_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
#     python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decindep_encissmis_decisdmmis_lairdmisr0_i5_k5_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
# done


# for seed in ${SEEDS[@]}; do
#     python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decall_encissmis_decissmis_lairdmisr0_i1_k5_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed

#     # python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decall_encissmis_decissmis_lairdmisr0_i1_k15_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
#     # python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decall_encissmis_decissmis_lairdmisr0_i5_k3_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
#     # python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decall_encissmis_decissmis_lairdmisr0_i3_k5_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed

#     # python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decall_encissmis_decissmis_lairdmisr0_i1_k25_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
#     # python train.py --config=configs/toy_mog2/compare/mis50/mvb2iwae_encindep_decall_encissmis_decissmis_lairdmisr0_i5_k5_stl_mlp.yaml --trainer.gpus=1 --seed_everything=$seed --data.setup_seed=$seed
# done
