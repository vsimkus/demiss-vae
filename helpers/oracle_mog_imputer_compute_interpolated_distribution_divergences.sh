#!/bin/bash

SEEDS=( '20220118' 
      #  '2022011811' '2022011822' '2022011833' '2022011844'
      )

# CONFIGS=('interp000' 'interp025' 'interp050' 'interp075' 'interp100')
# CONFIGS=('interp000')
CONFIGS=('interp025' 'interp050' 'interp075' 'interp100')

# oracle_exp_subdir='oracle_imp_with_interpolate_control_study_indepmargmog'
# oracle_exp_subdir='oracle_imp_with_interpolate_control_study_marggauss'
# oracle_exp_subdir='oracle_imp_with_interpolate_control_study_indepmarggauss'
# oracle_exp_subdir='oracle_imp_with_interpolate_control_study_wassindepmargmog'
# oracle_exp_subdir='oracle_imp_with_interpolate_control_study_wassindepmarggauss'
# oracle_exp_subdir='oracle_imp_with_interpolate_control_study_indepmargmog_to_indepmarggauss'
# oracle_exp_subdir='oracle_imp_with_interpolate_control_study_wass_indepmargmog_to_indepmarggauss'
# oracle_exp_subdir='oracle_imp_with_interpolate_control_study_conditional_mog_with_mode_collapse_to_maxprob_mode'
# oracle_exp_subdir='oracle_imp_with_interpolate_control_study_conditional_mog_with_mode_collapse_to_firstnonzero_mode'
# oracle_exp_subdir='oracle_imp_with_interpolate_control_study_conditional_mog_with_mode_collapse_to_secondlargestprob_mode'
# oracle_exp_subdir='oracle_imp_with_interpolate_control_study_conditional_indep_mog_with_var_shrinkage_to_001var'
# oracle_exp_subdir='oracle_imp_with_interpolate_control_study_conditional_mog_with_equal_component_probabilities'
oracle_exp_subdir='oracle_imp_with_interpolate_control_study_conditional_indep_mog_created_from_true_indep_mog'

for seed in "${SEEDS[@]}"
do
    for config in "${CONFIGS[@]}"
    do
        echo "Running $config with seed $seed."
        python3 helpers/oracle_mog_imputer_compute_interpolated_distribution_divergences.py --seed_everything=$seed --data.setup_seed=$seed --config ./configs/toy_mog2_large/${oracle_exp_subdir}/${config}.yaml
    done
done

oracle_exp_subdir='oracle_imp_with_interpolate_control_study_conditional_indep_mog'

for seed in "${SEEDS[@]}"
do
    for config in "${CONFIGS[@]}"
    do
        echo "Running $config with seed $seed."
        python3 helpers/oracle_mog_imputer_compute_interpolated_distribution_divergences.py --seed_everything=$seed --data.setup_seed=$seed --config ./configs/toy_mog2_large/${oracle_exp_subdir}/${config}.yaml
    done
done
