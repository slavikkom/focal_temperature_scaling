#!/bin/bash


function define_seeds() {
    local -n seeds_ref=$1
    seeds_ref+=(42 123 2023) # 7 99)
    # seeds_ref+=(42)
}

function build_configs() {
    local -n configs_ref=$1
    local -a labelsmoothing=(0.05 0.1 0.15)
    local -a gammas=(0.25 0.5 0.75 1 1.5 2 3 5 7) 
    # local -a gammas=(2 3 5) 
    local -a betas=(0.25 0.5 0.75 1 1.5 2 3 5 7)
    # local -a gammas=(1) 
    # local -a betas=(1)

    # Random loss
    # configs_ref+=("--loss random_loss")

    # Softmax
    configs_ref+=("--loss cross_entropy")
    configs_ref+=("--loss brier_score")

    # CE with label smoothing
    for ls in "${labelsmoothing[@]}"; do
        configs_ref+=("--loss cross_entropy --label-smoothing $ls")
    done

    # Focal and its variants
    for g in "${gammas[@]}"; do
        configs_ref+=("--loss focal_loss --gamma $g")
        configs_ref+=("--loss linear --gamma $g")
        configs_ref+=("--loss exp_p --gamma $g")
        configs_ref+=("--loss exp_1mp --gamma $g")
        configs_ref+=("--loss one_minus_power --gamma $g")
        configs_ref+=("--loss log_power --gamma $g")
        # configs_ref+=("--loss focal_loss_adaptive --gamma $g")
        configs_ref+=("--loss proper_focal_loss --gamma $g")
    done
    # configs_ref+=("--loss adafocal")

    # Generalized focal (grid over beta × gamma)
    # for b in "${betas[@]}"; do
    #     for g in "${gammas[@]}"; do
    #         configs_ref+=("--loss generalized_focal --beta $b --gamma $g")
    #     done
    # done
}
