#!/bin/bash


function define_seeds() {
    local -n seeds_ref=$1
    seeds_ref+=(42 123 2023) # 7 99)
}

function build_configs() {
    local -n configs_ref=$1
    local -a gammas=(0.25 0.5 1 2 3 5 7) # (1 2) # 
    local -a betas=(0.25 0.5 1 2 3 5 7) # (1 2) # 

    # Softmax
    configs_ref+=("--loss cross_entropy")

    # Focal and its variants
    for g in "${gammas[@]}"; do
        configs_ref+=("--loss focal_loss --gamma $g")
        configs_ref+=("--loss focal_loss_adaptive --gamma $g")
        configs_ref+=("--loss linear --gamma $g")
        configs_ref+=("--loss exp_p --gamma $g")
        configs_ref+=("--loss exp_1mp --gamma $g")
        configs_ref+=("--loss one_minus_power --gamma $g")
        configs_ref+=("--loss log_power --gamma $g")
    done

    # Generalized focal (grid over beta × gamma)
    for b in "${betas[@]}"; do
        for g in "${gammas[@]}"; do
            configs_ref+=("--loss generalized_focal --beta $b --gamma $g")
        done
    done
}
