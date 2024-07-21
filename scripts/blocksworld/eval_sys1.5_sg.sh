python src_blocksworld/test.py \
    --data_dir data/blocksworld \
    --output_dir output \
    --meta_planner models/blocksworld/a_star_heuristic_sliding_sample_system_0.5_3000_epoch_1_lr_0.0005_bs_2 \
    --system1_planner models/blocksworld/a_star_heuristic_sliding_task_system_0.0_3000_epoch_1_lr_0.0005_bs_4 \
    --system2_planner models/blocksworld/a_star_heuristic_sliding_task_system_1.0_3000_epoch_3_lr_0.0005_bs_2 \
    --level sample \
    --search_algo a_star \
    --system 0.5