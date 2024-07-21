python src_maze/test.py \
    --data_dir data/maze \
    --output_dir output \
    --meta_planner models/maze/a_star_obstacles_sliding_sample_system_0.5_3200_epoch_1_lr_0.0005_bs_2 \
    --system1_planner models/maze/a_star_obstacles_sliding_task_system_0.0_3200_epoch_3_lr_0.0005_bs_2 \
    --system2_planner models/maze/a_star_obstacles_sliding_task_system_1.0_3200_epoch_3_lr_0.0005_bs_2 \
    --level sample \
    --search_algo a_star \
    --system 0.5