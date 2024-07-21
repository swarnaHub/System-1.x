import json
import os
import sys
import random
sys.path.insert(1, '/nas-hdd/swarna/projects/System-1.x')
from tasks.blocksworld import BlocksWorld
import decomp_hybrid

def define_prompt(sample):
    prompt = sample['prompt']

    task_description = 'Your task is to generate a plan for a blocksworld problem given an initial state and a goal state.\n\n'

    return task_description + prompt


def get_system2_indices(samples, system, method):
    system2_indices = []

    if method == "heuristic":
        dists = [(i, BlocksWorld.heuristic_func(sample['start_as_blocks'], sample['goal_as_blocks'])) for i, sample in enumerate(samples)]
        dists.sort(key = lambda x:x[1], reverse=True)
        system2_dists = dists[:int(len(samples)*(system))]
        system2_indices = [system2_dist[0] for system2_dist in system2_dists]
    else:
        assert False, f"Method {method} not recognized"

    return system2_indices

def select_system2_plan(args, sample):
    search_algo = args.search_algo
    if search_algo == "a_star":
        return sample["system2_plan_v1"]
    else:
        assert False, "search algo not detected"

def prepare_prompts_task_level(args, 
                               file, 
                               start = 0, 
                               end = -1
                               ):
    
    samples = json.load(open(os.path.join(args.data_dir, file), 'r'))
    if end >= 0:
        samples = samples[start:end]

    system2_indices = get_system2_indices(samples, args.system, args.method)

    print(f"System 2 indices = {system2_indices}")

    data, bw_samples = [], []
    for i, sample in enumerate(samples):
        idx = sample['idx']
        start = sample['start']
        goal = sample['goal']
        start_as_blocks = sample['start_as_blocks']
        goal_as_blocks = sample['goal_as_blocks']
        system1_plan = sample['system1_plan']
        system2_plan = select_system2_plan(args, sample)
        
        if args.method == "random" or args.system in [0.0, 1.0]:
            prompt = define_prompt(sample)
            if i in system2_indices:
                output = f"<start system 2> {system2_plan} <end system 2>" 
            else:
                output = f"<start system 1> {system1_plan} <end system 1>"
        elif args.method == "heuristic":
            prompt = define_prompt(sample) 
            output = BlocksWorld.verbalized_heuristic(start_as_blocks, goal_as_blocks)

            if i in system2_indices:
                output = output + f" So the plan should be generated slowly using system 2"
            else:
                output = output + f" So the plan should be generated fast using system 1"
        
        data.append({
                'idx': idx,
                'prompt': prompt,
                'plan': output
            })

        bw_sample = BlocksWorld(
            initial_state = start,
            goal_state = goal,
            initial_state_as_blocks = start_as_blocks,
            goal_state_as_blocks = goal_as_blocks,
            system1_plan = system1_plan,
            idx = idx
        )

        assert bw_sample.is_valid_plan() == True
        
        bw_samples.append(bw_sample)

    return data, bw_samples


def get_test_prompts(args, test_file):
    meta_planner_data, test_samples = prepare_prompts_task_level(args, 
                                                                 file = test_file
                                                                 )
    meta_planner_prompts = [meta_planner_sample['prompt'] for meta_planner_sample in meta_planner_data]
    old_system = args.system

    # Preparing these prompts too as they will be invoked during inference from the meta-planner
    args.system = 0.0
    system1_planner_data, _ = prepare_prompts_task_level(args, 
                                                         file = test_file
                                                         )
    system1_planner_prompts = [system1_planner_sample['prompt'] for system1_planner_sample in system1_planner_data]

    args.system = 1.0
    system2_planner_data, _ = prepare_prompts_task_level(args, 
                                                         file = test_file
                                                         )
    system2_planner_prompts = [system2_planner_sample['prompt'] for system2_planner_sample in system2_planner_data]

    args.system = old_system

    return meta_planner_prompts, system1_planner_prompts, system2_planner_prompts, test_samples

def get_finetuning_data(args):
    if args.level == "task" or args.system in [0.0, 1.0]:
        data_train, _ = prepare_prompts_task_level(args, 
                                                   file = 'train.json', 
                                                   start = args.n_train_start, 
                                                   end = args.n_train_end
                                                   )
        data_val, _ = prepare_prompts_task_level(args, 
                                                 file = 'validation.json',
                                                 start = 0, 
                                                 end = args.n_val
                                                 )
        data_test, _ = prepare_prompts_task_level(args, 
                                                  'test.json', 
                                                  start = 0, 
                                                  end = args.n_test
                                                  )
    else:
        data_train, _, _ = decomp_hybrid.prepare_prompts_sample_level(args,
                                                                     'train.json',
                                                                     start = args.n_train_start, 
                                                                     end = args.n_train_end
                                                                     )
        data_val, _, _ = decomp_hybrid.prepare_prompts_sample_level(args,
                                                                   'validation.json',
                                                                   start = 0, 
                                                                   end = args.n_val
                                                                   )
        data_test, _, _ = decomp_hybrid.prepare_prompts_sample_level(args,
                                                                    'test.json',
                                                                    start = 0, 
                                                                    end = args.n_test
                                                                    )

    return data_train, data_val, data_test



    






