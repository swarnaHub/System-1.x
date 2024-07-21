import json
import re
import os
import random
import itertools
import eval_utils
from tasks.blocksworld import BlocksWorld

def get_plan_decomposition_sliding_window(sample, start_state, goal_state, start_as_blocks, goal_as_blocks, system2_plan, system):
    plan = eval_utils.extract_plan_from_system2(system2_plan).split(" | ")
    plan_len = len(plan)

    decompositions = []
    for system2_sub_goal_start in range(0, plan_len-int(plan_len*system)+1):
        system2_sub_goal_end = system2_sub_goal_start + int(plan_len*system)

        system2_sub_goal_start_state, system2_sub_goal_start_state_as_list = sample.follow_plan(start_as_blocks, plan[:system2_sub_goal_start])
        system2_sub_goal_end_state, system2_sub_goal_end_state_as_list = sample.follow_plan(system2_sub_goal_start_state, plan[system2_sub_goal_start:system2_sub_goal_end])

        system2_cost = BlocksWorld.heuristic_func(system2_sub_goal_start_state, system2_sub_goal_end_state)
        
        system1_cost_first = BlocksWorld.heuristic_func(start_as_blocks, system2_sub_goal_start_state)

        system1_cost_second = BlocksWorld.heuristic_func(system2_sub_goal_end_state, goal_as_blocks)

        total_cost = system1_cost_first - system2_cost + system1_cost_second

        system2_sub_goal_start_state_as_list = [["table"] + stack for stack in system2_sub_goal_start_state_as_list]
        system2_sub_goal_end_state_as_list = [["table"] + stack for stack in system2_sub_goal_end_state_as_list]

        num_decomps = 2 if start_state == system2_sub_goal_start_state_as_list or system2_sub_goal_end_state_as_list == goal_state else 3 

        # Interleaving of sys1, sys2
        decompositions.append(([start_state, system2_sub_goal_start_state_as_list, system2_sub_goal_end_state_as_list, goal_state], total_cost, num_decomps))

        assert system2_sub_goal_end <= len(plan)

    return decompositions

def get_planner_prompt(sample, start, goal):
    task_description = 'Your task is to generate a plan for a blocksworld problem given an initial state and a goal state.\n\n'

    prompt = 'The initial state:\nThe hand is empty.\n'
    for stack in start: 
        for i in range(len(stack)): 
            if i == 0: 
                prompt += f'{stack[i]} is on the table. '
            else: 
                prompt += f'{stack[i]} is on {stack[i - 1]}. ' 
        prompt += f'{stack[-1]} is clear.\n'

    prompt += '\nThe goal is:\n'
    for stack in goal: 
        for i in range(len(stack)): 
            if i == 0: 
                prompt += f'{stack[i]} is on the table. '
            else: 
                prompt += f'{stack[i]} is on {stack[i - 1]}. ' 
        prompt += f'{stack[-1]} is clear.\n'

    return task_description + prompt

def convert_decomp_to_subgoals(best_decomp):
    system = 2
    sub_problem_counter = 1
    sub_goals = []
    for i in range(len(best_decomp)-1):
        start_state = best_decomp[i]
        goal_state = best_decomp[i+1]

        system = 2 if system == 1 else 1

        if start_state == goal_state:
            continue

        sub_goals.append((start_state, goal_state, system))
        sub_problem_counter += 1

    return sub_goals

def verbalize_state(state):
    verbalized_state = ""
    for i, stack in enumerate(state):
        stack = stack[1:]
        verbalized_state += f"Stack {i+1}: {stack}, "

    return verbalized_state[:-2]

def generate_interleaved_meta_plan(sub_goals):
    meta_planner = ""
    for sub_problem_counter, sub_goal in enumerate(sub_goals):
        system = sub_goal[2]
        start_state = verbalize_state(sub_goal[0]).replace("'", "")
        end_state = verbalize_state(sub_goal[1]).replace("'", "")

        if system == 1:
            meta_planner += f" {sub_problem_counter+1}. Start State -> {start_state} and Goal State -> {end_state}. Planning for this subgoal is easy and can be solved fast using system 1 |"
        else:
            meta_planner += f" {sub_problem_counter+1}. Start State -> {start_state} and Goal State -> {end_state}. Planning for this subgoal is hard and can be solved slow using system 2 |"
        
    meta_planner = f" Based on this, the plan should be broken down into {len(sub_goals)} sub-goals:" + meta_planner

    return meta_planner

def define_prompt(sample):
    prompt = sample['prompt']

    task_description = 'You are given a blocksworld problem and its initial state, and goal state. You task is to first generate a set of subgoals that can then be solved to generate the plan between the states.\n\n'

    return task_description + prompt

def get_decomposition_indices(samples, system, method):
    decomposition_indices = []

    if method == 'random':
        decomposition_indices = random.sample(range(len(samples)), int(len(samples)*system))
    elif method == "heuristic":
        dists = [(i, BlocksWorld.heuristic_func(sample['start_as_blocks'], sample['goal_as_blocks'])) for i, sample in enumerate(samples)]
        dists.sort(key = lambda x:x[1], reverse=True)
        decomposition_dists = dists[:int(len(samples)*(system))]
        decomposition_indices = [decomposition_dist[0] for decomposition_dist in decomposition_dists]

    return decomposition_indices

def select_system2_plan(args, sample):
    search_algo = args.search_algo
    if search_algo == "a_star":
        return sample["system2_plan_v1"]
    else:
        assert False, "search algo not detected"

def prepare_prompts_sample_level(args, 
                                 file, 
                                 start = 0, 
                                 end = -1):
    output_file = open(os.path.join('/nas-hdd/swarna/projects/Learning2Plan/data/blocksworld_sample_level', args.decomp_style + '_' + args.method + '_' + file), 'w')

    samples = json.load(open(os.path.join(args.data_dir, file), 'r'))
    if end >= 0:
        samples = samples[start:end]

    decomposition_indices = get_decomposition_indices(samples, args.system, args.method)

    data, bw_samples, all_sub_goals = [], [], []
    for i, sample in enumerate(samples):
        # print(i)
        idx = sample['idx']
        start = sample['start']
        goal = sample['goal']
        start_as_blocks = sample['start_as_blocks']
        goal_as_blocks = sample['goal_as_blocks']
        system1_plan = sample['system1_plan']
        system2_plan = select_system2_plan(args, sample)

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

        # If no decomposition is needed, solve the whole problem using system1
        # else, decompose into sub-goals
        if i not in decomposition_indices:
            decompositions = [([start, goal], -1, 1)]
        else:
            decompositions = get_plan_decomposition_sliding_window(bw_sample, start, goal, start_as_blocks, goal_as_blocks, system2_plan, args.system)

        decompositions.sort(key = lambda x: (x[1], x[2]))
        best_decomp = decompositions[0][0]
        sub_goals = convert_decomp_to_subgoals(best_decomp)
        all_sub_goals.append(sub_goals)
        
        prompt = define_prompt(sample)
        output = generate_interleaved_meta_plan(sub_goals).strip()

        data.append({
            'idx': idx,
            'prompt': prompt,
            'plan': output
        })

    json.dump(data, output_file, indent=4)

    return data, bw_samples, all_sub_goals

def get_meta_planner_test_prompts(args, test_file='test.json'):
    meta_planner_data, test_samples, decomps = prepare_prompts_sample_level(args, test_file)
    meta_planner_prompts = [meta_planner_sample['prompt'] for meta_planner_sample in meta_planner_data]

    return meta_planner_prompts, test_samples, decomps

def check_subgoal_sanity(start, sub_goal_state):
    return set(list(itertools.chain(*start))) == set(list(itertools.chain(*sub_goal_state)))

def parse_meta_plan(meta_plan, start, goal):
    start = [stack[1:] for stack in start]
    goal = [stack[1:] for stack in goal]

    sub_goals = []
    sub_goal_pattern = r"Start State -> (.*?) and Goal State -> (.*?)\."
    stack_pattern = r'\[.*?\]'
    for segment in meta_plan.split(" | "):
        sub_goal = re.findall(sub_goal_pattern, segment)
        if len(sub_goal) == 1 and len(sub_goal[0]) == 2:
            sub_goal_start, sub_goal_end = sub_goal[0][0], sub_goal[0][1]

            sub_goal_start_stacks = re.findall(stack_pattern, sub_goal_start)
            sub_goal_end_stacks = re.findall(stack_pattern, sub_goal_end)

            if not len(sub_goal_start_stacks) or not len(sub_goal_end_stacks):
                return [(start, goal, 2)]
            
            sub_goal_start_stacks = [stack.strip('[]').split(', ') for stack in sub_goal_start_stacks]
            sub_goal_end_stacks = [stack.strip('[]').split(', ') for stack in sub_goal_end_stacks]

            if not check_subgoal_sanity(start, sub_goal_start_stacks) or not check_subgoal_sanity(start, sub_goal_end_stacks):
                return [(start, goal, 2)]

            if (len(sub_goals) == 0 and sub_goal_start_stacks != start) or (len(sub_goals) > 0 and sub_goal_start_stacks != sub_goals[-1][1]):
                return [(start, goal, 2)]

            system = 1 if 'system 1' in segment else 2
            sub_goals.append((sub_goal_start_stacks, sub_goal_end_stacks, system))

    
    if len(sub_goals) == 0 or len(sub_goals[-1]) != 3 or sub_goals[-1][1] != goal:
        return [(start, goal, 2)]
    
    return sub_goals