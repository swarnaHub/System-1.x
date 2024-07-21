import json
import os
import sys
import random
sys.path.insert(1, '/nas-hdd/swarna/projects/System-1.x')
from tasks.maze import Maze
import decomp_hybrid

def define_prompts():
    task_description = 'You are in a 2d maze of dimensions {l} and {w} and some of the cells have walls. The walls are placed in cells {walls}.'

    permissible_actions = 'The list of permissible actions that you can take at any given cell are {actions}.'

    optimal_plan_objective = 'Given a start and a goal state, your task is to generate the optimal plan as a sequence of actions. The optimal plan is one that has the minimum number of steps.'
    prompt = task_description + ' ' + optimal_plan_objective + ' ' + permissible_actions + ' The optimal plan from {start} to {goal} is'

    prompt_with_manhattan = task_description + ' ' + optimal_plan_objective + ' ' + permissible_actions + ' Find the optimal plan from {start} to {goal}. The manhattan distance from {start} to {goal} is'

    prompt_with_obstacles = task_description + ' ' + optimal_plan_objective + ' ' + permissible_actions + ' Find the optimal plan from {start} to {goal}. The walls between {start} to {goal} are at'

    return prompt, prompt_with_manhattan, prompt_with_obstacles


def format_plan(plan, format):
    plan_steps = plan[1:-1].split(' | ')
    formatted_plan_steps = []
    for i, plan_step in enumerate(plan_steps):
        parts = plan_step.split(' ')
        if format == 'action_only':
            if i > 0:
                formatted_plan_steps.append(parts[0])
        else:
            assert False, f'wrong format {format}'

    return ' | '.join(formatted_plan_steps)


def get_system2_indices(samples, system, method):
    system2_indices = []

    if method == 'random':
        system2_indices = random.sample(range(len(samples)), int(len(samples)*system))
    elif method == "manhattan":
        dists = [(i, Maze.manhattan_distance(sample['start'], sample['goal'])) for i, sample in enumerate(samples)]
        dists.sort(key = lambda x:x[1], reverse=True)
        system2_dists = dists[:int(len(samples)*(system))]
        system2_indices = [system2_dist[0] for system2_dist in system2_dists]
    elif method == "obstacles":
        obstacles = [(i, Maze.obstacles(sample['start'], sample['goal'], sample['walls'])) for i, sample in enumerate(samples)]
        obstacles.sort(key = lambda x:x[1], reverse=True)
        system2_obstacles = obstacles[:int(len(samples)*system)]
        system2_indices = [system2_obstacle[0] for system2_obstacle in system2_obstacles]

    return system2_indices

def format_system1_plan(plan):
    formatted_plan = ""
    prev_state = None
    plan_so_far = []
    for i, step in enumerate(plan.split(" | ")):
        action = step.split(" ")[0]
        state = step[len(action):].strip()
        if i == 0:
            formatted_plan += f"Moved to state {state} | Plan so far {plan_so_far}"
        else:
            plan_so_far = plan_so_far + [f'{action}']
            formatted_plan += f" | Taking action '{action}' from state {prev_state} | Moved to state {state} | Plan so far {plan_so_far}"

        prev_state = state

    formatted_plan += f" | Goal state {prev_state} reached!"

    return formatted_plan

def select_system2_plan(args, sample):
    search_algo = args.search_algo
    if search_algo == "bfs":
        return sample["system2_bfs_plan"]
    elif search_algo == "dfs":
        return sample["system2_dfs_plan"]
    elif search_algo == "a_star":
        return sample["system2_a_star_plan"]
    else:
        assert False, "search algo not detected"

def prepare_prompts_task_level(args, 
                               file, 
                               start = 0, 
                               end = -1
                               ):
    prompt_wrapper, manhattan_prompt_wrapper, obstacles_prompt_wrapper = define_prompts()

    samples = json.load(open(os.path.join(args.data_dir, file), 'r'))
    if end >= 0:
        samples = samples[start:end]

    system2_indices = get_system2_indices(samples, args.system, args.method)

    # print(f"System 2 indices = {system2_indices}")

    data, maze_samples = [], []
    for i, sample in enumerate(samples):
        idx = sample['idx']
        l = sample['l']
        w = sample['w']
        start = sample['start']
        goal = sample['goal']
        actions = sample['actions']
        walls = sample['walls']
        system1_plan = sample['system1_plan']
        system2_plan = select_system2_plan(args, sample)
        
        if args.method == "random" or args.system in [0.0, 1.0]:
            prompt = prompt_wrapper.format(
            l = l,
            w = w,
            walls = walls,
            actions = actions,
            start = start,
            goal = goal
            )
            if i in system2_indices:
                output = f"<start system 2> {system2_plan} <end system 2>" 
            else:
                output = f"<start system 1> {system1_plan} <end system 1>"

        elif args.method == "manhattan":
            prompt = manhattan_prompt_wrapper.format(
                    l = l,
                    w = w,
                    walls = walls,
                    actions = actions,
                    start = start,
                    goal = goal
                    )
            x_dist = abs(start[0]-goal[0])
            y_dist = abs(start[1]-goal[1])
            total_dist = x_dist + y_dist
            output = f"abs({start[0]}-{goal[0]}) + abs({start[1]}-{goal[1]}) = {x_dist} + {y_dist} = {total_dist}."

            if i in system2_indices:
                output = output + f" So the plan should be generated slowly using system 2"
            else:
                output = output + f" So the plan should be generated fast using system 1"

        elif args.method == "obstacles":
            prompt = obstacles_prompt_wrapper.format(
                    l = l,
                    w = w,
                    walls = walls,
                    actions = actions,
                    start = start,
                    goal = goal
                    )
            output = ""
            count_walls = 0
            for row in range(min(start[0], goal[0]), max(start[0], goal[0])+1):
                for col in range(min(start[1], goal[1]), max(start[1], goal[1])+1):
                    if [row, col] in sample['walls']:
                        count_walls += 1
                        output = output + ", " + str([row, col]) if output != "" else str([row, col])

            output += f" which are a total of {count_walls} walls."
            if i in system2_indices:
                output = output + f" So the plan should be generated slowly using system 2"
            else:
                output = output + f" So the plan should be generated fast using system 1"
            
        
        data.append({
                'idx': idx,
                'prompt': prompt,
                'plan': output
            })

        maze_sample = Maze(
            l = l,
            w = w,
            actions = actions,
            start = start,
            goal = goal,
            walls = walls,
            system1_plan = system1_plan,
            system2_plan = system2_plan,
            idx = idx
        )
        
        maze_samples.append(maze_sample)

    return data, maze_samples


def get_test_prompts(args):
    meta_planner_data, test_samples = prepare_prompts_task_level(args, 
                                                                 file = 'test.json'
                                                                 )
    meta_planner_prompts = [meta_planner_sample['prompt'] for meta_planner_sample in meta_planner_data]
    old_system = args.system

    # Preparing these prompts too as they will be invoked during inference from the meta-planner
    args.system = 0.0
    system1_planner_data, _ = prepare_prompts_task_level(args, 
                                                         file = 'test.json'
                                                         )
    system1_planner_prompts = [system1_planner_sample['prompt'] for system1_planner_sample in system1_planner_data]

    args.system = 1.0
    system2_planner_data, _ = prepare_prompts_task_level(args, 
                                                         file = 'test.json'
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



    






