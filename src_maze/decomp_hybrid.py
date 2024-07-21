import json
import os
import re
import random
from tasks.maze import Maze

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
            

def generate_interleaved_meta_plan(sub_goals):
    meta_planner = ""
    for sub_problem_counter, sub_goal in enumerate(sub_goals):
        system = sub_goal[2]

        if system == 1:
            meta_planner += f" {sub_problem_counter+1}. Start State: {sub_goal[0]} and Goal State: {sub_goal[1]}. Planning for this subgoal is easy and can be solved fast using system 1 |"
        else:
            meta_planner += f" {sub_problem_counter+1}. Start State: {sub_goal[0]} and Goal State: {sub_goal[1]}. Planning for this subgoal is hard and can be solved slow using system 2 |"


    meta_planner = f" Based on this, the plan should be broken down into {len(sub_goals)} sub-goals:" + meta_planner

    return meta_planner

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


def get_plan_decomposition_into_two(sample, start_state, goal_state, system2_plan, system):
    plan = system2_plan.split(" | ")[-2][len("Plan so far ["):-1].replace("'", "").split(", ")
    plan_len = len(plan)

    decompositions = []
    system2_sub_goal_end = int(plan_len*system)
    system2_sub_goal_end_state = sample.follow_plan(start_state, plan[0:system2_sub_goal_end])
    system2_cost = Maze.obstacles(start_state, system2_sub_goal_end_state, sample.walls)
    system1_cost = Maze.obstacles(system2_sub_goal_end_state, goal_state, sample.walls)
    total_cost = system1_cost - system2_cost

    decompositions.append(([start_state, start_state, system2_sub_goal_end_state, goal_state], total_cost, 2))
    decompositions.append(([start_state, system2_sub_goal_end_state, goal_state, goal_state], -total_cost, 2))

    print(decompositions)

    return decompositions


def get_plan_decomposition_sliding_window(sample, start_state, goal_state, system2_plan, system):
    plan = system2_plan.split(" | ")[-2][len("Plan so far ["):-1].replace("'", "").split(", ")
    plan_len = len(plan)

    # Training data construction for intra-instance 1.x planner:
    decompositions = []
    for system2_sub_goal_start in range(0, plan_len-int(plan_len*system)+1):
        system2_sub_goal_end = system2_sub_goal_start + int(plan_len*system)

        system2_sub_goal_start_state = sample.follow_plan(start_state, plan[:system2_sub_goal_start])
        system2_sub_goal_end_state = sample.follow_plan(system2_sub_goal_start_state, plan[system2_sub_goal_start:system2_sub_goal_end])
        system2_cost = Maze.obstacles(system2_sub_goal_start_state, system2_sub_goal_end_state, sample.walls)
        
        system1_cost_first = Maze.obstacles(start_state, system2_sub_goal_start_state, sample.walls)

        system1_cost_second = Maze.obstacles(system2_sub_goal_end_state, goal_state, sample.walls)

        total_cost = system1_cost_first - system2_cost + system1_cost_second

        num_decomps = 2 if start_state == system2_sub_goal_start_state or system2_sub_goal_end_state == goal_state else 3 

        # Interleaving of sys1, sys2
        decompositions.append(([start_state, system2_sub_goal_start_state, system2_sub_goal_end_state, goal_state], total_cost, num_decomps))

        assert system2_sub_goal_end <= len(plan)

    return decompositions

def define_prompt():
    task_description = 'You are in a 2d maze of dimensions {l} and {w} and some of the cells have walls. The walls are placed in cells {walls}.'

    permissible_actions = 'The list of permissible actions that you can take at any given cell are {actions}.'

    optimal_plan_objective = 'Given a start and a goal state, your task is to generate the optimal plan as a sequence of actions. The optimal plan is one that has the minimum number of steps.'

    meta_plan_objective = 'Given a start and a goal state, your task is to first generate a set of subgoals that can then be solved to generate the optimal plan between the states. The optimal plan is one that has the minimum number of steps.'
    meta_prompt = task_description + ' ' + meta_plan_objective + ' ' + permissible_actions + ' Generate the subgoals for generating a plan from {start} to {goal}.'

    meta_prompt_with_manhattan = meta_prompt + ' The manhattan distance from {start} to {goal} is'

    meta_prompt_with_obstacles = meta_prompt + ' The walls between {start} to {goal} are at'

    optimal_plan_objective = 'Given a start and a goal state, your task is to generate the optimal plan as a sequence of actions. The optimal plan is one that has the minimum number of steps.'
    prompt = task_description + ' ' + optimal_plan_objective + ' ' + permissible_actions + ' The optimal plan from {start} to {goal} is'

    return meta_prompt_with_manhattan, meta_prompt_with_obstacles, prompt

def get_decomposition_indices(samples, system, method):
    decomposition_indices = []

    if method == "random":
        decomposition_indices = random.sample(range(len(samples)), int(len(samples)*system))
    elif method == "manhattan":
        dists = [(i, Maze.manhattan_distance(sample['start'], sample['goal'])) for i, sample in enumerate(samples)]
        dists.sort(key = lambda x:x[1], reverse=True)
        decomposition_dists = dists[:int(len(samples)*(system))]
        decomposition_indices = [decomposition_dist[0] for decomposition_dist in decomposition_dists]
    elif method == "obstacles":
        obstacles = [(i, Maze.obstacles(sample['start'], sample['goal'], sample['walls'])) for i, sample in enumerate(samples)]
        obstacles.sort(key = lambda x:x[1], reverse=True)
        decomposition_obstacles = obstacles[:int(len(samples)*system)]
        decomposition_indices = [decomposition_obstacle[0] for decomposition_obstacle in decomposition_obstacles]
    elif method == "both":
        dists = [(i, Maze.manhattan_distance(sample['start'], sample['goal'])) for i, sample in enumerate(samples)]
        obstacles = [(i, Maze.obstacles(sample['start'], sample['goal'], sample['walls'])) for i, sample in enumerate(samples)]
        dists_obstacles = [(dist[0], dist[1]+obstacle[1]) for (dist, obstacle) in zip(dists, obstacles)]
        dists_obstacles.sort(key = lambda x:x[1], reverse=True)
        decomposition_obstacles = dists_obstacles[:int(len(samples)*system)]
        decomposition_indices = [decomposition_obstacle[0] for decomposition_obstacle in decomposition_obstacles]

    return decomposition_indices

def select_system2_plan(args, sample):
    search_algo = args.search_algo
    if search_algo == "bfs":
        return sample["system2_bfs_plan"]
    elif search_algo == "dfs":
        return sample["system2_dfs_plan"]
    elif search_algo == "best_first":
        return sample["system2_best_first_plan"]
    elif search_algo == "a_star":
        return sample["system2_a_star_plan"]
    else:
        assert False, "search algo not detected"


def prepare_prompts_sample_level(args, file, start = 0, end = -1):
    output_file = open(os.path.join('/nas-hdd/swarna/projects/Learning2Plan/data/maze_sample_level', args.decomp_style + '_' + args.method + '_' + file), 'w')

    meta_prompt_with_manhattan_wrapper, meta_prompt_with_obstacles_wrapper, _ = define_prompt()

    samples = json.load(open(os.path.join(args.data_dir, file), 'r'))
    if end >= 0:
        samples = samples[start:end]

    # First get the samples that require interleaving/decomposition
    decomposition_indices = get_decomposition_indices(samples, args.system, args.method)

    data, maze_samples, all_sub_goals = [], [], []
    for i, sample in enumerate(samples):
        # print(i)
        idx = sample['idx']
        l = sample['l']
        w = sample['w']
        start = sample['start']
        goal = sample['goal']
        actions = sample['actions']
        walls = sample['walls']
        system1_plan = sample['system1_plan']
        system2_plan = select_system2_plan(args, sample)

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

        # If no decomposition is needed, solve the whole problem using system1
        # else, decompose into sub-goals
        if i not in decomposition_indices:
            decompositions = [([start, goal], -1, 1)]
        else:
            if args.decomp_style == 'sliding':
                decompositions = get_plan_decomposition_sliding_window(maze_sample, start, goal, system2_plan, args.system)
            else:
                decompositions = get_plan_decomposition_into_two(maze_sample, start, goal, system2_plan, args.system)

        decompositions.sort(key = lambda x: (x[1], x[2]))
        best_decomp = decompositions[0][0]
        sub_goals = convert_decomp_to_subgoals(best_decomp)
        print(sub_goals)
        all_sub_goals.append(sub_goals)

        # Based on the heuristic, prepare the prompt
        if args.method == "manhattan":
            prompt = meta_prompt_with_manhattan_wrapper.format(
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
        elif args.method == "obstacles":
            prompt = meta_prompt_with_obstacles_wrapper.format(
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

        
        output += generate_interleaved_meta_plan(sub_goals)

        data.append({
            'idx': idx,
            'prompt': prompt,
            'plan': output
        })

    json.dump(data, output_file, indent=4)

    return data, maze_samples, all_sub_goals

def get_meta_planner_test_prompts(args, test_file='test.json'):
    meta_planner_data, test_samples, decomps = prepare_prompts_sample_level(args, test_file)
    meta_planner_prompts = [meta_planner_sample['prompt'] for meta_planner_sample in meta_planner_data]

    return meta_planner_prompts, test_samples, decomps

def get_planner_prompt(test_sample, start, goal):
    _, _, prompt_wrapper = define_prompt()
    prompt = prompt_wrapper.format(
                l = test_sample.l,
                w = test_sample.w,
                walls = test_sample.walls,
                actions = test_sample.actions,
                start = start,
                goal = goal
                )
    return prompt


def parse_meta_plan(meta_plan, start, goal):
    sub_goals = []
    sub_goal_pattern = r"Start State: (.*?) and Goal State: (.*?)\."
    for segment in meta_plan.split(" | "):
        sub_goal = re.findall(sub_goal_pattern, segment)
        if len(sub_goal) == 1 and len(sub_goal[0]) == 2:
            sub_goal_start, sub_goal_end = sub_goal[0][0], sub_goal[0][1]

            sub_goal_start = [int(num) for num in re.findall(r'\d+', sub_goal_start)]
            sub_goal_end = [int(num) for num in re.findall(r'\d+', sub_goal_end)]

            # Could not parse properly
            if len(sub_goal_start) != 2 or len(sub_goal_end) != 2:
                return [(start, goal, 2)]

            # sub_goal_start, sub_goal_end = ast.literal_eval(sub_goal_start), ast.literal_eval(sub_goal_end)

            # If these sub-goals are not contiguous, detect error and solve the whole sub-problem
            if (len(sub_goals) == 0 and sub_goal_start != start) or (len(sub_goals) > 0 and sub_goal_start != sub_goals[-1][1]):
                return [(start, goal, 2)]

            system = 1 if 'system 1' in segment else 2
            sub_goals.append((sub_goal_start, sub_goal_end, system))
    
    if len(sub_goals) == 0 or len(sub_goals[-1]) != 3 or sub_goals[-1][1] != goal:
        return [(start, goal, 2)]
    
    return sub_goals


