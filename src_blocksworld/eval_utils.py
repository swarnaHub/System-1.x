import re
import statistics

def extract_plan_from_system2(system2_plan):
    pattern = r"Plan so far \[([^\]]*)\]"
    all_plan_so_far = re.findall(pattern, system2_plan)
    if len(all_plan_so_far):
        return all_plan_so_far[-1]

    return ""

def extract_plan_from_system1(system1_plan):
    system1_pattern = r"<start system 1> (.*?) <end system 1>"
    system1_plan = re.findall(system1_pattern, system1_plan)
    system1_plan = system1_plan[0] if len(system1_plan) > 0 else ""
    return system1_plan


def eval_plans_task_level(test_samples, predicted_plans):
    assert len(test_samples) == len(predicted_plans), "Gold and predictions should be of equal lengths"

    valid_plans, states_visited = [], []
    for test_sample, predicted_plan in zip(test_samples, predicted_plans):
        if 'system 1' in predicted_plan:
            states_visited.append(predicted_plan.count("|")+1)
            predicted_plan = extract_plan_from_system1(predicted_plan)
        else:
            states_visited.append(predicted_plan.count("Exploring"))
            predicted_plan = extract_plan_from_system2(predicted_plan)
        print(f'Predicted plan = {predicted_plan}')
        
        validity = test_sample.is_valid_plan(plan = predicted_plan)
        
        valid_plans.append(validity)

    return valid_plans, states_visited

def eval_plan_sample_level(test_sample, predicted_plan):
    combined_plan = []
    states_visited = 0
    for sub_plan in predicted_plan:
        if 'system 1' in sub_plan:
            states_visited += sub_plan.count('|') + 1
            sub_plan = extract_plan_from_system1(sub_plan)
        else:
            states_visited += sub_plan.count('Exploring')
            sub_plan = extract_plan_from_system2(sub_plan)

        print(f'sub plan = {sub_plan}')

        combined_plan.extend(sub_plan.split(" | "))
    
    combined_plan = " | ".join(combined_plan)

    print(f"Combined plan = {combined_plan}")
        
    validity = test_sample.is_valid_plan(plan = combined_plan)
        
    return validity, combined_plan, states_visited


def compute_metrics(valid_plans, output):
    print(f'Valid plan accuracy = {valid_plans/len(output)}')

    valid_plans_by_length, all_plans_by_length = {}, {}
    valid_plans_by_blocks, all_plans_by_blocks = {}, {}
    for sample in output:
        plan_len = sample['gold_plan'].count('|') + 1
        num_blocks = sample['num_blocks']
        all_plans_by_length[plan_len] = all_plans_by_length.get(plan_len, 0) + 1
        all_plans_by_blocks[num_blocks] = all_plans_by_blocks.get(num_blocks, 0) + 1
        if sample['valid']:
            valid_plans_by_length[plan_len] = valid_plans_by_length.get(plan_len, 0) + 1
            valid_plans_by_blocks[num_blocks] = valid_plans_by_blocks.get(num_blocks, 0) + 1

    valid_accuracy_by_length = {l:valid_plans_by_length.get(l, 0)/all_plans_by_length[l] for l in all_plans_by_length.keys()}
    valid_accuracy_by_blocks = {b:valid_plans_by_blocks.get(b, 0)/all_plans_by_blocks[b] for b in all_plans_by_blocks.keys()}

    print(f'Count by length = {all_plans_by_length}')
    print(f'Valid plans accuracy by length = {valid_accuracy_by_length}')

    print(f'Count by blocks = {all_plans_by_blocks}')
    print(f'Valid plans accuracy by blocks = {valid_accuracy_by_blocks}')

    states_by_length = {}
    for pred in output:
        plan_len = pred['gold_plan'].count('|') + 1
        predicted_plan = pred['predicted_plan']
        if isinstance(predicted_plan, str):
            states = predicted_plan.count('|') + 1 if 'system 1' in predicted_plan else predicted_plan.count('Exploring')
        else:
            states = 0
            for sub_plan in predicted_plan:
                states = states + sub_plan.count('|') + 1 if 'system 1' in sub_plan else states + sub_plan.count('Exploring')
        if plan_len not in states_by_length:
            states_by_length[plan_len] = [states]
        else:
            states_by_length[plan_len].append(states)

    mean_states_by_length = {l:statistics.mean(states) for l, states in states_by_length.items()}

    print(f'Mean states = {statistics.mean(sum(states_by_length.values(), []))}')
    print(f'Mean states by length = {mean_states_by_length}')
