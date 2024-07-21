import re
import statistics

def extract_plan_from_system2(system2_plan):
    plan = ""
    pattern = r"\[(.*?)\]"
    for search_elem in reversed(system2_plan.split(" | ")):
        if 'Plan so far' in search_elem:
            plan = re.findall(pattern, search_elem)
            if len(plan) > 0:
                plan = plan[0]
                actions = plan.replace("'", "").split(", ")
                plan = " | ".join(actions)
                break

    return "start | " + plan

def extract_plan_from_system1(system1_plan):
    system1_pattern = r"<start system 1> (.*?) <end system 1>"
    system1_plan = re.findall(system1_pattern, system1_plan)
    system1_plan = system1_plan[0] if len(system1_plan) > 0 else ""
    return system1_plan


def eval_plans_task_level(test_samples, predicted_plans):
    assert len(test_samples) == len(predicted_plans), "Gold and predictions should be of equal lengths"

    valid_plans, optimal_plans, states_visited = [], [], []
    for test_sample, predicted_plan in zip(test_samples, predicted_plans):
        if 'system 1' in predicted_plan:
            states_visited.append(predicted_plan.count("|"))
            predicted_plan = extract_plan_from_system1(predicted_plan)
        else:
            states_visited.append(predicted_plan.count("Exploring"))
            predicted_plan = extract_plan_from_system2(predicted_plan)
        print(f'Predicted plan = {predicted_plan}')
        
        validity = test_sample.is_valid_plan(plan = predicted_plan)

        optimality = test_sample.is_optimal_plan(plan = predicted_plan,
                                                check_validity = False) if validity else False
        
        valid_plans.append(validity)
        optimal_plans.append(optimality)

    return valid_plans, optimal_plans, states_visited

def eval_plan_sample_level(test_sample, predicted_plan, silver_sub_goals, predicted_sub_goals):
    combined_plan = []
    states_visited = 0
    for sub_plan in predicted_plan:
        if 'system 1' in sub_plan:
            states_visited += sub_plan.count('|')
            sub_plan = extract_plan_from_system1(sub_plan)
        else:
            states_visited += sub_plan.count('Exploring')
            sub_plan = extract_plan_from_system2(sub_plan)
        print(f'sub plan = {sub_plan}')

        combined_plan.extend(sub_plan.split(" | ")[1:])
    
    combined_plan = " | ".join(["start"] + combined_plan)

    print(f"Combined plan = {combined_plan}")
        
    validity = test_sample.is_valid_plan(plan = combined_plan)

    optimality = test_sample.is_optimal_plan(plan = combined_plan,
                                            check_validity = False) if validity else False
    
    sub_goals_full_correct = silver_sub_goals == predicted_sub_goals
    sub_goals_valid = test_sample.is_sub_goals_valid(predicted_sub_goals)
    
    return validity, optimality, combined_plan, states_visited, sub_goals_full_correct, sub_goals_valid


def compute_metrics(valid_plans, optimal_plans, output):
    print(f'Valid plan accuracy = {valid_plans/len(output)}')
    print(f'Optimal plan accuracy = {optimal_plans/len(output)}')

    valid_plans_by_length, optimal_plans_by_length, all_plans_by_length = {}, {}, {}
    for sample in output:
        optimal_plan_len = len(sample['gold_plan'].split(' | ')) - 1
        all_plans_by_length[optimal_plan_len] = all_plans_by_length.get(optimal_plan_len, 0) + 1
        if sample['valid']:
            valid_plans_by_length[optimal_plan_len] = valid_plans_by_length.get(optimal_plan_len, 0) + 1

        if sample['optimal']:
            optimal_plans_by_length[optimal_plan_len] = optimal_plans_by_length.get(optimal_plan_len, 0) + 1


    valid_accuracy_by_length = {l:valid_plans_by_length.get(l, 0)/all_plans_by_length[l] for l in all_plans_by_length.keys()}
    optimal_accuracy_by_length = {l:optimal_plans_by_length.get(l, 0)/all_plans_by_length[l] for l in all_plans_by_length.keys()}

    print(f'Count by length = {all_plans_by_length}')
    print(f'Valid plans accuracy by length = {valid_accuracy_by_length}')
    print(f'Optimal plans accuracy by length = {optimal_accuracy_by_length}')

    states_by_length = {}
    for pred in output:
        optimal_plan_len = len(pred['gold_plan'].split(' | ')) - 1
        if isinstance(pred['predicted_plan'], str):
            if 'system 1' in pred['predicted_plan']:
                states = pred['predicted_plan'].count('|')
            else:
                states = pred['predicted_plan'].count('Exploring')
        else:
            states = 0
            for sub_plan in pred['predicted_plan']:
                if 'system 1' in sub_plan:
                    states += sub_plan.count('|')
                else:
                    states += sub_plan.count('Exploring')
        if optimal_plan_len not in states_by_length:
            states_by_length[optimal_plan_len] = [states]
        else:
            states_by_length[optimal_plan_len].append(states)

    mean_states_by_length = {l:statistics.mean(states) for l, states in states_by_length.items()}

    print(f'Mean states = {statistics.mean(sum(states_by_length.values(), []))}')
    print(f'Mean states by length = {mean_states_by_length}')
