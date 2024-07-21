import data_utils
import eval_utils
import os
import json
import numpy as np
import argparse
import decomp_hybrid
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_predictions(tokenizer, model, batch_prompts, with_confidence=False, intra_controller=False):
    input_tokens = tokenizer(batch_prompts, return_tensors='pt', padding=True).to('cuda')
    generated = model.generate(**input_tokens,
                                do_sample=False,
                                pad_token_id=tokenizer.eos_token_id,
                                max_new_tokens=args.max_new_tokens,
                                eos_token_id=tokenizer.eos_token_id,
                                num_return_sequences=1,
                                output_scores=True, 
                                return_dict_in_generate=True)
    
    predictions = tokenizer.batch_decode(generated[0], skip_special_tokens=True)
    predictions = [prediction[len(prompt):] for (prediction, prompt) in zip(predictions, batch_prompts)]
    predictions = [prediction.split("\n")[0].strip() for prediction in predictions]

    confidence = []
    if with_confidence:
        if not intra_controller:
            one_id, two_id = tokenizer.encode("1")[2], tokenizer.encode("2")[2]
            for i in range(len(batch_prompts)):
                generated_tokens_sample = generated[0][i]
                input_tokens_sample = input_tokens['input_ids'][i]
                generated_tokens_sample = generated_tokens_sample[generated_tokens_sample != 2]
                sys_index = len(generated_tokens_sample) - len(input_tokens_sample) - 2
                scores = softmax(generated['scores'][sys_index][i], dim=-1)
                sys1_conf = round(scores[one_id].item(), 4)
                sys2_conf = round(scores[two_id].item(), 4)

                confidence.append({
                    'sys1': sys1_conf,
                    'sys2': sys2_conf
                })
        else:
            one_id, two_id, delimeter_id = tokenizer.encode("1")[2], tokenizer.encode("2")[2], tokenizer.encode("|")[1]
            for i in range(len(batch_prompts)):
                sub_goal_confidence = []
                generated_tokens_sample = generated[0][i]
                input_tokens_sample = input_tokens['input_ids'][i]
                generated_tokens_sample = generated_tokens_sample[generated_tokens_sample != 2]
                for j in range(len(input_tokens_sample), len(generated_tokens_sample)):
                    if generated_tokens_sample[j] in [one_id, two_id] and generated_tokens_sample[j+1] == delimeter_id:
                        sys_index = j - len(input_tokens_sample)
                        scores = softmax(generated['scores'][sys_index][i], dim=-1)
                        sys1_conf = round(scores[one_id].item(), 4)
                        sys2_conf = round(scores[two_id].item(), 4)
                        sub_goal_confidence.append({
                            'sys1': sys1_conf,
                            'sys2': sys2_conf
                        })
                confidence.append(sub_goal_confidence)

    return predictions, confidence

def test_baseline_planner(tokenizer,
                          model,
                          batch_planner_prompts,
                          output
                          ):
    print(f'Planner batch prompts = {batch_planner_prompts}')
    predicted_plans, _ = get_predictions(tokenizer, model, batch_planner_prompts)
    print(f'Predicted plans = {predicted_plans}')

    validity, states_visited = eval_utils.eval_plans_task_level(batch_samples, predicted_plans)
    print(states_visited)
    output.extend([{
        'idx': test_sample.idx,
        'predicted_plan': predicted_plan,
        'num_blocks': len(test_sample.initial_state_as_blocks),
        'gold_plan': test_sample.system1_plan,
        'valid': valid,
        'states_visited': state_visited
    } for test_sample, predicted_plan, valid, state_visited in zip(batch_samples, predicted_plans, validity, states_visited)])

    return output, validity


def test_task_level_hybrid_planner(tokenizer,
                                   meta_planner,
                                   system1_planner,
                                   system2_planner,
                                   batch_meta_planner_prompts,
                                   batch_system1_planner_prompts,
                                   batch_system2_planner_prompts,
                                   output
                                   ):
    predicted_meta_plans, confidences = get_predictions(tokenizer, meta_planner, batch_meta_planner_prompts, with_confidence=True)
    predicted_plans_sys1, _ = get_predictions(tokenizer, system1_planner, batch_system1_planner_prompts)
    predicted_plans_sys2, _ = get_predictions(tokenizer, system2_planner, batch_system2_planner_prompts)
    
    print(f'Predicted meta plans = {predicted_meta_plans}')
    print(f'Predicted sys1 plans = {predicted_plans_sys1}')
    print(f'Predicted sys2 plans = {predicted_plans_sys2}')

    validity, states_visited = [], []
    for test_sample, confidence, predicted_plan_sys1, predicted_plan_sys2, predicted_meta_plan in zip(batch_samples, confidences, predicted_plans_sys1, predicted_plans_sys2, predicted_meta_plans):
        sample_validity_sys1, sample_states_visited_sys1 = eval_utils.eval_plans_task_level([test_sample], [predicted_plan_sys1])
        sample_validity_sys2, sample_states_visited_sys2 = eval_utils.eval_plans_task_level([test_sample], [predicted_plan_sys2])

        if confidence['sys1'] > confidence['sys2']:
            predicted_plan = predicted_plan_sys1
            valid = sample_validity_sys1[0]
            states = sample_states_visited_sys1[0]
        else:
            predicted_plan = predicted_plan_sys2
            valid = sample_validity_sys2[0]
            states = sample_states_visited_sys2[0]
        
        output.append({
                'idx': test_sample.idx,
                'gold_plan': test_sample.system1_plan,

                'confidence': confidence,
                'predicted_plan': predicted_plan,
                'predicted_plan_sys1': predicted_plan_sys1,
                'predicted_plan_sys2': predicted_plan_sys2,
                'predicted_meta_plan': predicted_meta_plan,

                'valid_sys1': sample_validity_sys1[0],
                'states_visited_sys1': sample_states_visited_sys1[0],

                'valid_sys2': sample_validity_sys2[0],
                'states_visited_sys2': sample_states_visited_sys2[0],

                'num_blocks': len(test_sample.initial_state_as_blocks),

                'valid': valid,
                'states_visited': states
                })
        
        validity.append(valid)
        states_visited.append(states)

    return output, validity

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/blocksworld', type=str)
    parser.add_argument('--output_dir', default='output', type=str)

    parser.add_argument('--meta_planner', default='models/blocksworld/direct_a_star_heuristic_new_sliding_sample_system_0.5_3000_epoch_1_lr_0.0005_bs_2_blocksworld_final', type=str)
    parser.add_argument('--system1_planner', default='models/blocksworld/a_star_heuristic_new_sliding_task_system_0.0_3000_epoch_1_lr_0.0005_bs_4_blocksworld_final', type=str)
    parser.add_argument('--system2_planner', default='models/blocksworld/a_star_heuristic_new_sliding_task_system_1.0_3000_epoch_3_lr_0.0005_bs_2_blocksworld_final', type=str)

    parser.add_argument('--task', default='blocksworld', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--cache_dir', default='cache', type=str)
    parser.add_argument('--max_new_tokens', default=5000, type=int)
    parser.add_argument('--k_shot', default=0, type=int)
    parser.add_argument('--method', choices=['heuristic'], default='heuristic', type=str)
    parser.add_argument('--level', choices=['task', 'sample'], default='sample', type=str)
    parser.add_argument('--system', choices=[0.0, 0.5, 1.0], default=0.5, type=float)
    parser.add_argument('--search_algo', choices=['a_star'], default='a_star', type=str)

    parser.add_argument('--test_file', default='test.json', type=str)
    parser.add_argument('--decomp_style', choices=['sliding'], default='sliding', type=str)

    args = parser.parse_args()

    print(f"=== evaluating {args.meta_planner} on {args.data_dir} dataset ===")

    # Output file
    meta_planner_name = args.meta_planner.split('/')[-1]
    output_file = os.path.join(args.output_dir, f'{args.task}_{meta_planner_name}.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.meta_planner, padding_side='left', add_eos_token=False)
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    # Loading all models
    print("Loading all models...")
    meta_planner = AutoModelForCausalLM.from_pretrained(
            args.meta_planner, 
            cache_dir=args.cache_dir if args.cache_dir else 'cache', 
            device_map='auto'
            )
    meta_planner.resize_token_embeddings(len(tokenizer))

    # If it is a full system 1 or system 2, meta-planner is the planner
    # Or, if it is with implicit decomposition, meta-planner is the planner
    if args.system not in [0.0, 1.0]:
        system1_planner = AutoModelForCausalLM.from_pretrained(
            args.system1_planner, 
            cache_dir=args.cache_dir if args.cache_dir else 'cache', 
            device_map='auto'
            )
        system1_planner.resize_token_embeddings(len(tokenizer))
        
        system2_planner = AutoModelForCausalLM.from_pretrained(
            args.system2_planner, 
            cache_dir=args.cache_dir if args.cache_dir else 'cache', 
            device_map='auto'
            )
        system2_planner.resize_token_embeddings(len(tokenizer))

    print("All models loaded!")
    
    valid_plans, output = 0, []
    if args.level == "task":
        meta_planner_prompts, system1_planner_prompts, system2_planner_prompts, test_samples = data_utils.get_test_prompts(args, test_file=args.test_file)
        for idx in range(0, len(meta_planner_prompts), args.batch_size):
            print(f'Index = {idx}')
            batch_samples = test_samples[idx : idx+args.batch_size]

            if args.system in [0.0, 1.0]:
                batch_planner_prompts = meta_planner_prompts[idx : idx+args.batch_size]
                output, validity = test_baseline_planner(tokenizer,
                                               meta_planner,
                                               batch_planner_prompts,
                                               output)
            else:
                batch_meta_planner_prompts = meta_planner_prompts[idx : idx+args.batch_size]
                batch_system1_planner_prompts = system1_planner_prompts[idx : idx+args.batch_size]
                batch_system2_planner_prompts = system2_planner_prompts[idx : idx+args.batch_size]

                print(f'Meta-planner batch prompts = {batch_meta_planner_prompts}')
                print(f'System1 batch prompts = {batch_system1_planner_prompts}')
                print(f'System2 batch prompts = {batch_system2_planner_prompts}')
                output, validity = test_task_level_hybrid_planner(tokenizer,
                                                        meta_planner,
                                                        system1_planner,
                                                        system2_planner,
                                                        batch_meta_planner_prompts,
                                                        batch_system1_planner_prompts,
                                                        batch_system2_planner_prompts,
                                                        output)

            with open(output_file, 'w') as f:
                json.dump(output, f, indent=4)

            valid_plans += np.count_nonzero(validity)

            eval_utils.compute_metrics(valid_plans, output)

    elif args.level == "sample":
        meta_planner_prompts, test_samples, all_silver_sub_goals = decomp_hybrid.get_meta_planner_test_prompts(args, test_file=args.test_file)

        for i, (meta_planner_prompt, test_sample, silver_sub_goals) in enumerate(zip(meta_planner_prompts, test_samples, all_silver_sub_goals)):
            print(f'Index = {i}')
            print(f'Meta-planner prompt = {meta_planner_prompt}')

            predicted_meta_plan, confidences = get_predictions(tokenizer, meta_planner, [meta_planner_prompt], with_confidence=True, intra_controller=True)
            predicted_meta_plan = predicted_meta_plan[0]
            print(f'Predicted meta plan = {predicted_meta_plan}')

            sub_goals = decomp_hybrid.parse_meta_plan(predicted_meta_plan, test_sample.initial_state, test_sample.goal_state)
            print(f'Sub-goals = {sub_goals}')

            predicted_plan, predicted_sys1_plan, predicted_sys2_plan = [], [], []
            for sub_goal in sub_goals:
                planner_prompt = decomp_hybrid.get_planner_prompt(test_sample, sub_goal[0], sub_goal[1])
                if sub_goal[2] == 1:
                    # Get results using both, so that we can rank later on
                    predicted_sub_plan, _ = get_predictions(tokenizer, system1_planner, [planner_prompt])
                    predicted_sub_plan = predicted_sub_plan[0]

                    predicted_sys2_sub_plan, _ = get_predictions(tokenizer, system2_planner, [planner_prompt])
                    predicted_sys2_sub_plan = predicted_sys2_sub_plan[0]

                    predicted_sys1_plan.append(predicted_sub_plan)
                    predicted_sys2_plan.append(predicted_sys2_sub_plan)
                else:
                    predicted_sub_plan, _ = get_predictions(tokenizer, system2_planner, [planner_prompt])
                    predicted_sub_plan = predicted_sub_plan[0]

                    predicted_sys1_sub_plan, _ = get_predictions(tokenizer, system1_planner, [planner_prompt])
                    predicted_sys1_sub_plan = predicted_sys1_sub_plan[0]

                    predicted_sys1_plan.append(predicted_sys1_sub_plan)
                    predicted_sys2_plan.append(predicted_sub_plan)
                
                print(f'Predicted sub plan = {predicted_sub_plan}')
                predicted_plan.append(predicted_sub_plan)

            validity, combined_plan, states_visited = eval_utils.eval_plan_sample_level(test_sample, predicted_plan)

            output.append({
                'idx': test_sample.idx,
                'num_blocks': len(test_sample.initial_state_as_blocks),
                'predicted_meta_plan': predicted_meta_plan,
                'predicted_sub_goals': sub_goals,
                'sub_goal_confidence': confidences[0],
                'predicted_sys1_sub_plan': predicted_sys1_plan,
                'predicted_sys2_sub_plan': predicted_sys2_plan,
                'silver_sub_goals': silver_sub_goals,
                'predicted_plan': predicted_plan,
                'combined_plan': combined_plan,
                'gold_plan': test_sample.system1_plan,
                'valid': validity,
                'states_visited': states_visited
            })

            with open(output_file, 'w') as f:
                json.dump(output, f, indent=4)

            valid_plans += np.count_nonzero(validity)

            eval_utils.compute_metrics(valid_plans, output)





    



    