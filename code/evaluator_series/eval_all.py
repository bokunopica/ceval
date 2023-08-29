import os
import argparse
import pandas as pd
import torch
from evaluators.chatgpt import ChatGPT_Evaluator
from evaluators.moss import Moss_Evaluator
from evaluators.chatglm import ChatGLM_Evaluator
from evaluators.minimax import MiniMax_Evaluator
from evaluators.finetuned_chatglm import FinetunedChatGLM_Evaluator

import time
choices = ["A", "B", "C", "D"]

subject_list = [
    'accountant',
    'advanced_mathematics',
    'art_studies',
    'basic_medicine',
    'business_administration',
    'chinese_language_and_literature',
    'civil_servant',
    'clinical_medicine',
    'college_chemistry',
    'college_economics',
    'college_physics',
    'college_programming',
    'computer_architecture',
    'computer_network',
    'discrete_mathematics',
    'education_science',
    'electrical_engineer',
    'environmental_impact_assessment_engineer',
    'fire_engineer',
    'high_school_biology',
    'high_school_chemistry',
    'high_school_chinese',
    'high_school_geography',
    'high_school_history',
    'high_school_mathematics',
    'high_school_physics',
    'high_school_politics',
    'ideological_and_moral_cultivation',
    'law',
    'legal_professional',
    'logic',
    'mao_zedong_thought',
    'marxism',
    'metrology_engineer',
    'middle_school_biology',
    'middle_school_chemistry',
    'middle_school_geography',
    'middle_school_history',
    'middle_school_mathematics',
    'middle_school_physics',
    'middle_school_politics',
    'modern_chinese_history',
    'operating_system',
    'physician',
    'plant_protection',
    'probability_and_statistics',
    'professional_tour_guide',
    'sports_science',
    'tax_accountant',
    'teacher_qualification',
    'urban_and_rural_planner',
    'veterinary_medicine',
]

def main(args):

    if "turbo" in args.model_name or "gpt-4" in args.model_name:
        evaluator=ChatGPT_Evaluator(
            choices=choices,
            k=args.ntrain,
            api_key=args.openai_key,
            model_name=args.model_name
        )
    elif "moss" in args.model_name:
        evaluator=Moss_Evaluator(
            choices=choices,
            k=args.ntrain,
            model_name=args.model_name
        )
    elif "chatglm" in args.model_name:
        if args.cuda_device:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
        device = torch.device(f"cuda:{str(args.cuda_device)}")
        if args.model_name.startswith('finetuned'):
            evaluator=FinetunedChatGLM_Evaluator(
                choices=choices,
                k=args.ntrain,
                model_name=args.model_name,
                device=device,
                args=args
            )
        else:
            evaluator=ChatGLM_Evaluator(
                choices=choices,
                k=args.ntrain,
                model_name=args.model_name,
                device=device
            )
    elif "minimax" in args.model_name:
        evaluator=MiniMax_Evaluator(
            choices=choices,
            k=args.ntrain,
            group_id=args.minimax_group_id,
            api_key=args.minimax_key,
            model_name=args.model_name
        )
    else:
        print("Unknown model name")
        return -1

    result_list = []
    for subject_name in subject_list:
    # subject_name=args.subject
        if not os.path.exists(r"logs"):
            os.mkdir(r"logs")
        run_date=time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
        save_result_dir=os.path.join(r"logs",f"{args.model_name}_{run_date}")
        os.mkdir(save_result_dir)
        print(subject_name)
        val_file_path=os.path.join('data/val',f'{subject_name}_val.csv')
        val_df=pd.read_csv(val_file_path)
        if args.few_shot:
            dev_file_path=os.path.join('data/dev',f'{subject_name}_dev.csv')
            dev_df=pd.read_csv(dev_file_path)
            correct_ratio = evaluator.eval_subject(subject_name, val_df, dev_df, few_shot=args.few_shot,save_result_dir=save_result_dir,cot=args.cot)
        else:
            correct_ratio = evaluator.eval_subject(subject_name, val_df, few_shot=args.few_shot,save_result_dir=save_result_dir)
        print("Acc:",correct_ratio)
        result_list.append([
            subject_name, 
            correct_ratio
        ])
        break
    result_df = pd.DataFrame(result_list)
    result_df.columns = ["subject_name", "correct_ratio"]
    result_df.to_csv(f"{save_result_dir}/results.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--openai_key", type=str,default="xxx")
    parser.add_argument("--minimax_group_id", type=str,default="xxx")
    parser.add_argument("--minimax_key", type=str,default="xxx")
    parser.add_argument("--few_shot", action="store_true") # few shot提示
    parser.add_argument("--model_name",type=str)
    parser.add_argument("--cot",action="store_true") # chain of thought
    # parser.add_argument("--subject","-s",type=str,default="operating_system")
    parser.add_argument("--cuda_device", type=int, default=0)
    # Model Args
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None)
    parser.add_argument("--ckpt_path", type=str)
    args = parser.parse_args()
    main(args)