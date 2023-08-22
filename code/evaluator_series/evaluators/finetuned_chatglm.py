import os
import re
import argparse
import json
from tqdm import tqdm
from copy import deepcopy
import torch
from torch import nn
from sat.model.finetune import PTuningV2Mixin
from sat.model.finetune.lora2 import LoraMixin
from sat.model.finetune import AdapterMixin
from sat.model.base_model import non_conflict
from transformers import AutoTokenizer, AutoModel
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from evaluators.evaluator import Evaluator

from .model import VisualGLMModel




class AdjustAdapterMixin(AdapterMixin):
    @non_conflict
    def layer_forward(self, hidden_states, mask, old_impl, *args, **kw_args):
        """
        hidden_states: [batch, seq_len, hidden_size]
        mask: [(1, 1), seq_len, seq_len]
        """
        layer = self.transformer.layers[kw_args["layer_id"]]
        # Layer norm at the begining of the transformer layer.
        hidden_states = layer.input_layernorm(hidden_states)
        # Self attention.
        attention_output = layer.attention(hidden_states, mask, **kw_args)

        attention_output = attention_output + self.ff2[kw_args["layer_id"]](
            nn.functional.gelu(self.ff1[kw_args["layer_id"]](attention_output))
        )

        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = layer.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = layer.mlp(layernorm_output, **kw_args)
        mlp_output = mlp_output + self.ff4[kw_args["layer_id"]](
            nn.functional.gelu(self.ff3[kw_args["layer_id"]](mlp_output))
        )

        # Second residual connection.
        output = layernorm_output + mlp_output

        return output



class FineTuneVisualGLMModel(VisualGLMModel):
    def __init__(self, args, transformer=None, parallel_output=True, **kw_args):
        super().__init__(
            args, transformer=transformer, parallel_output=parallel_output, **kw_args
        )
        if args.use_ptuning:
            self.add_mixin(
                "ptuning",
                PTuningV2Mixin(
                    args.num_layers,
                    args.hidden_size // args.num_attention_heads,
                    args.num_attention_heads,
                    args.pre_seq_len,
                ),
            )
        if args.use_lora:
            self.add_mixin(
                "lora",
                LoraMixin(
                    args.num_layers,
                    args.lora_rank,
                    layer_range=args.layer_range,
                ),
                reinit=True,
            )
            # self.get_mixin("eva").model.glm_proj = replace_linear_with_lora(self.get_mixin("eva").model.glm_proj, LoraLinear, args.lora_rank)
        elif args.use_qlora:
            self.add_mixin(
                "lora",
                LoraMixin(
                    args.num_layers,
                    args.lora_rank,
                    layer_range=args.layer_range,
                    qlora=True,
                ),
                reinit=True,
            )
        elif args.use_adapter:
            # adapter finetune
            self.add_mixin(
                "adapter",
                AdjustAdapterMixin(
                    num_layers=args.num_layers, # 28-transformer一致
                    hidden_size=args.hidden_size, # 4096
                    adapter_hidden=args.adapter_hidden, # specified in .sh
                ),
            )
            pass
        self.args = args

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group(
            "VisualGLM-finetune", "VisualGLM finetune Configurations"
        )
        group.add_argument("--pre_seq_len", type=int, default=8)
        group.add_argument("--lora_rank", type=int, default=10)
        group.add_argument("--use_ptuning", action="store_true")
        group.add_argument("--use_lora", action="store_true")
        group.add_argument("--use_qlora", action="store_true")
        group.add_argument("--layer_range", nargs="+", type=int, default=None)
        group.add_argument("--use_adapter", action="store_true")
        group.add_argument("--adapter_hidden", type=int, default=128)
        group.add_argument("--adapter_num_layers", type=int, default=28)
        group.add_argument("--use_freeze", action="store_true")
        group.add_argument("--unfreeze_layers", type=str, default="")
        return super().add_model_specific_args(parser)

    def disable_untrainable_params(self):
        enable = []
        if self.args.use_ptuning:
            enable.extend(["ptuning"])
        if self.args.use_lora or self.args.use_qlora:
            enable.extend(["matrix_A", "matrix_B"])
        if self.args.use_freeze:
            unfreeze_layers = self.args.unfreeze_layers.split(',')
        else:
            unfreeze_layers = []
        print('------------unfreeze_layer--------------')
        for n, p in self.named_parameters():
            flag = False
            # adapter unfreeze
            if self.args.use_adapter and n.startswith("mixins.adapter"):
                flag = True
            elif self.args.use_freeze:
                for unfreeze_layer in unfreeze_layers:
                    if n.startswith(f"transformer.layers.{unfreeze_layer}."):
                        flag = True
                        break
            else:
                for e in enable:
                    if e.lower() in n.lower():
                        flag = True
                        break
            if not flag:
                p.requires_grad_(False)
            else:
                print(n)
        print('------------unfreeze_layer--------------')




class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores

class FinetunedChatGLM_Evaluator(Evaluator):
    def __init__(self, choices, k, model_name, device, ckpt_path, quant):
        super(FinetunedChatGLM_Evaluator, self).__init__(choices, model_name, k)
        # try adding 'mirror="tuna"' and 'resume_download=True' if facing the 'read timed out' problem
        # or directly clone the model
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, mirror="tuna")
        model, model_args = FineTuneVisualGLMModel.from_pretrained(
            ckpt_path,
            args=argparse.Namespace(
                fp16=True,
                skip_init=True,
                use_gpu_initialization=True if (torch.cuda.is_available() and quant is None) else False,
                device=device if (torch.cuda.is_available() and quant is None) else 'cpu',
            )
        )
        model = model.eval()
        self.model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, mirror="tuna", resume_download=True).half().to(device)

    def eval_subject(self, subject_name, test_df, dev_df=None, few_shot=False, cot=False, save_result_dir=None):
        correct_num = 0
        if save_result_dir:
            if few_shot:
                result = []
            score = []
        if few_shot:
            history = self.generate_few_shot_prompt(subject_name, dev_df, cot=cot)
        else:
            history = []
        answers = list(test_df['answer'])
        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = self.format_example(row, include_answer=False, cot=cot)
            if few_shot:
                response, _ = self.model.chat(self.tokenizer, question, do_sample=False, history=history)
                response = response.strip()
                # For ChatGLM, we use answer extraction in answer-only mode too.
                ans, direct_extract = self.extract_cot_answer(row, response)
            else:   # zero-shot by extracting answer from distribution
                ans = self.generate_dist(self.model, self.tokenizer, question, do_sample=False, max_length=2048, history=history)
            if ans == answers[row_index]:
                correct_num += 1
                correct = 1
            else:
                correct = 0
            if save_result_dir:
                if few_shot:
                    result.append(response)
                score.append(correct)
        correct_ratio = 100*correct_num/len(answers)
        
        if save_result_dir:
            if few_shot:
                test_df['model_output'] = result
            test_df['correctness'] = score
            test_df.to_csv(os.path.join(save_result_dir, f'{subject_name}_test.csv'))

        return correct_ratio
    
    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        message = []
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        message.append(self.format_example(dev_df.iloc[0, :], cot=cot, add_prompt=f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"))
        for i in range(1, k):
            message.append(self.format_example(dev_df.iloc[i, :], cot=cot))
        return message
        
    def format_example(self, line, include_answer=True, cot=False, add_prompt=''):
        example = add_prompt + line['question']
        # print(example)
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'
        example += '\n答案：'
        if include_answer:
            if cot:
                ans = "让我们一步一步思考，\n" + line["explanation"] + f"\n所以答案是{line['answer']}。"
            else:
                ans = line["answer"]
            m = (example, ans)
            return m
        return example
    
    def extract_cot_answer(self, line, gen_ans):
        m = re.findall(r'所以答案是(.+?)。', gen_ans, re.M)
        if len(m) > 0 and m[-1] in self.choices:
            return m[-1], True
        answer_patterns = [
            r'([ABCD])是正确的',
            r'选项([ABCD])正确',
            r'答案为([ABCD])',
            r'答案是([ABCD])',
            r'答案([ABCD])',
            r'选择([ABCD])',
            r'答案：([ABCD])',
            r'选择答案([ABCD])'
        ]
        # RE extraction
        for answer_pattern in answer_patterns:
            m = re.search(answer_pattern, gen_ans, re.M)
            if m:
                answer = m.group(1)
                return answer, False
        # only containing one choice-character
        m = re.findall(r'[ABCD]', gen_ans, re.M)
        if len(m) == 1:
            answer = m[0]
            return answer, False
        answer_word_counter = 0
        # only containing one choice-context
        for c in self.choices:
            if str(line[f'{c}']) in gen_ans:
                answer = c
                answer_word_counter += 1
        if answer_word_counter == 1:
            return answer, False
        return '-', False
    
    def generate_dist(self, model, tokenizer, query, history, num_beams=1, max_length=2048,
                      do_sample=False, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"num_beams": num_beams, "do_sample": do_sample, "top_p": top_p, "max_length": 2048,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(model.device)
        outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True, **gen_kwargs)
        
        score = outputs.scores[0][0].tolist()
        choice_score = [score[167], score[333], score[251], score[416]]
        ranked_index = [index for index, value in sorted(list(enumerate(choice_score)), key=lambda x:x[1], reverse=True)]
        return self.choices[ranked_index[0]]
