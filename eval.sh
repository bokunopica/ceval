python code/evaluator_series/eval.py \
    --model_name 'finetuned_chatglm' \
    --ckpt_path '/home/qianq/mycodes/VisualGLM-6B/checkpoints/2-gpus/finetune-visualglm-6b-ptuning-pre-128' \
    --cuda_device '1' \
    --few_shot \
    --subject 'accountant' \
    --quant 8
    # --cot

