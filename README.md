# GMLLM-SRR
Cross-Attention Fusion of Graph and MLLM for Social Relation Recognition

## Environment

Please refer to the "environment.txt".

## Dataset
[PISC](https://zenodo.org/record/1059155#.WznPu_F97CI) was released by [[Li et al. ICCV 2017](https://arxiv.org/abs/1708.00634)]. It involves a two-level relationship, i.e., coarse-level relationships with 3 categories and fine-level relationships with 6 categories.

[PIPA-relation](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/human-activity-recognition/social-relation-recognition/) was released by [[Sun et al. CVPR 2017](https://arxiv.org/abs/1704.06456)]. It covers 5 social domains, which can be further divided into 16 social relationships. On this dataset, we focus on the 16 social relationships.

## Usage
    optional arguments:
      -h, --help                show this help message and exit
      -j N, --workers N         number of data loading workers (defult: 4)
      -b N, --batch-size N      mini-batch size (default: 1)
      --print-freq N, -p N      print frequency (default: 10)
      --weights PATH            path to weights (default: none)
      --scale-size SCALE_SIZE   input size
      --world-size WORLD_SIZE   number of distributed processes
      -n N, --num-classes N     number of classes / categories
      --write-out               write scores
      --adjacency-matrix PATH   path to adjacency-matrix of graph
      --crop-size CROP_SIZE     crop size
      --result-path PATH        path for saving result (default: none)

## Prompt templates
    def fine_grained_generation(image_path, content1, content2):
    return {
              "messages": [
                {
                  "role": "user",
                  "content": "How many different kinds of fine-grained social relationships are in the image?",
                  "image": f"{image_path}"
                },
                {
                  "role": "assistant",
                  "content": f"{content1}"
                },
                {
                  "role": "user",
                  "content": "What is the fine-grained social relationship between each person in the image?"
                },
                {
                  "role": "assistant",
                  "content": f"{content2}"
                }
              ]
            }


    def coarse_grained_generation(image_path, content1, content2):
        return {
                  "messages": [
                    {
                      "role": "user",
                      "content": "How many different kinds of coarse-grained social relationships are in the image?",
                      "image": f"{image_path}"
                    },
                    {
                      "role": "assistant",
                      "content": f"{content1}"
                    },
                    {
                      "role": "user",
                      "content": "What is the coarse-grained social relationship between each person in the image?"
                    },
                    {
                      "role": "assistant",
                      "content": f"{content2}"
                    }
                  ]
                }
    

## Fine-tuning command
    CUDA_VISIBLE_DEVICES=0 swift sft \
        --model_id_or_path /your_path/glm-4v-9b \
        --model_type glm4v-9b-chat \
        --dataset /your_path/SRR_train.json \
        --num_train_epochs 5 \
        --sft_type lora \
        --output_dir /your_path/finetune_output \
        --eval_steps 300 \
        --batch_size 1 \
        --max_length 2048 \
        --lora_rank 8 \
        --lora_alpha 32 \
        --lora_dropout_p 0.05 \
        --gradient_checkpointing true \
        --weight_decay 0.1 \
        --learning_rate 1e-4 \
        --gradient_accumulation_steps $(expr 16 / 3) \
        --max_grad_norm 0.5 \
        --warmup_ratio 0.03 \
        --save_steps 300 \
        --save_total_limit 2


## Contributing
For any questions, feel free to open an issue or contact us (tangwang@stu.scu.edu.cn)
