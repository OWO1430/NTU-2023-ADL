# ADL
B09611048 蘇蓁葳
## Training environment
Google Colab: A100 or V100
## Multiple question
1. Load ADL_multiple_question.ipynb to Google Colab
2. Mount the Colab notebook to my drive
3. Go into the directory ADL in my Google drive
    ```
    /content/drive/MyDrive/ADL/HW3
    ```
4. Install requirements library
    ```
    !pip install bitsandbytes==0.40.0
    !pip install transformers==4.31.0
    !pip install peft==0.4.0
    !pip install accelerate==0.21.0
    !pip install einops==0.6.1
    !pip install evaluate==0.4.0
    !pip install scikit-learn==1.2.2
    !pip install sentencepiece==0.1.99
    !pip install wandb==0.15.3
    ```
5. Load context.json into the program
6. Adjust `args` and `training_args` to set parameters for the training
    ```python
        args.__dict__.update({
        "model_name_or_path": "/content/drive/MyDrive/ADL/HW3/Taiwan-LLM-7B-v2.0-chat",
        "output_dir": "/content/drive/MyDrive/ADL/HW3/model_3000_data",
        "dataset":"/content/drive/MyDrive/ADL/HW3/data/train.json",
        "do_eval":True,
        "evaluation_strategy": "steps",
        "eval_steps": 50,
        "lora_r":4,
        "max_steps":500,
        "max_train_samples":3000,
        "max_eval_samples":750
    })

    training_args.__dict__.update({
        "max_steps":500,
        "output_dir": "/content/drive/MyDrive/ADL/HW3/model_3000_data",
    })
    ```
7. Run the main program and wait for the result! The trained model will be stored at `output_dir`

