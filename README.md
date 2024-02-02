# ADL
B09611048 蘇蓁葳
## Training environment
Google Colab: A100 or V100
## Multiple question
1. Load ADL_multiple_question.ipynb to Google Colab
2. Mount the Colab notebook to my drive
3. Go into the directory ADL in my Google drive
    ```
    /content/drive/MyDrive/ADL/HW1
    ```
4. Install requirements library
    ```
    !pip install datasets
    !pip install evaluate
    !pip install accelerate
    !pip install huggingface_hub
    !pip install transformers
    ```
5. Load context.json into the program
6. Adjust `arg_string` to set parameters for the training
    ```python
        args_string = '''--model_name_or_path ckiplab/bert-tiny-chinese
                        --learning_rate 2e-5 --num_train_epochs 1
                        --output_dir /content/drive/MyDrive/ADL/HW1/swag/chinese-macbert-tiny/
                        --gradient_accumulation_steps 2
                        --per_device_train_batch_size 2
                        --train_file train.json
                        --max_seq_length 512
                        --validation_file valid.json'''
    ```
7. Run the main program and wait for the result! The trained model will be stored at `output_dir` in `args_string`

## Question Answering
1. Load ADL_multiple_question.ipynb to Google Colab
2. Mount the Colab notebook to my drive
3. Go into the directory ADL in my Google drive
    ```
    /content/drive/MyDrive/ADL/HW1
    ```
4. Install requirements library
    ```
    !pip install datasets
    !pip install evaluate
    !pip install accelerate
    !pip install huggingface_hub
    !pip install transformers
    ```
5. Load context.json into the program
6. Load post_processing function for the main program
6. Adjust `arg_string` to set parameters for the training
    ```python
        args_string = '''--model_name_or_path ckiplab/bert-tiny-chinese
                        --learning_rate 2e-5 --num_train_epochs 1
                        --output_dir /content/drive/MyDrive/ADL/HW1/swag/chinese-macbert-tiny/
                        --gradient_accumulation_steps 2
                        --per_device_train_batch_size 2
                        --train_file train.json
                        --max_seq_length 512
                        --validation_file valid.json'''
    ```
7. Run the main program and wait for the result! The trained model will be stored at `output_dir` in `args_string`

