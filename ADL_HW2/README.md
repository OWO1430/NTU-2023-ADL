# ADL
B09611048 蘇蓁葳
## Training environment
Google Colab: T4 or V100
## Multiple question
1. Load ADL_multiple_question.ipynb to Google Colab
2. Mount the Colab notebook to my drive
3. Go into the directory ADL in my Google drive
    ```
    /content/drive/MyDrive/ADL/HW2
    ```
4. Install requirements library
    ```
    !pip install datasets
    !pip install evaluate
    !pip install accelerate
    !pip install huggingface_hub
    !pip install rouge_score
    ```
    ```
    !pip install --no-cache-dir transformers sentencepiece
    ```
    ```
    !pip install ckiptagger
    !pip install rouge
    ```
5. Load ckiptagger and the function get_rouge to the program, and set your choptagger data dir.
    ``` python
    download_dir = '/content/drive/MyDrive/ADL/HW2/ckiptagger'
    data_dir = '/content/drive/MyDrive/ADL/HW2/ckiptagger/data'
    ```

6. Adjust `arg_string` to set parameters for the training
    ```python
    args_string = '''--model_name_or_path /content/drive/MyDrive/ADL/HW2/epoch1/
                    --learning_rate 2.5e-5 --num_train_epochs 3
                    --output_dir /content/drive/MyDrive/ADL/HW2/test_1/
                    --gradient_accumulation_steps 2
                    --per_device_eval_batch_size 4
                    --per_device_train_batch_size 1
                    --train_file train_short.jsonl
                    --validation_file train_short.jsonl
                    --num_beams 5'''
    ```

