python3 inference.py  \
--model_name_or_path ./HW2_final/summarization \
--output_dir ${2} \
--gradient_accumulation_steps 2 \
--per_device_eval_batch_size 4 \
--validation_file ${1} \
--num_beams 5 \