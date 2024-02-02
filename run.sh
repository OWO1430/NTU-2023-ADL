python multiple_choice.py \
--model_name_or_path ./HW1_final/multiple_choice \
--output_dir ./ \
--context_file ${1} \
--test_file ${2}\

python QA.py \
--model_name_or_path ./HW1_final/QA \
--output_dir ${3} \
--context_file ${1} \
--validation_file data.json \

