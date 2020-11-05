echo "Calculating L1.."

python /N/u/anakuzne/Carbonate/curr_learning/DeepSpeech/evaluate.py -W ignore --test_files='/N/slate/ak16/Basque/cv-corpus-5.1-2020-06-22/eu/clips/batch_exp3.csv' --test_batch_size 32 --checkpoint_dir='/N/slate/ak16/eu_ckpt_automated_curr/main_model/' --alphabet_config_path='/N/slate/ak16/Basque/cv-corpus-5.1-2020-06-22/eu/eu_alphabet.txt' --test_output_file='/N/u/ak16/Carbonate/curr_learning/automated_curr/loss_before.json' --report_count 4700

echo "Training on a batch..."

python /N/u/ak16/Carbonate/DeepSpeech/DeepSpeech.py -W ignore --train_files='/N/slate/ak16/Basque/cv-corpus-5.1-2020-06-22/eu/clips/batch.csv' --alphabet_config_path='/N/slate/ak16/Basque/cv-corpus-5.1-2020-06-22/eu/eu_alphabet.txt' --train_batch_size 64 --dropout_rate 0.15  --learning_rate 0.0001 --epochs 1 --export_dir='/N/slate/ak16/eu_automated_curr/' --checkpoint_dir='/N/slate/ak16/eu_ckpt_automated_curr/main_model/' --source_model_checkpoint_dir='/N/slate/ak16/deepspeech_released/ckpt_pretrained/' --drop_source_layers 2

python /N/u/anakuzne/Carbonate/curr_learning/DeepSpeech/evaluate.py -W ignore --test_files='/N/slate/ak16/Basque/cv-corpus-5.1-2020-06-22/eu/clips/batch.csv' --test_batch_size 32 --checkpoint_dir='/N/slate/ak16/eu_ckpt_automated_curr/main_model_exp3/' --alphabet_config_path='/N/slate/ak16/Basque/cv-corpus-5.1-2020-06-22/eu/eu_alphabet.txt' --test_output_file='/N/u/ak16/Carbonate/curr_learning/automated_curr/loss_after.json' --report_count 4700
