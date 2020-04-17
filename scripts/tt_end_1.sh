echo "Eval 1.."

python /N/u/anakuzne/Carbonate/curr_learning/DeepSpeech/evaluate.py -W ignore --test_files='/N/slate/anakuzne/tatar/clips/test.csv' --test_batch_size 32 --checkpoint_dir='/N/slate/anakuzne/tt_ckpt_automated_curr/' --test_output_file='/N/u/anakuzne/Carbonate/curr_learning/automated_curr/loss_before.json' --report_count 4700

echo "Training on a batch"
python /N/u/anakuzne/Carbonate/curr_learning/DeepSpeech/DeepSpeech.py -W ignore --train_files='/N/slate/anakuzne/tatar/clips/batch.csv' --dev_files='/N/slate/anakuzne/tatar/clips/dev.csv' --alphabet_config_path='/N/slate/anakuzne/tatar/tt_alphabet.txt' --train_batch_size 64 --dev_batch_size 64 --dropout_rate 0.15 --lm_binary_path='/N/slate/anakuzne/tatar/tt_lm.binary' --lm_trie_path='/N/slate/anakuzne/tatar/tt_trie' --learning_rate 0.00001 --epochs 1 --export_dir='/N/slate/anakuzne/tt_automated_curr/' --checkpoint_dir='/N/slate/anakuzne/tt_ckpt_automated_curr/' --source_model_checkpoint_dir='/N/slate/anakuzne/deepspeech_released/ckpt_pretrained/' --drop_source_layers 2

echo "Eval 2"
python /N/u/anakuzne/Carbonate/curr_learning/DeepSpeech/evaluate.py -W ignore --test_files='/N/slate/anakuzne/tatar/clips/test.csv' --test_batch_size 32 --checkpoint_dir='/N/slate/anakuzne/tt_ckpt_automated_curr/' --test_output_file='/N/u/anakuzne/Carbonate/curr_learning/automated_curr/loss_after.json' --report_count 4700