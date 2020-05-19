echo "Running validation..."

python /N/u/anakuzne/Carbonate/curr_learning/DeepSpeech/evaluate.py -W ignore --test_files='/N/slate/anakuzne/tatar/clips/dev.csv' --test_batch_size 64 --checkpoint_dir='/N/slate/anakuzne/tt_ckpt_automated_curr/main_model/' --alphabet_config_path='/N/slate/anakuzne/tatar/tt_alphabet.txt' --test_output_file='/N/u/anakuzne/Carbonate/curr_learning/automated_curr/validation_loss.json' --lm_binary_path='/N/slate/anakuzne/tatar/tt_lm.binary' --lm_trie_path='/N/slate/anakuzne/tatar/tt_trie' --report_count 4700
