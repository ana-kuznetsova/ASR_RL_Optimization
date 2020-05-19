## DeepSpeech
To run the experiments end-to-end ASR system needs to be installed. I trained the models on IU's Carbonate deep learning partition since training takes a lot of time and resources. 

We need a specific branch of **DeepSpeech** to run the experiments:

```bash
git clone https://github.com/mozilla/DeepSpeech.git
git checkout transfer-learning2
```
### Installation
```bash
python3 -m venv deepspeech-transfer

source $HOME/tmp/deepspeech-transfer/bin/activate

cd DeepSpeech

pip install -r requirements.txt
```

**Note:** Make sure Tensorflow GPU is installed: `tensorflow-gpu == 1.13.1`.

### CTC Decoder Installation
```bash
cd DeepSpeech/native_client/ctcdecode
python setup.py install
```

Additionaly you will need to download [checkpoints](https://github.com/mozilla/DeepSpeech/releases/tag/v0.5.0) for a pretrained English language model of `version 0.5.0`.

## Training data and files

Training data is stored in `data/` inside the project submission directory. It includes:

* Audio `clips/` directory
* Inside this directory we have: `train.csv`, `dev.csv` and `test.csv` files. In the same directory `.sh` scripts for server jobs store files `batch.csv` for training the pipeline. It is important so that the batches are saved in the `clips/` directory.
* Language model file `tt_lm.binary`
* Alphabet file `tt_alphabet.txt`
* Trie file `tt_trie`

## Scripts

The `scripts/` directory contains all the `bash` scripts to run **DeepSpeech** jobs either on the server or locally. Unfortunately there is no other way but change all of the absolute directories that **DeepSpeech** uses inside the `.sh` scripts since it is just a research code and there was not enough time to wrap it all into the proper `util` functions.

For example in `tt_init_1.sh`:
```bash
--train_files='/N/slate/anakuzne/tatar/clips/batch.csv'
```
should be changed to 
```bash
--train_files='/data/clips/batch.csv'
```
Source model and checkpoint directories also should be adjusted to the machine where you run the code. 

Checkpoint directories are the arguments for `DeepSpeech.py`. 

`--export_dir` path to the directory where the model is stored.

`--checkpoint_dir` path to the directory where the checkpoints for the current model will be stored.

**Important:**


`--source_model_checkpoint_dir` is the path to the pretrained English model's checkpoints.


## Running experiments

To run the experiments go to the `code/` directory.
There are two data files: `tt_train_with_scores.csv` for ranking the data according to compression ration and `train.csv` (same as in the `data/clips/` directory, it is here for convenience).

There are three modes in which we can run `main.py` `UCB1`, `EXP3` and `LinUCB`. Below is the example command to run the code with `UCB1` algorithm with tunable parameters.

```bash
python main.py --mode='UCB1' --df_path='tt_train_with_scores.csv' --num_tasks=3 --csv='train.csv' --num_episodes=15 --batch_size=64 --gain_type='SPG' --c 0.01
```

For more argument specification run
```bash
python main.py --help
```

Training results are stored in `.pickle` files for each algorithm:

* `sc_reward_hist_{algo name}.pickle` for the history of scaled rewards;
* `loss_hist_{algo name}.pickle` for saving losses;
* `action_hist_{algo name}.pickle` for action history.

Additionally, there are running log prints in every algorithm where the current Q-function or weights are printed. 

**NOTE:** there could be issues with the **DeepSpeech** installation. If something emerges, do not hesitate to message me anakuzne@iu.edu.