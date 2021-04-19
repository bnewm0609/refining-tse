# Refining Targeted Syntactic Evaluation of Language Models

Benjamin Newman, Kai-Siang Ang, Julia Gong, and John Hewitt

## Getting Started

To get started, set up your environment:
```
conda env create -f environment.yml
```
Run a small experiment:
```
python run.py configs/bert-base-cased/ML_simple_agrmt/mw.yaml
```
You should see results printed to the console, as well as some logs
in the `results` folder.

To reproduce all the experiments from the paper:
```
python run_all.py configs --whitelist bert-large-cased,bert-large-uncased,gpt2-xl,roberta-large
```

And to generate the main plots and table:
```
python src/plots.py
```


## Code Structure

Recommended exploration:
- Begin in `run.py`, follow it to `src/experiment.py`,
then explore the `src/dataset`, `src/models` and `src/metrics` folders.
- Examine some `.yaml` file in `configs`.
- Try running an experiment and inspecting the `results`.

Your own extensions:
- Extend `TransformersModel` or `MPEModel` to your own model
implementation in `src/models/<YOUR_MODEL>.py`.
- Extend `MetricComputer` to your own
metric implementation in `src/metrics/<YOUR_METRIC>.py`.
- Adapt your dataset to the `CustomDataset` APIs and
the `data` folder. If needed, extend `MPEDataset` to your own dataset type
in `src/datasets/<YOUR_DATASET>.py` and `data/<YOUR_DATASET_TYPE>`.

```
.
├── configs
│   ├── <MODEL>/<TEMPLATE>/<METRIC>.yaml    # complete config specification for an experiment
│   └── ⋮
├── data
│   ├── ML                                  # Marvin and Linzen (2018) S/V agreement templates
│   │   ├── <SOME_TEMPLATE>.jsonl
│   │   └── ⋮
│   ├── blimp                               # BLiMP (Warstadt et al., 2020) S/V agreement templates
│   │   ├── <SOME_TEMPLATE>.jsonl
│   │   └── ⋮
│   ├── verbs                               # the verb lemmas used for experiments
│   │   └── combined_verb_list.csv
│   ├── <SOME_OTHER_DATASETE>               # folder with data to support your own templates
│   └── ⋮
├── src                                     # source code
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── datasets.py                     # routes user to appropriate dataset
│   │   ├── MPE_dataset.py                  # base dataset class
│   │   ├── custom_dataset.py               # implementation for Marvin and Linzen (2018) and BLiMP dataset
│   │   ├── <SOME_DATASET>.py               # your own specific dataset implementation
│   │   └── ⋮
│   ├── metrics
│   │   ├── __init__.py
│   │   ├── metric_computer.py              # base metric class
│   │   ├── metrics.py                      # routes user to appropriate metrics
│   │   ├── ML_metric.py                    # implements TSE
│   │   ├── main_metric.py                  # implements MW and EW
│   │   ├── <SOME_METRIC_COMPUTER>.py       # your own specific metric implementation
│   │   └── ⋮
│   ├── models
│   │   ├── __init__.py
│   │   ├── models.py                       # routes user to appropriate model
│   │   ├── MPE_model.py                    # base model class
│   │   ├── transformers_model.py           # base class to interface with Hugging Face Transformers
│   │   ├── utils.py                        # utilities used by models
│   │   ├── <SOME_MODEL>.py                 # specific model implementation
│   │   └── ⋮
│   ├── constants.py
│   ├── experiments.py                      # core logic of experiment
│   ├── logger.py                           # manages record-keeping
│   ├── plots.py                            # generates summary plots and tables
├── results
│   ├── <MODEL>/<TEMPLATE>/<METRIC>/<NAME>  # experiment description path
│   │   ├── figs                            # folder of figures (currently unused)
│   │   ├── metrics                         # folder of per-metric outputs
│   │   ├── npzs                            # folder of .npz files
│   │   ├── pickles                         # folder of .pkl files
│   │   ├── config.yaml                     # config file for this experiment
│   │   └── log.txt                         # human-readable experiment log
│   └── ⋮
├── plots                                   # directory with plots and latex table
├── .gitignore
├── environment.yml
├── LICENSE
├── README.md
├── run.py                                  # entry point for running an experiment; does set-up
└── run_all.py                              # entry point for launching multiple experiments
```

## Custom Datasets

### Templates

`CustomDataset` parses `.jsonl` template files from the specified template directory.

It expects the datasets to be in the following form:
- Each line of a template file is a dict with a `sentence_good` field,
a `sentence_bad` field, and a `label` field.
- The `sentence_{good/bad}` fields contain the strings of correct and
incorrect sentences. These should differ by exactly one verb.
- The `label` field is `-2` if the correct verb is plural, and is `-1` if
the correct verb is singular.

### Verbs
The verbs in `data/verbs/combined_verb_list.csv` are derived from COCA and the Penn Treebank. The rest of the verbs can be found [here](https://patternbasedwriting.com/1/Giant-Verb-List-3250-Verbs.pdf)
and can be appended to the csv file.

## Custom Models

Custom models should extend `MPEModel` and implement two methods: `word_to_index` which maps vocabulary item strings to indices and `predict` which returns logits given a batch of left and right contexts on the sides of the verb of interest.

To add a custom model that is in the Hugging Face Transformers library, extend the `TransformersModel` class. (See `src/models/roberta_model.py` for an example.) This class automatically creates the `word_to_index` method from a `transformers.PreTreainedTokenizer`.

Finally, to use the custom model add it to the `model_name_to_MPEModel_class` dictionary in `src/models/models.py`.

## Custom Metrics

Custom models should extend `MetricComputer` and implement a `_compute` method to generate a score for each example given a batch of logits, labels indicating if plural in singular conjugations are preferred, and the model's `word_to_index`.
To access the custom metric, add it to the `metric_name_to_MetricComputer_class` dictionary in `src/models/models.py`


## Configs
Below is an annotated config explaining how to run your own experiments
```
dataset:
  add_trailing_period: true                         # adds trailing period to right context. Should be true.
  capitalize_first_word: true                       # captilizes first word of minimal pair. Should be true for cased models.
  max_examples_per_template: null                   # controls number of examples held in the dataset (useful for debugging)
  name: custom                                      # dataset name. Should always be custom
  template_dir: data/ML                             # folder where templates is stored
  template_files:
  - simple_agrmt_all.jsonl                          # names of templates to evaluate
experiment:
  max_examples: null                                # controls number of examples to send to the model. Functions the same as dataset.max_examples_per_template 
logger:
  path: results/bert-base-cased/ML_simple_agrmt/mw 
  print_metrics_to_general_log: false               # prints individual metrics logging info to the general log in addition to metric-specific log
  print_to_stdout: false                            # prints log to STDOUT as well as log file
metrics:
   ML:                                              # ML: TSE
    example_aggregator: mean                        # averages scores over templates
    use_custom_dataset: true                        # should always be true
  main:                                             # main: EW or MW (see use_equal_verb_voting)
    cutoffs_bot:                                    # bottom percentile cutoffs to investigate
    - 0.5
    - 0.1
    - 0.01
    - 0.001
    - 0.0001
    - 1.0e-05
    - 1.0e-06
    cutoffs_top:                                    # top percentile cutoffs to investigate
    - 0.1
    - 0.2
    - 0.3
    - 0.4
    - 0.5
    - 0.6
    - 0.7
    - 0.8
    - 0.9
    - 0.95
    - 0.97
    - 1.0
    example_aggregator: mean                       
    lemma_inflections_path: data/verbs/combined_verb_list.csv
    use_custom_dataset: true                       
    use_equal_verb_voting: false                   # true = use EW, false = use MW
model:
  name: bert-base-cased                            # name of the model. If using Hugging Face Transformers, should match the pretrained model name
```

## References
1. Rebecca Marvin and Tal Linzen. 2018. [Targeted syntactic evaluation of language models](https://www.aclweb.org/anthology/D18-1151/). In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_, pages 1192-1202, Brussels, Belgium. Association for Computational Linguistics.
2. Alex Warstadt, Alicia Parrish, Haokun Liu, Anhad Mohananey, Wei Peng, Sheng-Fu Wang, and Samuel R. Bowman. 2020. [BLiMP: The benchmark of linguistic minimal pairs for English](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00321/96452/BLiMP-The-Benchmark-of-Linguistic-Minimal-Pairs). _Transactions of the Association for Computational Linguistics_, 8:377–392.
