# tf-bert-mlm-pattern

This repository implements common pattern appears when implementing BERT model pretraining code with Masked Language Model(MLM) task. Since this task requires a pretrained tokenizer, code implementation is simply incremented from pattern implemented in [this repository](https://github.com/fungeeksunsik/spm-tokenizer-pattern). 

* `preprocess.py`: contains codes mostly copied from repository above to wrangle IMDb corpus file and train sentencepiece tokenizer with it

Codes that are uniquely added in this repository contains postprocess logic described in the BERT paper. Tensorflow text APIs are utilized to implement every required operation to compose them within single Tensorflow graph, for purpose explained in the repository mentioned above.

* `modules.py`: contains refactored implementation of codes explained in the [official tutorial](https://www.tensorflow.org/text/guide/bert_preprocessing_guide) 

These codes are implemented and executed on macOS environment with Python 3.9 version.

## Execute

To execute implemented process, change directory into the cloned repository and execute following:

```shell
python3 --version  # Python 3.9.5
python3 -m venv venv  # generate virtual environment for this project
source venv/bin/activate  # activate generated virtual environment
pip install -r requirements.txt  # install required packages
```

To fetch IMDb corpus file into local directory and train Sentencepiece tokenizer to be utilized within text tokenize layer, execute:

```shell
python3 preprocess.py
```

Then execute Python executable to try out each layer defined in `modules.py` and see how it works. Installing jupyter to execute following codes are also possible. Layer instances can be generated as below.

```python
import config
import pandas as pd
import tensorflow as tf

from modules import TextTokenizeLayer, PostProcessLayer, SequenceMaskLayer

tokenize_layer = TextTokenizeLayer(
    tokenizer_path=f"{config.SPM_TRAINER_CONFIG['model_prefix']}.model"
)
postprocess_layer = PostProcessLayer()
sequence_mask_layer = SequenceMaskLayer(
   mask_token_id=tokenize_layer.tokenizer.string_to_id(config.MASK_TOKEN).numpy()
)
```

First, load test part of IMDb corpus and take some samples of it.

```python
sample = pd.read_csv(f"{config.LOCAL_DIR}/test.csv")["review"].sample(10)
```

Original text contained in the variable would look like: 

```
14473    Visually, this film is interesting. Light is l...
7101     OK,so this film is NOT very well known,and was...
23159    I just saw this movie for the second time with...
12937    I have no idea how anyone can give this movie ...
14575    Man kills bear. Man becomes bear. Man/bear mee...
15309    One of quite a few cartoon Scooby Doo films, "...
17982    1st watched 12/26/2008 -(Dir-Eugene Levy): Cor...
23730    I've never been a big Larry Clark fan, but som...
20086    This is not your typical Indian film. There is...
18572    I normally don't comment on movies on IMDB, bu...
```

Next, tokenize the texts using `tokenize_layer` defined above. It will return token sequences with diverse lengths in tf.RaggedTensor format. When passing this result to `postprocess_layer`, it attaches special tokens required to execute MLM task. Before that, define trivial utility function that prints decoded version of token sequences line by line.

```python
def print_decoded_version(token_sequences):
    if isinstance(token_sequences, tf.RaggedTensor):
        token_sequences = token_sequences.to_tensor(0)  
    for token_sequence in token_sequences.numpy():
       decoded_sequence = [
           tokenize_layer.tokenizer.id_to_string(token).numpy().decode("utf-8")
           for token in token_sequence
       ]
       print(decoded_sequence)

tokenized_result = tokenize_layer(tf.constant(sample.values))
postprocess_result = postprocess_layer(tokenized_result)
print_decoded_version(postprocess_result)
```

For effective result visualization, maximum sequence length is set to 5. Definitely, in real world application on this type of data, max length has to be larger than this because it trims too much information out from the data.

```shell
['[CLS]', '▁visually', '▁this', '▁film', '▁is', '▁interesting', '[SEP]']
['[CLS]', '▁ok', '▁so', '▁this', '▁film', '▁is', '[SEP]']
['[CLS]', '▁i', '▁just', '▁saw', '▁this', '▁movie', '[SEP]']
['[CLS]', '▁i', '▁have', '▁no', '▁idea', '▁how', '[SEP]']
['[CLS]', '▁man', '▁kills', '▁bear', '▁man', '▁becomes', '[SEP]']
['[CLS]', '▁one', '▁of', '▁quite', '▁a', '▁few', '[SEP]']
['[CLS]', '▁1', 'st', '▁watched', '▁12', '▁2', '[SEP]']
['[CLS]', '▁i', '▁ve', '▁never', '▁been', '▁a', '[SEP]']
['[CLS]', '▁this', '▁is', '▁not', '▁your', '▁typical', '[SEP]']
['[CLS]', '▁i', '▁normally', '▁don', '▁t', '▁comment', '[SEP]']
```

Then apply token masking on the `postprocess_result`. 

```python
mask_result = sequence_mask_layer(postprocess_result)
input_ids = mask_result["input_ids"]
masked_pos = mask_result["masked_pos"]
masked_values = mask_result["masked_values"]
```

As in BERT paper original setting, it randomly masks each token with probability of 20%. Since other settings are all equal, possible events and corresponding probability can be calculated as:

* token is not masked and remain unchanged: 0.80
* token is masked and converted to mask token: 0.16
* token is masked and converted to random token: 0.02
* token is masked but remain unchanged: 0.02

When executing `print_decoded_version(input_ids)`, how token masking works can be illustrated like below. Compared with original tokenization result above, it can be seen that some tokens are converted into mask token or random other token.  

```
['[CLS]', '▁visually', '[MASK]', '▁film', '▁is', '▁interesting', '[SEP]']
['[CLS]', '▁ok', '[MASK]', '▁this', '▁film', '▁is', '[SEP]']
['[CLS]', '▁i', '▁just', '▁bedroom', '▁this', '▁movie', '[SEP]']
['[CLS]', '[MASK]', '▁have', '▁no', '▁idea', '▁how', '[SEP]']
['[CLS]', '▁man', 'lu', '▁bear', '▁man', '▁becomes', '[SEP]']
['[CLS]', '▁one', '▁of', '▁quite', '▁a', '[MASK]', '[SEP]']
['[CLS]', '▁1', 'st', '▁watched', '▁12', '▁2', '[SEP]']
['[CLS]', '▁i', '▁ve', '▁never', '▁been', '[MASK]', '[SEP]']
['[CLS]', '▁this', '▁is', '▁not', '[MASK]', '▁typical', '[SEP]']
['[CLS]', '▁i', '▁normally', '▁don', '[MASK]', '▁comment', '[SEP]']
```

Combining `masked_pos` and `masked_values`, information on which token with which value is converted can be checked.

```python
from pprint import pprint

decoded_masked_values = [
   tokenize_layer.tokenizer.id_to_string(token).numpy().decode("utf-8")
   for token in masked_values.numpy().flatten()
]
pprint(list(zip(masked_pos.numpy().flatten(), decoded_masked_values)))
```

First element refers to index of masked token and second one refers to its original value.

```shell
[(2, '▁this'),
 (2, '▁so'),
 (3, '▁saw'),
 (1, '▁i'),
 (2, '▁kills'),
 (5, '▁few'),
 (1, '▁1'),
 (5, '▁a'),
 (4, '▁your'),
 (4, '▁t')]
```

## Comments on which Trimmer to choose

During implementation, there were a few things to note when dealing with `Trimmer` which truncates token sequences to certain fixed length. Rest part of the logics were pretty straightforward to understand, so comments are omitted.

First, token sequences of size n to be passed into `Trimmer` has to be in following format. Of course, if BERT module will be trained only by MLM task, second ragged tensor can be omitted. Still, corresponding ragged tensor has to be enveloped in Python list.

```
[                                        ex. [
  <tf.RaggedTensor [                           <tf.RaggedTensor [
    [first token array of sentence 1],           [A1, A2, A3, A4],     
    [first token array of sentence 2],           [B1, B2, B3, B4, B5],  
    ...                                          ... (n-2 more token sequences)]>
    [first token array of sentence n]]>,       <tf.RaggedTensor [
  <tf.RaggedTensor [                             [A5, A6, A7, A8, A9],  
    [second token array of sentence 1],          [B6],                   
    [second token array of sentence 2],          ... (n-2 more token sequences)]> 
    ...                                      ]  
    [second token array of sentence n]]>       
]
```

For now, there are three predefined trimming strategies implemented as Tensorflow text API and all of them are used to make concatenated length of sequence pairs identical to certain predefined max length.
  
1. `RoundRobinTrimmer`: alternately appends token to empty list until concatenated length fills maximum value
2. `ShrinkLongestTrimmer`: pops token from original sequences until concatenated length is less than max length(pops from array whose length of remaining elements are largest)
3. `WaterfallTrimmer`: assume that sequences are concatenated first, then pops element from that hypothetical sequence until its length satisfies max length budget

Given these explanations, these are my personal comments on which trimmer to use.

1. Computation cost: `Waterfall` <= `ShrinkLongest` < `RoundRobin`
    * `RoundRobin` strategy always require some operations to fill blank lists until it fills maximum length budget, while others don't have to do anything if concatenated length already satisfies max length budget. So if maximum length is set to be large enough to cover every information in the sequence, there will be many cases where `Waterfall` and `ShrinkLongest` are more cost-efficient.
2. `Waterfall` vs `ShrinkLongest`: it depends.
   * It is true that `Waterfall` strategy is even more simple than `ShrinkLongest` since it doesn't even need to alternate sequences to determine sequence of the longest length. 
   * However, it only sacrifices information contained in second sequence, which can make second sequence blank if first sequence already had consumed every max length budget. If such imbalanced length distribution can be likely to occur, `ShrinkLongest` can be relatively preferable for stability.
