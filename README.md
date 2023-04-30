# tf-bert-mlm-pattern

This repository implements common pattern appears when implementing BERT model pretraining code with Masked Language Model(MLM) task. Since this task requires a pretrained tokenizer, code implementation is simply incremented from pattern implemented in [this repository](https://github.com/fungeeksunsik/spm-tokenizer-pattern). 

* `preprocess.py`: contains codes mostly copied from above repository to wrangle IMDb corpus file and train sentencepiece tokenizer with it

Codes that are uniquely added in this repository contains postprocess logic described in the [BERT paper](https://arxiv.org/abs/1810.04805). Tensorflow text APIs are utilized to implement every required operation to compose them within single Tensorflow graph, for purpose explained in the repository mentioned above.

* `evaluate.py`: contains codes that 

## Comments on which Trimmer to choose

During implementation, there were a few things to note when dealing with `Trimmer` which truncates token sequences to certain fixed length. Rest part of the logics were pretty straightforward to understand, so comments are omitted.

First, token sequences of size n to be passed into `Trimmer` has to be in following format. Of course, if BERT module will be trained only by MLM task, second sentence arrays can be omitted. Still, corresponding ragged tensor has to be enveloped in Python list.

```
[                                        ex. [
  <tf.RaggedTensor [                           <tf.RaggedTensor [
    [first token array of sentence 1],           [A1, A2, A3, A4],     
    [first token array of sentence 2],           [B1, B2, B3, B4, B5],  
    ...                                          ... (n-2 more token sequences)
    [first token array of sentence n]          ]>,
  ]>,                                          <tf.RaggedTensor [
  <tf.RaggedTensor [                             [A5, A6, A7, A8, A9],  
    [second token array of sentence 1],          [B6],              
    [second token array of sentence 2],          ... (n-2 more token sequences)
    ...                                        ]>
    [second token array of sentence n]       ]
  ]>,
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
