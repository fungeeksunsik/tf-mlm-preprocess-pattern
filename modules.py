from typing import Dict

import tensorflow as tf
import tensorflow_text as text

import config


class TextTokenizeLayer(tf.keras.layers.Layer):

    def __init__(self, tokenizer_path: str, name: str = "text_tokenize_layer"):
        super(TextTokenizeLayer, self).__init__(name=name)
        if not tokenizer_path.endswith(".model"):
            raise ValueError("tokenizer_path must be a file path ends with '.model' suffix")
        else:
            self.tokenizer = text.SentencepieceTokenizer(
                model=open(tokenizer_path, "rb").read(),
                out_type=tf.int32,  # setting out_type as tf.string is also possible
                add_bos=False,
                add_eos=False,
                name="text_tokenizer"
            )

    def call(self, inputs: tf.Tensor) -> tf.RaggedTensor:
        """
        tokenize text within inputs argument into sequence of tokens
        :param inputs: series of raw text in tf.Tensor type
        :return: square tensor whose element contains sequence of tokens
        """
        x = tf.strings.lower(
            input=inputs,
            encoding="utf-8",
            name="lowercase_converter",
        )
        x = tf.strings.regex_replace(
            input=x,
            pattern="[^a-z0-9 ]",
            rewrite=" ",
            replace_global=True,
            name="text_normalizer",
        )
        return self.tokenizer.tokenize(x)


class PostProcessLayer(tf.keras.layers.Layer):

    def __init__(self, name: str = "post_process_layer"):
        super(PostProcessLayer, self).__init__(name=name)
        self.trimmer = text.WaterfallTrimmer(config.SEQUENCE_MAX_LEN)
        self.bos_id = config.BOS_ID
        self.eos_id = config.EOS_ID

    def call(self, inputs: tf.RaggedTensor) -> tf.RaggedTensor:
        """
        Since NSP task is not considered here, second result of `combine_segments` method which stands for sentence
        indices became redundant. Therefore, first result(ragged tensor of token indices) is returned only.
        :param inputs: output of TextTokenizeLayer
        :return: post-processed
        """
        trimmed_inputs = self.trimmer.trim(segments=[inputs])  # as explained in README, `inputs` is enveloped by list
        combined_segments, _ = text.combine_segments(
            segments=trimmed_inputs,
            start_of_sequence_id=self.bos_id,
            end_of_segment_id=self.eos_id,
        )
        return combined_segments


class SequenceMaskLayer(tf.keras.layers.Layer):

    def __init__(self, mask_token_id: int, name: str = "sequence_mask_layer"):
        super(SequenceMaskLayer, self).__init__(name=name)
        self.mask_selector = text.RandomItemSelector(
            max_selections_per_batch=config.MAX_SELECTION,
            selection_rate=config.SELECTION_PROB,
            unselectable_ids=[config.PAD_ID, config.UNK_ID, config.BOS_ID, config.EOS_ID],
        )
        self.mask_value_chooser = text.MaskValuesChooser(
            vocab_size=config.VOCAB_SIZE,
            mask_token=mask_token_id,
            mask_token_rate=0.8,
            random_token_rate=0.1,
        )

    def call(self, inputs: tf.RaggedTensor) -> Dict[str, tf.Tensor]:
        """

        :param inputs: BERT-postprocessed inputs
        :return: pad token filled tensor
        """
        masked_token_ids, masked_pos, masked_values = text.mask_language_model(
            input_ids=inputs,
            item_selector=self.mask_selector,
            mask_values_chooser=self.mask_value_chooser,
        )
        input_ids, token_types = text.pad_model_inputs(
            input=masked_token_ids,
            max_seq_length=config.SEQUENCE_MAX_LEN + 2,  # max length + cls + sep
            pad_value=config.PAD_ID,
        )
        masked_pos, _ = text.pad_model_inputs(
            input=masked_pos,
            max_seq_length=config.MAX_SELECTION,
            pad_value=config.PAD_ID,
        )
        masked_values, _ = text.pad_model_inputs(
            input=masked_values,
            max_seq_length=config.MAX_SELECTION,
            pad_value=config.PAD_ID,
        )
        return {
            "input_ids": input_ids,
            "masked_pos": masked_pos,
            "masked_values": masked_values,
            "token_types": token_types,
        }
