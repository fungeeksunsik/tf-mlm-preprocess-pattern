import tensorflow as tf
import tensorflow_text as tft


class TextTokenizeLayer(tf.keras.layers.Layer):

    def __init__(self, tokenizer_path: str, name: str = "text_tokenize_layer"):
        super(TextTokenizeLayer, self).__init__(name=name)
        if not tokenizer_path.endswith(".model"):
            raise ValueError("tokenizer_path must be a file path ends with '.model' suffix")
        else:
            self.tokenizer = tft.SentencepieceTokenizer(
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


