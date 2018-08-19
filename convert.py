import argparse
import numpy as np
from unittest import mock

import chainer
import chainer.functions as F
import onnx_chainer

from chainer_repo.examples.image_captioning import model


class ImageCaptionModel(model.ImageCaptionModel):

    def __init__(self, vocab_size):
        super().__init__(vocab_size, rnn='nsteplstm')
        self._transpose_embed_word()
        with self.lang_model.embed_word.init_scope():
            self.lang_model.embed_word.b = chainer.Parameter(
                0, (self.lang_model.embed_word.W.shape[0],))

    def _transpose_embed_word(self):
        self.lang_model.embed_word.W.array = \
            self.lang_model.embed_word.W.array.transpose()
        self.lang_model.embed_word.W.grad = \
            self.lang_model.embed_word.W.grad.transpose()

    def embed_img(self, x):
        h = self.feat_extractor(x)
        y = self.lang_model.embed_img(h)
        return y

    def embed_word(self, x):
        y = F.linear(
            x, self.lang_model.embed_word.W, self.lang_model.embed_word.b)
        return y

    def lstm(self, h, x):
        lstm = self.lang_model.lstm[0]
        a = F.linear(x, lstm.w2, lstm.b2) + F.linear(h, lstm.w6, lstm.b6)
        i = F.linear(x, lstm.w0, lstm.b0) + F.linear(h, lstm.w4, lstm.b4)
        f = F.linear(x, lstm.w1, lstm.b1) + F.linear(h, lstm.w5, lstm.b5)
        o = F.linear(x, lstm.w3, lstm.b3) + F.linear(h, lstm.w7, lstm.b7)
        return a, i, f, o

    def decode_caption(self, x):
        h = F.dropout(x, 0.5)
        y = self.lang_model.decode_caption(h)
        return y

    def __call__(
            self,
            embed_img_in, embed_word_in, lstm_h, lstm_x, decode_caption_in):
        embed_img_in.node._onnx_name = 'embed_img_in'
        embed_img_out = self.embed_img(embed_img_in)
        embed_img_out.node._onnx_name = 'embed_img_out'

        embed_word_in.node._onnx_name = 'embed_word_in'
        embed_word_out = self.embed_word(embed_word_in)
        embed_word_out.node._onnx_name = 'embed_word_out'

        lstm_h.node._onnx_name = 'lstm_h'
        lstm_x.node._onnx_name = 'lstm_x'
        lstm_a, lstm_i, lstm_f, lstm_o = self.lstm(lstm_h, lstm_x)
        lstm_a.node._onnx_name = 'lstm_a'
        lstm_i.node._onnx_name = 'lstm_i'
        lstm_f.node._onnx_name = 'lstm_f'
        lstm_o.node._onnx_name = 'lstm_o'

        decode_caption_in.node._onnx_name = 'decode_caption_in'
        decode_caption_out = self.decode_caption(decode_caption_in)
        decode_caption_out.node._onnx_name = 'decode_caption_out'

        return embed_img_out, embed_word_out, \
            lstm_a, lstm_i, lstm_f, lstm_o, \
            decode_caption_out

    def serialize(self, serializer):
        self._transpose_embed_word()
        super().serialize(serializer)
        self._transpose_embed_word()


class IDGenerator(object):

    def __init__(self):
        # keep original
        self._id = id

    def __call__(self, obj):
        if isinstance(obj, chainer.Variable):
            obj = obj.node
        return getattr(obj, '_onnx_name', self._id(obj))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--out', default='ImageCaptionModel.onnx')
    args = parser.parse_args()

    image_size = 224
    vocab_size = 8942
    hidden_size = 512

    model = ImageCaptionModel(vocab_size)
    chainer.serializers.load_npz(args.model, model, strict=False)

    embed_img_in = np.empty((1, 3, image_size, image_size), dtype=np.float32)
    embed_word_in = np.empty((1, vocab_size), dtype=np.float32)
    lstm_h = np.empty((1, hidden_size), dtype=np.float32)
    lstm_x = np.empty((1, hidden_size), dtype=np.float32)
    decode_caption_in = np.empty((1, hidden_size), dtype=np.float32)
    with chainer.using_config('train', False), \
            mock.patch('builtins.id', IDGenerator()):
        onnx_chainer.export(
            model,
            (embed_img_in, embed_word_in, lstm_h, lstm_x, decode_caption_in),
            filename=args.out)


if __name__ == '__main__':
    main()
