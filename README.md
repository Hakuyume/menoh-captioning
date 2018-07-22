# menoh-captioning

Image captioning on [Menoh](https://github.com/pfnet-research/menoh)

The model is based on [Chainer's example](https://github.com/chainer/chainer/blob/master/examples/image_captioning/)

## Requirements

- Rust 1.27+
- [Menoh](https://github.com/pfnet-research/menoh) 1.0+

## Demo

```
$ git clone https://github.com/Hakuyume/menoh-captioning.git
$ cd menoh-captioning

$ curl -LO https://github.com/Hakuyume/menoh-captioning/releases/download/assets/ImageCaptionModel.onnx
$ curl -LO https://github.com/Hakuyume/menoh-captioning/releases/download/assets/vocab.txt

$ curl -LO https://upload.wikimedia.org/wikipedia/commons/7/79/Trillium_Poncho_cat_dog.jpg
$ cargo run --release -- Trillium_Poncho_cat_dog.jpg
a dog laying on a bed with a blanket .
```
