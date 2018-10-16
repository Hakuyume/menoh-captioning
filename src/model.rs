use image;
use menoh;
use std::cmp;
use std::path;

use image::GenericImage;

const IMAGE_SIZE: usize = 224;
const VOCAB_SIZE: usize = 8942;
const HIDDEN_SIZE: usize = 512;

pub struct ImageCaptionModel {
    embed_img: menoh::Model,
    embed_word: menoh::Model,
    lstm: menoh::Model,
    decode_caption: menoh::Model,
}

impl ImageCaptionModel {
    pub fn from_onnx<P>(path: P, backend: &str, backend_config: &str) -> Result<Self, menoh::Error>
    where
        P: AsRef<path::Path>,
    {
        let build = |outputs: &[&str]| {
            let builder = menoh::Builder::from_onnx(&path)?
                .add_input::<f32>("embed_img_in", &[1, 3, IMAGE_SIZE, IMAGE_SIZE])?
                .add_input::<f32>("embed_word_in", &[1, VOCAB_SIZE])?
                .add_input::<f32>("lstm_h", &[1, HIDDEN_SIZE])?
                .add_input::<f32>("lstm_x", &[1, HIDDEN_SIZE])?
                .add_input::<f32>("decode_caption_in", &[1, HIDDEN_SIZE])?;
            outputs
                .iter()
                .fold(Ok(builder), |b, o| b.and_then(|b| b.add_output(o)))?
                .build(backend, backend_config)
        };

        Ok(Self {
            embed_img: build(&["embed_img_out"])?,
            embed_word: build(&["embed_word_out"])?,
            lstm: build(&["lstm_a", "lstm_i", "lstm_f", "lstm_o"])?,
            decode_caption: build(&["decode_caption_out"])?,
        })
    }

    pub fn predict(
        &mut self,
        img: &image::DynamicImage,
    ) -> Result<Vec<Option<usize>>, menoh::Error> {
        let mut h = [0.; HIDDEN_SIZE];
        let mut c = [0.; HIDDEN_SIZE];

        self.embed_img(img)?;
        self.lstm(&mut h, &mut c, true)?;

        let mut caption = Vec::new();
        let mut t = 0;
        for _ in 0..30 {
            self.embed_word(t)?;
            self.lstm(&mut h, &mut c, false)?;
            t = self.decode_caption(&h)?;
            if t == 1 {
                break;
            } else if t >= 3 {
                caption.push(Some(t - 3));
            } else {
                caption.push(None);
            }
        }

        Ok(caption)
    }

    fn embed_img(&mut self, img: &image::DynamicImage) -> Result<(), menoh::Error> {
        let img = img.resize_exact(IMAGE_SIZE as _, IMAGE_SIZE as _, image::FilterType::Nearest);
        {
            let in_ = self.embed_img.get_variable_mut::<f32>("embed_img_in")?.1;
            for y in 0..IMAGE_SIZE {
                for x in 0..IMAGE_SIZE {
                    in_[(0 * IMAGE_SIZE + y) * IMAGE_SIZE + x] =
                        img.get_pixel(x as _, y as _).data[2] as f32 - 103.939;
                    in_[(1 * IMAGE_SIZE + y) * IMAGE_SIZE + x] =
                        img.get_pixel(x as _, y as _).data[1] as f32 - 116.779;
                    in_[(2 * IMAGE_SIZE + y) * IMAGE_SIZE + x] =
                        img.get_pixel(x as _, y as _).data[0] as f32 - 123.68;
                }
            }
        }
        self.embed_img.run()
    }

    fn embed_word(&mut self, t: usize) -> Result<(), menoh::Error> {
        {
            let in_ = self.embed_word.get_variable_mut::<f32>("embed_word_in")?.1;
            for k in 0..VOCAB_SIZE {
                in_[k] = if k == t { 1. } else { 0. };
            }
        }
        self.embed_word.run()
    }

    fn lstm(&mut self, h: &mut [f32], c: &mut [f32], use_img: bool) -> Result<(), menoh::Error> {
        assert_eq!(h.len(), HIDDEN_SIZE);
        assert_eq!(c.len(), HIDDEN_SIZE);

        self.lstm
            .get_variable_mut::<f32>("lstm_h")?
            .1
            .copy_from_slice(h);
        self.lstm
            .get_variable_mut::<f32>("lstm_x")?
            .1
            .copy_from_slice(if use_img {
                self.embed_img.get_variable("embed_img_out")?.1
            } else {
                self.embed_word.get_variable("embed_word_out")?.1
            });

        self.lstm.run()?;

        let a = self.lstm.get_variable::<f32>("lstm_a")?.1;
        let i = self.lstm.get_variable("lstm_i")?.1;
        let f = self.lstm.get_variable("lstm_f")?.1;
        let o = self.lstm.get_variable("lstm_o")?.1;

        for k in 0..HIDDEN_SIZE {
            c[k] = a[k].tanh() * sigmoid(i[k]) + sigmoid(f[k]) * c[k];
            h[k] = sigmoid(o[k]) * c[k].tanh();
        }

        Ok(())
    }

    fn decode_caption(&mut self, h: &[f32]) -> Result<usize, menoh::Error> {
        assert_eq!(h.len(), HIDDEN_SIZE);

        self.decode_caption
            .get_variable_mut::<f32>("decode_caption_in")?
            .1
            .copy_from_slice(h);

        self.decode_caption.run()?;

        let out = self
            .decode_caption
            .get_variable::<f32>("decode_caption_out")?
            .1;
        Ok((0..VOCAB_SIZE)
            .max_by(|&i, &j| out[i].partial_cmp(&out[j]).unwrap_or(cmp::Ordering::Equal))
            .unwrap())
    }
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}
