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
                .add_input::<f32>("lstm_h_in", &[1, HIDDEN_SIZE])?
                .add_input::<f32>("lstm_c_in", &[1, HIDDEN_SIZE])?
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
            lstm: build(&["lstm_h_out", "lstm_c_out"])?,
            decode_caption: build(&["decode_caption_out"])?,
        })
    }

    pub fn predict(
        &mut self,
        img: &image::DynamicImage,
    ) -> Result<Vec<Option<usize>>, menoh::Error> {
        let x = embed_img(&mut self.embed_img, img)?;
        lstm(&mut self.lstm, x, true)?;

        let mut caption = Vec::new();
        let mut t = 0;
        for _ in 0..30 {
            let x = embed_word(&mut self.embed_word, t)?;
            let h = lstm(&mut self.lstm, x, false)?;
            t = decode_caption(&mut self.decode_caption, h)?;
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
}

fn embed_img<'a>(
    model: &'a mut menoh::Model,
    img: &image::DynamicImage,
) -> Result<&'a [f32], menoh::Error> {
    let img = img.resize_exact(IMAGE_SIZE as _, IMAGE_SIZE as _, image::FilterType::Nearest);
    {
        let in_ = model.get_variable_mut::<f32>("embed_img_in")?.1;
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
    model.run()?;
    Ok(model.get_variable("embed_img_out")?.1)
}

fn embed_word(model: &mut menoh::Model, t: usize) -> Result<&[f32], menoh::Error> {
    {
        let in_ = model.get_variable_mut::<f32>("embed_word_in")?.1;
        for k in 0..VOCAB_SIZE {
            in_[k] = if k == t { 1. } else { 0. };
        }
    }
    model.run()?;
    Ok(model.get_variable("embed_word_out")?.1)
}

fn lstm<'a>(
    model: &'a mut menoh::Model,
    x: &[f32],
    reset: bool,
) -> Result<&'a [f32], menoh::Error> {
    assert_eq!(x.len(), HIDDEN_SIZE);

    let mut h = [0.; HIDDEN_SIZE];
    let mut c = [0.; HIDDEN_SIZE];
    if !reset {
        h.copy_from_slice(model.get_variable("lstm_h_out")?.1);
        c.copy_from_slice(model.get_variable("lstm_c_out")?.1);
    }
    model.get_variable_mut("lstm_h_in")?.1.copy_from_slice(&h);
    model.get_variable_mut("lstm_c_in")?.1.copy_from_slice(&c);
    model.get_variable_mut("lstm_x")?.1.copy_from_slice(x);
    model.run()?;
    Ok(model.get_variable("lstm_h_out")?.1)
}

fn decode_caption(model: &mut menoh::Model, h: &[f32]) -> Result<usize, menoh::Error> {
    assert_eq!(h.len(), HIDDEN_SIZE);

    model
        .get_variable_mut::<f32>("decode_caption_in")?
        .1
        .copy_from_slice(h);
    model.run()?;
    let out = model.get_variable::<f32>("decode_caption_out")?.1;
    Ok((0..VOCAB_SIZE)
        .max_by(|&i, &j| out[i].partial_cmp(&out[j]).unwrap_or(cmp::Ordering::Equal))
        .unwrap())
}
