extern crate docopt;
extern crate image;
extern crate menoh;
#[macro_use]
extern crate serde_derive;

use std::error;
use std::fs;
use std::io;
use std::path;

mod model;

use std::io::BufRead;

const USAGE: &'static str = r#"
Image captioning on Menoh

Usage: menoh-captioning <src>
"#;

#[derive(Debug, Deserialize)]
struct Args {
    arg_src: path::PathBuf,
}

fn main() -> Result<(), Box<dyn(error::Error)>> {
    let args: Args = docopt::Docopt::new(USAGE)
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());

    let mut model = model::ImageCaptionModel::from_onnx("ImageCaptionModel.onnx", "mkldnn", "")?;
    let vocab = load_vocab("vocab.txt")?;

    let caption = model.predict(&image::open(args.arg_src)?)?;
    for t in caption {
        match t {
            Some(t) => print!("{} ", vocab[t]),
            None => print!("<unk> "),
        }
    }
    println!("");

    Ok(())
}

fn load_vocab<P>(path: P) -> io::Result<Vec<String>>
    where P: AsRef<path::Path>
{
    let mut vocab = Vec::new();
    for line in io::BufReader::new(fs::File::open(path)?).lines() {
        vocab.push(line?);
    }
    Ok(vocab)
}
