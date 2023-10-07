use anyhow::{anyhow, ensure, Result};
use clap::Parser;
use image::io::Reader as ImageReader;
use std::path::{Path, PathBuf};
use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

const MODEL_DIR: &str = "models";
const TEST_DATA_DIR: &str = "images";
const LABELS: [&str; 10] = [
    "Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot",
];

type TestData = Vec<(image::DynamicImage, usize)>;

#[derive(Parser, Debug)]
#[clap(name = "fmnist-sample")]
struct Opt {
    #[command(subcommand)]
    pub sub: Subcommands,
}

#[derive(Parser, Debug)]
enum Subcommands {
    Test,
    Classify {
        #[arg(short, long)]
        file: PathBuf,
    },
}

fn main() -> Result<()> {
    let opt = Opt::parse();
    match opt.sub {
        Subcommands::Test => test_accuracy()?,
        Subcommands::Classify { file } => classify(file)?,
    }
    Ok(())
}

fn test_accuracy() -> Result<()> {
    let data = load_test_images()?;
    let data_total = data.len();
    let classifier = FashionMnistClassifier::load()?;
    let mut test_result = vec![];

    for (i, (img, label_idx)) in data.iter().enumerate() {
        let input_value = img.to_luma8().to_vec();
        let res = classifier.classify(&input_value)?;
        if let Some(idx) = get_max_index(&res) {
            print!("progress: [ {:5} / {:5} ]       \r", i + 1, data_total);
            test_result.push(*label_idx == idx);
        }
    }
    println!();

    let total = test_result.len();
    let success = test_result.iter().filter(|&&x| x).count();
    let accuracy = success as f32 / total as f32;
    println!(
        "total: {}, success: {}, failure: {}, accuracy: {} %",
        total,
        success,
        total - success,
        accuracy * 100.0
    );
    Ok(())
}

fn classify<P: AsRef<Path>>(file: P) -> Result<()> {
    let classifier = FashionMnistClassifier::load()?;

    let img = load_image(file)?;
    let input_value = img.to_luma8().to_vec();
    let res = classifier.classify(&input_value)?;

    println!("{:?}", res);
    if let Some(idx) = get_max_index(&res) {
        println!("classified: {}", LABELS[idx]);
    }

    Ok(())
}

fn load_image<P: AsRef<Path>>(filename: P) -> Result<image::DynamicImage> {
    let img = ImageReader::open(filename)?.decode()?;
    Ok(img)
}

fn load_test_images() -> Result<TestData> {
    let mut v = vec![];
    for (i, label) in LABELS.iter().enumerate() {
        let pattern = format!("{}/{}/*.png", TEST_DATA_DIR, label);
        for entry in glob::glob(&pattern)? {
            match entry {
                Ok(path) => {
                    let img = load_image(path)?;
                    v.push((img, i));
                }
                Err(e) => println!("{:?}", e),
            }
        }
    }
    Ok(v)
}

fn get_max_index(v: &[f32]) -> Option<usize> {
    let mut max = 0.0f32;
    let mut pos = None;
    for (i, &val) in v.iter().enumerate() {
        if val >= max {
            max = val;
            pos = Some(i);
        }
    }
    pos
}

struct FashionMnistClassifier {
    graph: Graph,
    bundle: SavedModelBundle,
}

impl FashionMnistClassifier {
    pub fn load() -> Result<Self> {
        let path = PathBuf::from(MODEL_DIR);
        ensure!(path.exists(), anyhow!("directory not found"));

        let mut graph = Graph::new();
        let bundle = SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, &path)?;

        Ok(Self { graph, bundle })
    }

    pub fn classify(&self, img: &[u8]) -> Result<Vec<f32>> {
        let values = img.iter().map(|&b| b as f32 / 255.0).collect::<Vec<f32>>();
        let img_tensor = Tensor::<f32>::new(&[28, 28]).with_values(&values)?;

        let session = &self.bundle.session;

        let meta = self.bundle.meta_graph_def();
        let signature = meta.get_signature("serving_default")?;

        let input_info = signature.get_input("flatten_input")?;
        let input_op = self
            .graph
            .operation_by_name_required(&input_info.name().name)?;

        let output_info = signature.get_output("dense_1")?;
        let output_op = self
            .graph
            .operation_by_name_required(&output_info.name().name)?;

        let mut run_args = SessionRunArgs::new();
        run_args.add_feed(&input_op, input_info.name().index, &img_tensor);

        let token = run_args.request_fetch(&output_op, output_info.name().index);
        session.run(&mut run_args)?;

        let output = run_args.fetch::<f32>(token)?;
        let res = output.to_vec();

        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_get_max() {
        let x = [0.2, 0.3, 0.45435, 0.1, 0.01];
        assert_eq!(get_max_index(&x), Some(2));

        let y = [];
        assert_eq!(get_max_index(&y), None);
    }
}
