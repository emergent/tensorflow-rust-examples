use anyhow::{anyhow, ensure, Result};
use image::io::Reader as ImageReader;
use std::path::PathBuf;
use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

const MODEL_DIR: &str = "models";
const IMAGE_FILE: &str = "images/Dress/0.png";

fn main() -> Result<()> {
    let path = PathBuf::from(MODEL_DIR);
    ensure!(path.exists(), anyhow!("directory not found"));

    // loading image
    let img = ImageReader::open(IMAGE_FILE)?.decode()?;

    let values = img
        .to_luma8()
        .to_vec()
        .into_iter()
        .map(|b| b as f32 / 255.0)
        .collect::<Vec<f32>>();
    let img_tensor = Tensor::<f32>::new(&[28, 28]).with_values(&values)?;

    let mut graph = Graph::new();
    let bundle = SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, &path)?;
    let session = &bundle.session;

    let meta = bundle.meta_graph_def();
    let signature = meta.get_signature("serving_default")?;

    let input_info = signature.get_input("flatten_input")?;
    let input_op = graph.operation_by_name_required(&input_info.name().name)?;

    let output_info = signature.get_output("dense_1")?;
    let output_op = graph.operation_by_name_required(&output_info.name().name)?;

    let mut run_args = SessionRunArgs::new();
    run_args.add_feed(&input_op, input_info.name().index, &img_tensor);

    let token = run_args.request_fetch(&output_op, output_info.name().index);
    session.run(&mut run_args)?;

    let output = run_args.fetch::<f32>(token)?;
    let res = output.to_vec();
    println!("{:?}", res);

    Ok(())
}
