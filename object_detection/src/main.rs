use anyhow::{anyhow, ensure, Result};
use image::io::Reader as ImageReader;
use std::path::PathBuf;
use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

const MODEL_DIR: &str = "models";
const IMAGE_FILE: &str = "images/beach.jpg";

fn main() -> Result<()> {
    let path = PathBuf::from(MODEL_DIR);
    ensure!(path.exists(), anyhow!("directory not found"));

    // loading image
    let img = ImageReader::open(IMAGE_FILE)?.decode()?;
    let height = img.height() as u64;
    let width = img.width() as u64;
    let img_tensor = Tensor::<u8>::new(&[1, height, width, 3]).with_values(&img.into_rgb8())?;

    let mut graph = Graph::new();
    let bundle = SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, &path)?;
    let session = &bundle.session;

    let meta = bundle.meta_graph_def();
    let signature = meta.get_signature("serving_default")?;

    let input_info = signature.get_input("input_tensor")?;
    let input_op = graph.operation_by_name_required(&input_info.name().name)?;

    let output_infos = vec![
        signature.get_output("detection_boxes")?,
        signature.get_output("detection_classes")?,
        signature.get_output("detection_scores")?,
        signature.get_output("num_detections")?,
    ];
    let output_ops = output_infos.iter().map(|info| {
        (
            graph.operation_by_name_required(&info.name().name).unwrap(),
            info.name().index,
        )
    });

    let mut run_args = SessionRunArgs::new();
    run_args.add_feed(&input_op, input_info.name().index, &img_tensor);

    let mut tokens = vec![];
    for (op, idx) in output_ops {
        tokens.push((idx, run_args.request_fetch(&op, idx)));
    }

    session.run(&mut run_args)?;

    for (idx, t) in tokens {
        let output = run_args.fetch::<f32>(t)?;
        dbg!(&output);
        match idx {
            // detection_boxes: dim (1,100,4)
            0 => {
                let mut v: Vec<Vec<f32>> = vec![vec![0.0; 4]; 100];
                dbg!(&output[0]);
                dbg!(&output.dims());
                dbg!(&output.shape());
                for i in 0..100 {
                    for j in 0..4 {
                        v[i][j] = output[i * 4 + j];
                    }
                }
                dbg!(&v);
            }
            // detection_classes: dim (1,100)
            1 => {
                dbg!(&output[0]);
            }
            // detection_scores: dim (1,100)
            2 => {
                dbg!(&output[0]);
            }
            // num_detections: dim (1)
            _ => {
                dbg!(&output[0]);
            }
        }
    }

    Ok(())
}
