use anyhow::{Context, Result, ensure};
use bullet_trainer::model::{ModelInputs, ModelInputsMapper};
use bullet_trainer::optimiser::{Optimiser, ranger::Ranger};
use bullet_trainer::reader::{DataReader, ReadMapLoader};
use bullet_trainer::run::{Step, TrainingSchedule, TrainingSteps, train};

use crate::cli::TrainConfig;
use crate::features::{COMBINED_DIMS, EmberCombinedFeatures};
use crate::model::{
    L2, MAX_ACTIVE_COMBINED, NUM_BUCKETS, build_model, init_weights, saved_formats,
};
use crate::position::{BinpackReader, EmberPosition};

#[cfg(feature = "cuda")]
type Gpu = bullet_gpu::runtime::cuda::Cuda;
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
type Gpu = bullet_gpu::runtime::rocm::ROCm;
#[cfg(all(not(feature = "cuda"), not(feature = "rocm"), feature = "mock-train"))]
type Gpu = bullet_gpu::runtime::mock::MockGpu;

fn output_bucket(piece_count: u32) -> usize {
    let b = (piece_count as i32 - 2) / 4;
    b.clamp(0, 7) as usize
}

impl DataReader<EmberPosition> for BinpackReader {
    fn read_chunks<F: FnMut(&[EmberPosition]) -> bool>(&self, skip_count: usize, f: F) {
        BinpackReader::read_chunks(self, skip_count, f)
    }
}

fn make_loader(
    cfg: &TrainConfig,
    threads: u8,
    lambda: f32,
) -> ReadMapLoader<BinpackReader, EmberPosition> {
    let inputs = ModelInputs::default()
        .add_sparse("ft_stm", (COMBINED_DIMS, 1), MAX_ACTIVE_COMBINED)
        .add_sparse("ft_ntm", (COMBINED_DIMS, 1), MAX_ACTIVE_COMBINED)
        .add_sparse("buckets", (NUM_BUCKETS, 1), 1)
        .add_sparse("skip_idx_30", (L2 * NUM_BUCKETS, 1), 1)
        .add_sparse("skip_idx_31", (L2 * NUM_BUCKETS, 1), 1)
        .add_dense("us", (1, 1))
        .add_dense("score", (1, 1))
        .add_dense("outcome", (1, 1))
        .add_dense("lambda", (1, 1));

    let features = EmberCombinedFeatures::new();

    let mapper =
        ModelInputsMapper::build(
            &inputs,
            move |pos: &EmberPosition,
                  _step: Step,
                  (
                (((((((stm, ntm), buckets), skip30), skip31), us), score), outcome),
                lambda_buf,
            )| {
                stm.fill(-1);
                ntm.fill(-1);

                let mut stm_idx = Vec::with_capacity(MAX_ACTIVE_COMBINED);
                let mut ntm_idx = Vec::with_capacity(MAX_ACTIVE_COMBINED);
                features.collect(pos, &mut stm_idx, &mut ntm_idx);
                ensure_nnz(stm_idx.len(), ntm_idx.len());

                for (buf, idx) in [(stm, &stm_idx), (ntm, &ntm_idx)] {
                    for (dst, src) in buf.iter_mut().zip(idx.iter()) {
                        *dst = *src as i32;
                    }
                }

                let bucket = output_bucket(pos.piece_count());
                buckets[0] = bucket as i32;
                skip30[0] = (bucket * L2 + 30) as i32;
                skip31[0] = (bucket * L2 + 31) as i32;

                us[0] = 1.0;
                score[0] = f32::from(pos.score);
                outcome[0] = pos.result;
                lambda_buf[0] = lambda;
            },
        );

    let buffer_mb = 512;
    let reader = BinpackReader::new(&cfg.data, buffer_mb, threads as usize, |_| true);
    ReadMapLoader::new(reader, mapper, threads)
}

fn ensure_nnz(stm_len: usize, ntm_len: usize) {
    assert!(
        stm_len <= MAX_ACTIVE_COMBINED,
        "stm features {stm_len} > nnz {MAX_ACTIVE_COMBINED}"
    );
    assert!(
        ntm_len <= MAX_ACTIVE_COMBINED,
        "ntm features {ntm_len} > nnz {MAX_ACTIVE_COMBINED}"
    );
}

pub fn run_train(cfg: TrainConfig) -> Result<()> {
    ensure!(
        cfg.lambda >= 0.0 && cfg.lambda <= 1.0,
        "lambda must be in [0, 1]"
    );
    ensure!(cfg.start_superbatch >= 1, "start-superbatch must be >= 1");
    ensure!(cfg.superbatches >= 1, "superbatches must be >= 1");
    ensure!(cfg.save_rate >= 1, "save-rate must be >= 1");

    let mut batch_size = cfg.batch_size;
    let mut batches_per_superbatch = cfg.batches_per_superbatch;
    let mut superbatches = cfg.superbatches;
    let mut threads = cfg.threads;
    if cfg.smoke {
        batch_size = batch_size.min(1024);
        batches_per_superbatch = 8;
        superbatches = superbatches.min(2);
        threads = threads.min(2);
        println!(
            ">>> SMOKE: {superbatches} superbatches x {batches_per_superbatch} batches x {batch_size}"
        );
    }
    if threads == 0 {
        threads = std::thread::available_parallelism()
            .map_or(4, |n| n.get())
            .min(16) as u8;
    }

    let end_superbatch = cfg.start_superbatch + superbatches - 1;

    let model = build_model();
    let definition = model.definition;

    let seed = if cfg.load_checkpoint.is_some() {
        1
    } else {
        0x5EED_C0DE
    };
    let weights = init_weights(&definition, seed);

    let device = bullet_gpu::runtime::Device::<Gpu>::new(0)
        .map_err(|e| anyhow::anyhow!("device init: {e:?}"))?;

    let mut optimiser: Optimiser<Gpu, Ranger<Gpu>> =
        Optimiser::new(definition, weights, device, Default::default())
            .map_err(|e| anyhow::anyhow!("optimiser init: {e:?}"))?;

    if let Some(path) = &cfg.load_checkpoint {
        optimiser
            .load_from_checkpoint(path)
            .map_err(|e| anyhow::anyhow!("loading checkpoint from {path}: {e:?}"))?;
        println!("Resumed optimiser state from {path}");
    }

    let steps = TrainingSteps {
        batch_size,
        batches_per_superbatch,
        start_superbatch: cfg.start_superbatch,
        end_superbatch,
    };

    let schedule = TrainingSchedule {
        steps,
        lr_schedule: Box::new(move |step: Step| {
            let initial = cfg.lr;
            let final_lr = cfg.final_lr;
            if end_superbatch <= 1 {
                return initial;
            }
            let t = (step.superbatch() - 1) as f32 / (end_superbatch - 1) as f32;
            final_lr + (initial - final_lr) * 0.5 * (1.0 + (std::f32::consts::PI * t).cos())
        }),
        log_rate: 128,
        track_grad_norms: false,
    };

    let dataloader = make_loader(&cfg, threads, cfg.lambda);

    std::fs::create_dir_all(&cfg.out_dir).with_context(|| format!("creating {:?}", cfg.out_dir))?;

    let out_dir = cfg.out_dir.clone();
    let save_rate = cfg.save_rate;

    train(
        &mut optimiser,
        schedule,
        dataloader,
        |_, _, _| {},
        |optimiser, step| {
            let superbatch = step.superbatch();
            if superbatch % save_rate == 0 || superbatch == step.final_superbatch() {
                let path = format!("{out_dir}/ember-bullet-{superbatch}");
                std::fs::create_dir(&path).ok();

                let optimiser_path = format!("{path}/optimiser_state");
                std::fs::create_dir(&optimiser_path).ok();
                optimiser
                    .write_to_checkpoint(&optimiser_path)
                    .map_err(|e| anyhow::anyhow!("writing checkpoint: {e:?}"))
                    .unwrap();

                let guard = optimiser.cpu_weights().unwrap();
                let buf = guard
                    .to_quantised_buffer(&saved_formats(), true)
                    .map_err(|e| anyhow::anyhow!("quantising weights: {e}"))
                    .unwrap();
                std::fs::write(format!("{path}/quantised.bin"), &buf).unwrap();

                println!("Saved checkpoint [{path}]");
            }
        },
    )
    .map_err(|e| anyhow::anyhow!("training failed: {e:?}"))?;

    println!(
        "Training done (superbatches {}..{}).",
        cfg.start_superbatch, end_superbatch
    );
    Ok(())
}
