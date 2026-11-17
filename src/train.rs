use anyhow::{Context, Result, ensure};
use bullet_trainer::model::{ModelInputs, ModelInputsMapper};
use bullet_trainer::optimiser::{
    Optimiser,
    clip::{WeightClipping, WeightClippingParams},
    ranger::{Ranger, RangerParams},
    utils::Placement,
};
use bullet_trainer::reader::ReadMapLoader;
use bullet_trainer::run::{Step, TrainingSchedule, TrainingSteps, train};
use sfbinpack::TrainingDataEntry;

use crate::cli::TrainConfig;
use crate::features::EmberCombinedFeatures;
use crate::model::{
    CLIP_FT_BIAS, CLIP_FT_WEIGHT, CLIP_L1_WEIGHT, CLIP_L2_WEIGHT, CLIP_OUT_WEIGHT, HALFKA_DIMS, L2,
    MAX_ACTIVE_HALFKA, MAX_ACTIVE_THREATS, NUM_BUCKETS, THREAT_DIMS, build_model, init_weights,
    saved_formats,
};
use crate::position::{BinpackReader, EmberPosition};

#[cfg(feature = "cuda")]
type Gpu = bullet_gpu::runtime::cuda::Cuda;
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
type Gpu = bullet_gpu::runtime::rocm::ROCm;
#[cfg(all(not(feature = "cuda"), not(feature = "rocm"), feature = "mock-train"))]
type Gpu = bullet_gpu::runtime::mock::MockGpu;

fn output_bucket(piece_count: u32) -> usize {
    let b = (piece_count as i32 - 1) / 4;
    b.clamp(0, 7) as usize
}

fn remap_tablebase_score(score: i16) -> f32 {
    const MATE_SCORE: i32 = 32000;
    const MAX_MATE_PLY: i32 = 245;
    const BASE: f32 = 7000.0;
    const SCALE: f32 = 20000.0;
    const DECAY: f32 = 0.8;

    let abs_score = (score as i32).abs();
    let tb_mate_threshold = MATE_SCORE - MAX_MATE_PLY;

    if abs_score >= tb_mate_threshold {
        let plies = (MATE_SCORE - abs_score).max(0) as f32;
        let remapped = BASE + DECAY.powf(plies) * SCALE;
        if score < 0 { -remapped } else { remapped }
    } else {
        score as f32
    }
}

fn ranger_params(min_weight: f32, max_weight: f32) -> RangerParams {
    RangerParams {
        decay: 0.0,
        beta1: 0.9,
        beta2: 0.999,
        min_weight,
        max_weight,
        alpha: 0.5,
        k: 5,
    }
}

#[derive(Clone, Debug, Default)]
pub struct DataSkipConfig {
    pub filtered: bool,
    pub wld_filtered: bool,
    pub random_fen_skipping: u32,
    pub early_fen_skipping: i32,
    pub soft_early_fen_skipping: i32,
    pub simple_eval_skipping: i32,
    pub pc_y0: f32,
    pub pc_y1: f32,
    pub pc_y2: f32,
    pub pc_y3: f32,
    pub pc_y4: f32,
    pub ply_x1: f32,
    pub ply_y1: f32,
    pub ply_x2: f32,
    pub ply_y2: f32,
    pub ply_x3: f32,
    pub ply_y3: f32,
    pub ply_x4: f32,
    pub ply_y4: f32,
}

impl DataSkipConfig {
    pub fn defaults() -> Self {
        Self {
            filtered: true,
            wld_filtered: true,
            random_fen_skipping: 0,
            early_fen_skipping: -1,
            soft_early_fen_skipping: 20,
            simple_eval_skipping: -1,
            pc_y0: 0.0,
            pc_y1: 0.4,
            pc_y2: 1.0,
            pc_y3: 1.0,
            pc_y4: 0.75,
            ply_x1: 0.0,
            ply_y1: 0.1,
            ply_x2: 6.0,
            ply_y2: 0.15,
            ply_x3: 10.0,
            ply_y3: 0.25,
            ply_x4: 18.0,
            ply_y4: 0.75,
        }
    }
}

fn spline(x: f32, points: &[(f32, f32)]) -> f32 {
    if points.is_empty() {
        return 0.0;
    }
    if x <= points[0].0 {
        return points[0].1;
    }
    if x >= points.last().unwrap().0 {
        return points.last().unwrap().1;
    }
    for i in 0..points.len().saturating_sub(1) {
        let (x0, y0) = points[i];
        let (x1, y1) = points[i + 1];
        if x >= x0 && x <= x1 {
            let t = if x1 == x0 { 0.0 } else { (x - x0) / (x1 - x0) };
            return y0 + t * (y1 - y0);
        }
    }
    points.last().unwrap().1
}

fn hash_entry(entry: &TrainingDataEntry) -> u64 {
    let mut h: u64 = 0x9E37_79B9_7F4A_7C15;
    h = h
        .wrapping_mul(0xFF51_AE7D_7163)
        .wrapping_add(entry.pos.occupied().bits());
    h = h.wrapping_add(entry.ply as u64);
    h = h.wrapping_add(entry.score as u64);
    h
}

fn make_train_filter() -> impl Fn(&TrainingDataEntry) -> bool + Send + Sync + 'static {
    let cfg = DataSkipConfig::defaults();
    move |entry: &TrainingDataEntry| {
        if !cfg.filtered {
            return true;
        }

        let ply = entry.ply as f32;
        let pc = entry.pos.occupied().bits().count_ones() as f32;

        if cfg.early_fen_skipping >= 0 && entry.ply <= cfg.early_fen_skipping as u16 {
            return false;
        }

        if cfg.soft_early_fen_skipping > 0 {
            let ply_points = [
                (cfg.ply_x1, cfg.ply_y1),
                (cfg.ply_x2, cfg.ply_y2),
                (cfg.ply_x3, cfg.ply_y3),
                (cfg.ply_x4, cfg.ply_y4),
            ];
            let prob = spline(ply, &ply_points);
            let threshold = prob * (cfg.soft_early_fen_skipping as f32 / 100.0);
            if threshold > 0.0 {
                let h = hash_entry(entry);
                if (h % 10_000) < (threshold * 10_000.0) as u64 {
                    return false;
                }
            }
        }

        let pc_points = [
            (0.0, cfg.pc_y0),
            (8.0, cfg.pc_y1),
            (16.0, cfg.pc_y2),
            (24.0, cfg.pc_y3),
            (32.0, cfg.pc_y4),
        ];
        let keep_prob = spline(pc, &pc_points);
        if keep_prob < 1.0 {
            let h = hash_entry(entry);
            if (h % 10_000) >= (keep_prob * 10_000.0) as u64 {
                return false;
            }
        }

        if cfg.wld_filtered {
            let score = entry.score as f32;
            let result = entry.result as f32;
            if score > 400.0 && result < 0.33 {
                return false;
            }
            if score < -400.0 && result > 0.67 {
                return false;
            }
        }

        if cfg.random_fen_skipping > 0 {
            let h = hash_entry(entry);
            if h % cfg.random_fen_skipping as u64 == 0 {
                return false;
            }
        }

        true
    }
}

fn make_loader(
    cfg: &TrainConfig,
    threads: u8,
    lambda: f32,
) -> ReadMapLoader<BinpackReader, EmberPosition> {
    let inputs = ModelInputs::default()
        .add_sparse("ft_t_stm", (THREAT_DIMS, 1), MAX_ACTIVE_THREATS)
        .add_sparse("ft_t_ntm", (THREAT_DIMS, 1), MAX_ACTIVE_THREATS)
        .add_sparse("ft_h_stm", (HALFKA_DIMS, 1), MAX_ACTIVE_HALFKA)
        .add_sparse("ft_h_ntm", (HALFKA_DIMS, 1), MAX_ACTIVE_HALFKA)
        .add_sparse("buckets", (NUM_BUCKETS, 1), 1)
        .add_sparse("skip_idx_30", (L2 * NUM_BUCKETS, 1), 1)
        .add_sparse("skip_idx_31", (L2 * NUM_BUCKETS, 1), 1)
        .add_dense("us", (1, 1))
        .add_dense("score", (1, 1))
        .add_dense("outcome", (1, 1))
        .add_dense("lambda", (1, 1));

    let features = EmberCombinedFeatures::new();

    let mapper = ModelInputsMapper::build(
        &inputs,
        move |pos: &EmberPosition,
              _step: Step,
              (
            (
                ((((((((t_stm, t_ntm), h_stm), h_ntm), buckets), skip30), skip31), us), score),
                outcome,
            ),
            lambda_buf,
        )| {
            t_stm.fill(-1);
            t_ntm.fill(-1);
            h_stm.fill(-1);
            h_ntm.fill(-1);

            let mut t_stm_idx = Vec::with_capacity(MAX_ACTIVE_THREATS);
            let mut t_ntm_idx = Vec::with_capacity(MAX_ACTIVE_THREATS);
            let mut h_stm_idx = Vec::with_capacity(MAX_ACTIVE_HALFKA);
            let mut h_ntm_idx = Vec::with_capacity(MAX_ACTIVE_HALFKA);
            features.collect(
                pos,
                &mut t_stm_idx,
                &mut t_ntm_idx,
                &mut h_stm_idx,
                &mut h_ntm_idx,
            );
            ensure_nnz(
                t_stm_idx.len(),
                t_ntm_idx.len(),
                h_stm_idx.len(),
                h_ntm_idx.len(),
            );

            for (buf, idx) in [
                (t_stm, &t_stm_idx),
                (t_ntm, &t_ntm_idx),
                (h_stm, &h_stm_idx),
                (h_ntm, &h_ntm_idx),
            ] {
                for (dst, src) in buf.iter_mut().zip(idx.iter()) {
                    *dst = *src as i32;
                }
            }

            let bucket = output_bucket(pos.piece_count());
            buckets[0] = bucket as i32;
            skip30[0] = (bucket * L2 + 30) as i32;
            skip31[0] = (bucket * L2 + 31) as i32;

            us[0] = 1.0;
            score[0] = remap_tablebase_score(pos.score);
            outcome[0] = pos.result;
            lambda_buf[0] = lambda;
        },
    );

    let buffer_mb = 512;
    let reader = BinpackReader::new(&cfg.data, buffer_mb, threads as usize, make_train_filter());
    ReadMapLoader::new(reader, mapper, threads)
}

fn ensure_nnz(t_stm: usize, t_ntm: usize, h_stm: usize, h_ntm: usize) {
    assert!(
        t_stm <= MAX_ACTIVE_THREATS && t_ntm <= MAX_ACTIVE_THREATS,
        "threat features {t_stm}/{t_ntm} > nnz {MAX_ACTIVE_THREATS}"
    );
    assert!(
        h_stm <= MAX_ACTIVE_HALFKA && h_ntm <= MAX_ACTIVE_HALFKA,
        "halfka features {h_stm}/{h_ntm} > nnz {MAX_ACTIVE_HALFKA}"
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

    let mut optimiser: Optimiser<Gpu, WeightClipping<Gpu, Ranger<Gpu>>> = Optimiser::new(
        definition,
        weights,
        device,
        WeightClippingParams {
            inner: ranger_params(-1.98, 1.98),
            placement: Placement::After,
            min: -1.98,
            max: 1.98,
        },
    )
    .map_err(|e| anyhow::anyhow!("optimiser init: {e:?}"))?;

    let set_clip_ranges = |optimiser: &mut Optimiser<Gpu, WeightClipping<Gpu, Ranger<Gpu>>>| {
        let clipped: [(&str, f32, f32); 11] = [
            ("ft_wt", -CLIP_FT_WEIGHT, CLIP_FT_WEIGHT),
            ("ft_b", -CLIP_FT_BIAS, CLIP_FT_BIAS),
            ("ft_wh", -CLIP_FT_WEIGHT, CLIP_FT_WEIGHT),
            ("ft_wt_psqt", -1000.0, 1000.0),
            ("ft_wh_psqt", -1000.0, 1000.0),
            ("l1w", -CLIP_L1_WEIGHT, CLIP_L1_WEIGHT),
            ("l1b", -100.0, 100.0),
            ("l2w", -CLIP_L2_WEIGHT, CLIP_L2_WEIGHT),
            ("l2b", -100.0, 100.0),
            ("outw", -CLIP_OUT_WEIGHT, CLIP_OUT_WEIGHT),
            ("outb", -100.0, 100.0),
        ];
        for (name, min, max) in clipped {
            optimiser.set_params_for_weight(
                name,
                WeightClippingParams {
                    inner: ranger_params(min, max),
                    placement: Placement::After,
                    min,
                    max,
                },
            );
        }
    };

    set_clip_ranges(&mut optimiser);

    if let Some(path) = &cfg.load_checkpoint {
        optimiser
            .load_from_checkpoint(path)
            .map_err(|e| anyhow::anyhow!("loading checkpoint from {path}: {e:?}"))?;
        println!("Resumed optimiser state from {path}");
    }

    set_clip_ranges(&mut optimiser);

    let steps = TrainingSteps {
        batch_size,
        batches_per_superbatch,
        start_superbatch: cfg.start_superbatch,
        end_superbatch,
    };

    let initial_lr = cfg.lr;
    let final_lr = cfg.final_lr;
    let lr_gamma = cfg.lr_gamma;
    let lr_schedule: Box<dyn Fn(Step) -> f32> = if lr_gamma > 0.0 {
        Box::new(move |step: Step| initial_lr * lr_gamma.powi((step.superbatch() - 1) as i32))
    } else {
        Box::new(move |step: Step| {
            if end_superbatch <= 1 {
                return initial_lr;
            }
            let t = (step.superbatch() - 1) as f32 / (end_superbatch - 1) as f32;
            final_lr + (initial_lr - final_lr) * 0.5 * (1.0 + (std::f32::consts::PI * t).cos())
        })
    };

    let schedule = TrainingSchedule {
        steps,
        lr_schedule,
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
                match guard.to_quantised_buffer(&saved_formats(), true) {
                    Ok(buf) => std::fs::write(format!("{path}/quantised.bin"), &buf).unwrap(),
                    Err(e) => {
                        eprintln!(
                            "WARNING: quantising weights failed ({e}); saving raw f32 weights to \
                             {path}/raw.bin and continuing training"
                        );
                        let mut file = std::fs::File::create(format!("{path}/raw.bin")).unwrap();
                        guard.write_into(&mut file).unwrap();
                    }
                }

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
