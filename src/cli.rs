use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

use crate::export::{parse_container, read_model_weights, validate_ranges, write_container};
use crate::model::build_model;

use bullet_trainer::model::ModelWeights;

#[derive(Parser)]
#[command(name = "ember-bullet-v2-trainer")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    Train {
        #[arg(value_name = "DATA")]
        data: PathBuf,

        #[arg(long, default_value_t = 4096)]
        batch_size: usize,

        #[arg(long, default_value_t = 1e-3)]
        lr: f32,

        #[arg(long, default_value_t = 10)]
        superbatches: usize,

        #[arg(long, default_value = "checkpoints")]
        out_dir: PathBuf,

        #[arg(long)]
        load_checkpoint: Option<PathBuf>,

        #[arg(long, default_value_t = 1.0)]
        lambda: f32,

        #[arg(long, action = clap::ArgAction::SetTrue)]
        smoke: bool,
    },
    Export {
        #[arg(long)]
        checkpoint: Option<PathBuf>,

        #[arg(long)]
        out: PathBuf,

        #[arg(long, default_value = "Ember NNUE v2 trained with bullet")]
        description: String,

        #[arg(long)]
        init_seed: Option<u64>,
    },
}

pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Train {
            data,
            batch_size,
            lr,
            superbatches,
            out_dir,
            load_checkpoint,
            lambda,
            smoke,
        } => train_cmd(
            data,
            batch_size,
            lr,
            superbatches,
            out_dir,
            load_checkpoint,
            lambda,
            smoke,
        ),
        Commands::Export {
            checkpoint,
            out,
            description,
            init_seed,
        } => export_cmd(checkpoint.as_deref(), &out, &description, init_seed),
    }
}

#[allow(clippy::too_many_arguments)]
fn train_cmd(
    _data: PathBuf,
    _batch_size: usize,
    _lr: f32,
    _superbatches: usize,
    _out_dir: PathBuf,
    _load_checkpoint: Option<PathBuf>,
    _lambda: f32,
    _smoke: bool,
) -> Result<()> {
    bail!(
        "train requires a CUDA/ROCm build of bullet; this binary was compiled without a GPU \
         backend. Use `export` instead, or build with `--features cuda` on a CUDA machine."
    )
}

fn export_cmd(
    checkpoint: Option<&Path>,
    out: &PathBuf,
    description: &str,
    init_seed: Option<u64>,
) -> Result<()> {
    let model = build_model();
    let mut weights = ModelWeights::zeroed(&model.definition);
    match (checkpoint, init_seed) {
        (Some(path), None) => {
            weights
                .load_from(
                    std::fs::File::open(path)
                        .with_context(|| format!("opening checkpoint {:?}", path))?,
                )
                .context("loading checkpoint weights")?;
        }
        (None, Some(seed)) => {
            weights = crate::model::init_weights(&model.definition, seed);
        }
        _ => {
            bail!("provide exactly one of --checkpoint or --init-seed");
        }
    }

    let exported = read_model_weights(&weights)?;
    validate_ranges(&exported).context("weight range check")?;

    let mut buf = Vec::new();
    write_container(&mut buf, &exported, description)?;
    std::fs::write(out, &buf).with_context(|| format!("writing {:?}", out))?;

    let parsed = parse_container(&buf)?;
    ensure_container_sane(&parsed)?;

    println!(
        "Exported {} bytes to {:?} (arch {:#x}, stacks {}, description: {:?})",
        buf.len(),
        out,
        parsed.arch_hash,
        parsed.num_stacks,
        parsed.description
    );
    Ok(())
}

fn ensure_container_sane(parsed: &crate::export::ParsedContainer) -> Result<()> {
    use crate::export::{
        ARCH_HASH, EMBER_V2_VERSION, HIDDEN_SIZE, NUM_STACKS, PSQ_DIMS, PSQT_BUCKETS, THREAT_DIMS,
    };

    ensure!(parsed.version == EMBER_V2_VERSION, "bad version");
    ensure!(parsed.arch_hash == ARCH_HASH, "bad arch hash");
    ensure!(parsed.num_stacks == NUM_STACKS, "bad stack count");
    ensure!(
        parsed.ft_bias.len() == HIDDEN_SIZE,
        "bad ft_bias length {}",
        parsed.ft_bias.len()
    );
    ensure!(
        parsed.threat_weights.len() == THREAT_DIMS * HIDDEN_SIZE,
        "bad threat_weights length {}",
        parsed.threat_weights.len()
    );
    ensure!(
        parsed.psq_weights.len() == PSQ_DIMS * HIDDEN_SIZE,
        "bad psq_weights length {}",
        parsed.psq_weights.len()
    );
    ensure!(
        parsed.psqt.len() == PSQ_DIMS * PSQT_BUCKETS,
        "bad psqt length {}",
        parsed.psqt.len()
    );
    Ok(())
}
