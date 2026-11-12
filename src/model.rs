use bullet_compiler::model::{InitSettings, ModelBuilder, ModelNode};
use bullet_trainer::model::{ModelDefinition, ModelWeights, SavedFormat};

use crate::features::{EmberFullThreats, EmberHalfKAv2Hm};

pub const L1: usize = 1024;
pub const L2: usize = 32;
pub const L3: usize = 1;
pub const NUM_BUCKETS: usize = 8;
pub const FT_OUTPUTS: usize = L1 + NUM_BUCKETS;
pub const THREAT_DIMS: usize = 60_720;
pub const HALFKA_DIMS: usize = 22_528;
pub const COMBINED_INPUTS: usize = THREAT_DIMS + HALFKA_DIMS;
pub const MAX_ACTIVE_THREATS: usize = 256;
pub const MAX_ACTIVE_HALFKA: usize = 32;
pub const MAX_ACTIVE_COMBINED: usize = MAX_ACTIVE_THREATS + MAX_ACTIVE_HALFKA;

pub const CLIP_FT_WEIGHT: f32 = 127.0 / 256.0;
pub const CLIP_L1_WEIGHT: f32 = 127.0 / 128.0;
pub const CLIP_L2_WEIGHT: f32 = 127.0 / 64.0;
pub const CLIP_OUT_WEIGHT: f32 = 127.0 / 128.0;

pub struct EmberModel {
    pub definition: ModelDefinition,
    pub input_names: Vec<String>,
}

fn ft_init() -> InitSettings {
    let stdev = (1.0 / COMBINED_INPUTS as f32).sqrt();
    InitSettings::Uniform { mean: 0.0, stdev }
}

pub fn build_model() -> EmberModel {
    let builder = ModelBuilder::default();

    let ft_stm = builder.new_sparse_input("ft_stm", (COMBINED_INPUTS, 1), MAX_ACTIVE_COMBINED);
    let ft_ntm = builder.new_sparse_input("ft_ntm", (COMBINED_INPUTS, 1), MAX_ACTIVE_COMBINED);
    let buckets = builder.new_sparse_input("buckets", (NUM_BUCKETS, 1), 1);
    let skip_idx_30 = builder.new_sparse_input("skip_idx_30", (L2 * NUM_BUCKETS, 1), 1);
    let skip_idx_31 = builder.new_sparse_input("skip_idx_31", (L2 * NUM_BUCKETS, 1), 1);
    let us = builder.new_dense_input("us", (1, 1));
    let score = builder.new_dense_input("score", (1, 1));
    let outcome = builder.new_dense_input("outcome", (1, 1));
    let lambda = builder.new_dense_input("lambda", (1, 1));

    let input_names = vec![
        "ft_stm".to_string(),
        "ft_ntm".to_string(),
        "buckets".to_string(),
        "skip_idx_30".to_string(),
        "skip_idx_31".to_string(),
        "us".to_string(),
        "score".to_string(),
        "outcome".to_string(),
        "lambda".to_string(),
    ];

    let ft_w = builder.new_weights("ft_w", (FT_OUTPUTS, COMBINED_INPUTS), ft_init());
    let ft_b = builder.new_weights("ft_b", (FT_OUTPUTS, 1), InitSettings::Zeroed);

    let ft_w_lo = ft_w.slice_rows(0, 512);
    let ft_w_hi = ft_w.slice_rows(512, 1024);
    let ft_w_psqt = ft_w.slice_rows(1024, FT_OUTPUTS);
    let ft_b_lo = ft_b.slice_rows(0, 512);
    let ft_b_hi = ft_b.slice_rows(512, 1024);
    let ft_b_psqt = ft_b.slice_rows(1024, FT_OUTPUTS);

    let stm = ft_forward(
        ft_stm, &ft_w_lo, &ft_w_hi, &ft_w_psqt, &ft_b_lo, &ft_b_hi, &ft_b_psqt,
    );
    let ntm = ft_forward(
        ft_ntm, &ft_w_lo, &ft_w_hi, &ft_w_psqt, &ft_b_lo, &ft_b_hi, &ft_b_psqt,
    );
    let (stm_w_lo, stm_w_hi, stm_psqt) = stm;
    let (ntm_w_lo, ntm_w_hi, ntm_psqt) = ntm;

    let us_b = us.broadcast_across_rows(512);
    let them_b = (1.0 - us).broadcast_across_rows(512);
    let stm_lo = (us_b * stm_w_lo + them_b * ntm_w_lo)
        .min(255.0 / 256.0)
        .max(0.0);
    let stm_hi = (us_b * stm_w_hi + them_b * ntm_w_hi)
        .min(255.0 / 256.0)
        .max(0.0);
    let ft_act_stm = stm_lo * stm_hi;

    let them2_b = (1.0 - us).broadcast_across_rows(512);
    let ntm_lo = (us_b * ntm_w_lo + them2_b * stm_w_lo)
        .min(255.0 / 256.0)
        .max(0.0);
    let ntm_hi = (us_b * ntm_w_hi + them2_b * stm_w_hi)
        .min(255.0 / 256.0)
        .max(0.0);
    let ft_act_ntm = ntm_lo * ntm_hi;

    let x = ft_act_stm.concat(ft_act_ntm);

    let stm_psqt_sel = stm_psqt.select(buckets);
    let ntm_psqt_sel = ntm_psqt.select(buckets);
    let psqt_diff = stm_psqt_sel - ntm_psqt_sel;
    let us_minus_half = us - 0.5;
    let psqt_term = psqt_diff * us_minus_half;

    let l1 = builder.new_affine("l1", L1, NUM_BUCKETS * L2);
    let l1_all = l1.forward(x);
    let l1_pre = l1_all.select(buckets);

    let skip_30 = l1_all.select(skip_idx_30);
    let skip_31 = l1_all.select(skip_idx_31);
    let skip = skip_30 - skip_31;

    let l1_sqr = l1_pre.sqrrelu().min(127.0 / 128.0).max(0.0);
    let l1_crelu = l1_pre.crelu().min(127.0 / 128.0).max(0.0);
    let h1 = l1_sqr.concat(l1_crelu);

    let l2 = builder.new_affine("l2", L2 * 2, NUM_BUCKETS * L2);
    let l2_pre = l2.forward(h1).select(buckets);
    let l2_sqr = l2_pre.sqrrelu().min(127.0 / 128.0).max(0.0);
    let l2_crelu = l2_pre.crelu().min(127.0 / 128.0).max(0.0);
    let h2 = l2_sqr.concat(l2_crelu);

    let l3_input = h1.concat(h2);
    let out_l = builder.new_affine("out", L2 * 2 + L2 * 2, NUM_BUCKETS * L3);
    let mut out_pre = out_l.forward(l3_input).select(buckets);
    out_pre = out_pre + skip;

    let output = out_pre + psqt_term;

    let cp = output * 600.0;
    let qf = ((cp - 270.0) / 340.0).sigmoid();
    let pf = ((score - 270.0) / 380.0).sigmoid();
    let pt = lambda * pf + (1.0 - lambda) * outcome;
    let loss = (pt - qf).abs_pow(2.5);

    let loss = loss.reduce_sum_batch();

    let definition = ModelDefinition::new(
        builder.ir().clone(),
        Some(loss.node()),
        [(output.node(), "output".to_string())],
    );

    EmberModel {
        definition,
        input_names,
    }
}

pub fn init_weights(defn: &ModelDefinition, seed: u64) -> ModelWeights {
    ModelWeights::new(defn, seed)
}

fn ft_forward<'a>(
    inp: ModelNode<'a>,
    ft_w_lo: &ModelNode<'a>,
    ft_w_hi: &ModelNode<'a>,
    ft_w_psqt: &ModelNode<'a>,
    ft_b_lo: &ModelNode<'a>,
    ft_b_hi: &ModelNode<'a>,
    ft_b_psqt: &ModelNode<'a>,
) -> (ModelNode<'a>, ModelNode<'a>, ModelNode<'a>) {
    let w_lo = (*ft_w_lo).sparse_matmul(inp) + *ft_b_lo;
    let w_hi = (*ft_w_hi).sparse_matmul(inp) + *ft_b_hi;
    let psqt = (*ft_w_psqt).sparse_matmul(inp) + *ft_b_psqt;
    (w_lo, w_hi, psqt)
}

pub fn saved_formats() -> Vec<SavedFormat> {
    vec![
        SavedFormat::id("ft_w").round().quantise::<i16>(255),
        SavedFormat::id("ft_b").round().quantise::<i16>(255),
        SavedFormat::id("l1w").round().quantise::<i8>(127),
        SavedFormat::id("l1b").round().quantise::<i32>(16384),
        SavedFormat::id("l2w").round().quantise::<i8>(63),
        SavedFormat::id("l2b").round().quantise::<i32>(8192),
        SavedFormat::id("outw").round().quantise::<i8>(127),
        SavedFormat::id("outb").round().quantise::<i32>(16384),
    ]
}

#[derive(Default)]
pub struct FeatureInputs {
    pub full_threats: EmberFullThreats,
    pub halfka: EmberHalfKAv2Hm,
}

impl FeatureInputs {
    pub fn new() -> Self {
        Self::default()
    }
}
