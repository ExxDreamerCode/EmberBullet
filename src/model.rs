use bullet_compiler::model::{InitSettings, ModelBuilder, ModelNode};
use bullet_compiler::tensor::TValue;
use bullet_trainer::model::{ModelDefinition, ModelWeights, SavedFormat};

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

pub const PY_NUM_INPUTS: usize = THREAT_DIMS + 24_576;

pub const CLIP_FT_WEIGHT: f32 = 127.0 / 256.0;
pub const CLIP_FT_BIAS: f32 = 32767.0 / 256.0;
pub const CLIP_L1_WEIGHT: f32 = 127.0 / 128.0;
pub const CLIP_L2_WEIGHT: f32 = 127.0 / 64.0;
pub const CLIP_OUT_WEIGHT: f32 = 127.0 / 128.0;

pub struct EmberModel {
    pub definition: ModelDefinition,
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn py_uniform(sigma: f32) -> InitSettings {
    InitSettings::Uniform {
        mean: 0.0,
        stdev: sigma / (3.0f32).sqrt(),
    }
}

fn ft_init(sigma: f32) -> InitSettings {
    py_uniform(sigma)
}

fn halfka_psqt_init() -> TValue {
    const BLOCK_VALUES: [f32; 11] = [
        126.0, -126.0, 781.0, -781.0, 825.0, -825.0, 1276.0, -1276.0, 2538.0, -2538.0, 0.0,
    ];

    let mut values = vec![0.0f32; NUM_BUCKETS * HALFKA_DIMS];
    for bucket in 0..32 {
        for (block, &val) in BLOCK_VALUES.iter().enumerate() {
            let base = bucket * 704 + block * 64;
            for sq in 0..64 {
                values[base + sq] = val / 600.0;
            }
        }
    }
    TValue::F32(values)
}

fn ft_bias_init() -> TValue {
    let sigma = (1.0 / PY_NUM_INPUTS as f32).sqrt();
    let mut rng = Rng::new(0x5EED_C0DE);
    let mut values = vec![0.0f32; FT_OUTPUTS];
    for v in values[..L1].iter_mut() {
        *v = rng.next_unit() * sigma;
    }
    TValue::F32(values)
}

fn stacked_init(fan_in: usize, out: usize, zero_bias: bool) -> (InitSettings, InitSettings) {
    let bound = 1.0 / (fan_in as f32).sqrt();
    let mut rng = Rng::new(0x5EED_C0DE ^ fan_in as u64);

    let mut w = vec![0.0f32; NUM_BUCKETS * out * fan_in];

    let mut first = vec![0.0f32; out * fan_in];
    for v in first.iter_mut() {
        *v = rng.next_unit() * bound;
    }
    for b in 0..NUM_BUCKETS {
        w[b * out * fan_in..(b + 1) * out * fan_in].copy_from_slice(&first);
    }

    let bias = if zero_bias {
        vec![0.0f32; NUM_BUCKETS * out]
    } else {
        let mut first_b = vec![0.0f32; out];
        for v in first_b.iter_mut() {
            *v = rng.next_unit() * bound;
        }
        let mut b = vec![0.0f32; NUM_BUCKETS * out];
        for bucket in 0..NUM_BUCKETS {
            b[bucket * out..(bucket + 1) * out].copy_from_slice(&first_b);
        }
        b
    };

    (
        InitSettings::Custom(TValue::F32(w)),
        InitSettings::Custom(TValue::F32(bias)),
    )
}

pub fn build_model() -> EmberModel {
    let builder = ModelBuilder::default();

    let ft_t_stm = builder.new_sparse_input("ft_t_stm", (THREAT_DIMS, 1), MAX_ACTIVE_THREATS);
    let ft_t_ntm = builder.new_sparse_input("ft_t_ntm", (THREAT_DIMS, 1), MAX_ACTIVE_THREATS);
    let ft_h_stm = builder.new_sparse_input("ft_h_stm", (HALFKA_DIMS, 1), MAX_ACTIVE_HALFKA);
    let ft_h_ntm = builder.new_sparse_input("ft_h_ntm", (HALFKA_DIMS, 1), MAX_ACTIVE_HALFKA);
    let buckets = builder.new_sparse_input("buckets", (NUM_BUCKETS, 1), 1);
    let skip_idx_30 = builder.new_sparse_input("skip_idx_30", (L2 * NUM_BUCKETS, 1), 1);
    let skip_idx_31 = builder.new_sparse_input("skip_idx_31", (L2 * NUM_BUCKETS, 1), 1);
    let us = builder.new_dense_input("us", (1, 1));
    let score = builder.new_dense_input("score", (1, 1));
    let outcome = builder.new_dense_input("outcome", (1, 1));
    let lambda = builder.new_dense_input("lambda", (1, 1));

    let ft_wt = builder.new_weights(
        "ft_wt",
        (FT_OUTPUTS, THREAT_DIMS),
        ft_init((1.0f32 / THREAT_DIMS as f32).sqrt()),
    );
    let ft_wt_psqt = builder.new_weights(
        "ft_wt_psqt",
        (NUM_BUCKETS, THREAT_DIMS),
        InitSettings::Zeroed,
    );
    let ft_wh = builder.new_weights(
        "ft_wh",
        (FT_OUTPUTS, HALFKA_DIMS),
        ft_init((1.0f32 / 24_576.0).sqrt()),
    );
    let ft_wh_psqt = builder.new_weights(
        "ft_wh_psqt",
        (NUM_BUCKETS, HALFKA_DIMS),
        InitSettings::Custom(halfka_psqt_init()),
    );
    let ft_b = builder.new_weights(
        "ft_b",
        (FT_OUTPUTS, 1),
        InitSettings::Custom(ft_bias_init()),
    );

    let ft_wt_lo = ft_wt.slice_rows(0, 512);
    let ft_wt_hi = ft_wt.slice_rows(512, 1024);
    let ft_wh_lo = ft_wh.slice_rows(0, 512);
    let ft_wh_hi = ft_wh.slice_rows(512, 1024);
    let ft_b_lo = ft_b.slice_rows(0, 512);
    let ft_b_hi = ft_b.slice_rows(512, 1024);
    let ft_b_psqt = ft_b.slice_rows(1024, FT_OUTPUTS);

    let stm = ft_forward(
        ft_t_stm,
        ft_h_stm,
        &ft_wt_lo,
        &ft_wt_hi,
        &ft_wh_lo,
        &ft_wh_hi,
        &ft_wt_psqt,
        &ft_wh_psqt,
        &ft_b_lo,
        &ft_b_hi,
        &ft_b_psqt,
    );
    let ntm = ft_forward(
        ft_t_ntm,
        ft_h_ntm,
        &ft_wt_lo,
        &ft_wt_hi,
        &ft_wh_lo,
        &ft_wh_hi,
        &ft_wt_psqt,
        &ft_wh_psqt,
        &ft_b_lo,
        &ft_b_hi,
        &ft_b_psqt,
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

    let ntm_lo = (us_b * ntm_w_lo + them_b * stm_w_lo)
        .min(255.0 / 256.0)
        .max(0.0);
    let ntm_hi = (us_b * ntm_w_hi + them_b * stm_w_hi)
        .min(255.0 / 256.0)
        .max(0.0);
    let ft_act_ntm = ntm_lo * ntm_hi;

    let x = ft_act_stm.concat(ft_act_ntm);

    let stm_psqt_sel = stm_psqt.select(buckets);
    let ntm_psqt_sel = ntm_psqt.select(buckets);
    let psqt_diff = stm_psqt_sel - ntm_psqt_sel;
    let us_minus_half = us - 0.5;
    let psqt_term = psqt_diff * us_minus_half;

    let (l1w_init, l1b_init) = stacked_init(L1, L2, false);
    let l1w = builder.new_weights("l1w", (NUM_BUCKETS * L2, L1), l1w_init);
    let l1b = builder.new_weights("l1b", (NUM_BUCKETS * L2, 1), l1b_init);
    let l1_all = l1w.matmul(x) + l1b;
    let l1_pre = l1_all.select(buckets);

    let skip_30 = l1_all.select(skip_idx_30);
    let skip_31 = l1_all.select(skip_idx_31);
    let skip = skip_30 - skip_31;

    let l1_sqr = (l1_pre * l1_pre).min(127.0 / 128.0).max(0.0);
    let l1_crelu = l1_pre.crelu().min(127.0 / 128.0).max(0.0);
    let h1 = l1_sqr.concat(l1_crelu);

    let (l2w_init, l2b_init) = stacked_init(L2 * 2, L2, false);
    let l2w = builder.new_weights("l2w", (NUM_BUCKETS * L2, L2 * 2), l2w_init);
    let l2b = builder.new_weights("l2b", (NUM_BUCKETS * L2, 1), l2b_init);
    let l2_pre = (l2w.matmul(h1) + l2b).select(buckets);
    let l2_sqr = (l2_pre * l2_pre).min(127.0 / 128.0).max(0.0);
    let l2_crelu = l2_pre.crelu().min(127.0 / 128.0).max(0.0);
    let h2 = l2_sqr.concat(l2_crelu);

    let l3_input = h1.concat(h2);
    let (outw_init, _outb_init) = stacked_init(L2 * 2 + L2 * 2, L3, true);
    let outw = builder.new_weights("outw", (NUM_BUCKETS * L3, L2 * 2 + L2 * 2), outw_init);
    let outb = builder.new_weights("outb", (NUM_BUCKETS * L3, 1), InitSettings::Zeroed);
    let mut out_pre = (outw.matmul(l3_input) + outb).select(buckets);
    out_pre = out_pre + skip;

    let output = out_pre + psqt_term;

    let cp = output * 600.0;
    let qf = ((cp - 270.0) / 340.0).sigmoid();
    let pf = ((score - 270.0) / 380.0).sigmoid();
    let pt = lambda * pf + (1.0 - lambda) * outcome;
    let loss = (pt - qf).abs_pow(2.5).reduce_sum_batch();

    let definition = ModelDefinition::new(
        builder.ir().clone(),
        Some(loss.node()),
        [(output.node(), "output".to_string())],
    );

    EmberModel { definition }
}

pub fn init_weights(defn: &ModelDefinition, seed: u64) -> ModelWeights {
    ModelWeights::new(defn, seed)
}

#[allow(clippy::too_many_arguments)]
fn ft_forward<'a>(
    t_inp: ModelNode<'a>,
    h_inp: ModelNode<'a>,
    wt_lo: &ModelNode<'a>,
    wt_hi: &ModelNode<'a>,
    wh_lo: &ModelNode<'a>,
    wh_hi: &ModelNode<'a>,
    wt_psqt: &ModelNode<'a>,
    wh_psqt: &ModelNode<'a>,
    b_lo: &ModelNode<'a>,
    b_hi: &ModelNode<'a>,
    b_psqt: &ModelNode<'a>,
) -> (ModelNode<'a>, ModelNode<'a>, ModelNode<'a>) {
    let w_lo = wt_lo.sparse_matmul(t_inp) + wh_lo.sparse_matmul(h_inp) + *b_lo;
    let w_hi = wt_hi.sparse_matmul(t_inp) + wh_hi.sparse_matmul(h_inp) + *b_hi;
    let psqt = wt_psqt.sparse_matmul(t_inp) + wh_psqt.sparse_matmul(h_inp) + *b_psqt;
    (w_lo, w_hi, psqt)
}

pub fn saved_formats() -> Vec<SavedFormat> {
    vec![
        SavedFormat::id("ft_wt")
            .transpose()
            .round()
            .quantise::<i8>(256),
        SavedFormat::id("ft_wt_psqt").round().quantise::<i32>(9600),
        SavedFormat::id("ft_wh")
            .transpose()
            .round()
            .quantise::<i16>(256),
        SavedFormat::id("ft_wh_psqt").round().quantise::<i32>(9600),
        SavedFormat::id("ft_b").round().quantise::<i16>(256),
        SavedFormat::id("l1w")
            .transpose()
            .round()
            .quantise::<i8>(128),
        SavedFormat::id("l1b").round().quantise::<i32>(16384),
        SavedFormat::id("l2w")
            .transpose()
            .round()
            .quantise::<i8>(64),
        SavedFormat::id("l2b").round().quantise::<i32>(8192),
        SavedFormat::id("outw")
            .transpose()
            .round()
            .quantise::<i8>(128),
        SavedFormat::id("outb").round().quantise::<i32>(16384),
    ]
}
