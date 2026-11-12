use std::fs;

use ember_bullet_v2_trainer::export::{
    ExportedStack, ExportedWeights, HIDDEN_SIZE, NUM_STACKS, PSQT_BUCKETS, parse_container,
    write_container,
};
use ember_bullet_v2_trainer::model::{
    FT_OUTPUTS, HALFKA_DIMS, L1, L2, L3, NUM_BUCKETS, THREAT_DIMS,
};

const SOURCE_FT_WT_MULT: f32 = 256.0;
const SOURCE_FT_WT_PSQT_MULT: f32 = 9600.0;
const SOURCE_FT_WH_MULT: f32 = 256.0;
const SOURCE_FT_WH_PSQT_MULT: f32 = 9600.0;
const SOURCE_FT_B_MULT: f32 = 256.0;
const SOURCE_L1W_MULT: f32 = 128.0;
const SOURCE_L1B_MULT: f32 = 16384.0;
const SOURCE_L2W_MULT: f32 = 64.0;
const SOURCE_L2B_MULT: f32 = 8192.0;
const SOURCE_OUTW_MULT: f32 = 128.0;
const SOURCE_OUTB_MULT: f32 = 16384.0;

fn read_i8(data: &[u8], pos: &mut usize) -> i32 {
    let val = data[*pos] as i8 as i32;
    *pos += 1;
    val
}

fn read_i16_le(data: &[u8], pos: &mut usize) -> i16 {
    let val = i16::from_le_bytes([data[*pos], data[*pos + 1]]);
    *pos += 2;
    val
}

fn read_i32_le(data: &[u8], pos: &mut usize) -> i32 {
    let val = i32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    val
}

fn main() {
    let input_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "quantised.bin".to_string());
    let output_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "network.nnue".to_string());

    let data = fs::read(&input_path).unwrap_or_else(|e| {
        eprintln!("Failed to read {input_path}: {e}");
        std::process::exit(1);
    });
    let mut pos = 0;

    let ft_wt_len = FT_OUTPUTS * THREAT_DIMS;
    let mut ft_wt = vec![0.0f32; ft_wt_len];
    for val in ft_wt.iter_mut() {
        *val = read_i8(&data, &mut pos) as f32 / SOURCE_FT_WT_MULT;
    }

    let ft_wt_psqt_len = PSQT_BUCKETS * THREAT_DIMS;
    let mut ft_wt_psqt = vec![0.0f32; ft_wt_psqt_len];
    for val in ft_wt_psqt.iter_mut() {
        *val = read_i32_le(&data, &mut pos) as f32 / SOURCE_FT_WT_PSQT_MULT;
    }

    let ft_wh_len = FT_OUTPUTS * HALFKA_DIMS;
    let mut ft_wh = vec![0.0f32; ft_wh_len];
    for val in ft_wh.iter_mut() {
        *val = read_i16_le(&data, &mut pos) as f32 / SOURCE_FT_WH_MULT;
    }

    let ft_wh_psqt_len = PSQT_BUCKETS * HALFKA_DIMS;
    let mut ft_wh_psqt = vec![0.0f32; ft_wh_psqt_len];
    for val in ft_wh_psqt.iter_mut() {
        *val = read_i32_le(&data, &mut pos) as f32 / SOURCE_FT_WH_PSQT_MULT;
    }

    let ft_b_len = FT_OUTPUTS;
    let mut ft_b = vec![0.0f32; ft_b_len];
    for val in ft_b.iter_mut() {
        *val = read_i16_le(&data, &mut pos) as f32 / SOURCE_FT_B_MULT;
    }

    let l1w_len = NUM_BUCKETS * L2 * L1;
    let mut l1w = vec![0.0f32; l1w_len];
    for val in l1w.iter_mut() {
        *val = read_i8(&data, &mut pos) as f32 / SOURCE_L1W_MULT;
    }

    let l1b_len = NUM_BUCKETS * L2;
    let mut l1b = vec![0.0f32; l1b_len];
    for val in l1b.iter_mut() {
        *val = read_i32_le(&data, &mut pos) as f32 / SOURCE_L1B_MULT;
    }

    let l2w_len = NUM_BUCKETS * L2 * (L2 * 2);
    let mut l2w = vec![0.0f32; l2w_len];
    for val in l2w.iter_mut() {
        *val = read_i8(&data, &mut pos) as f32 / SOURCE_L2W_MULT;
    }

    let l2b_len = NUM_BUCKETS * L2;
    let mut l2b = vec![0.0f32; l2b_len];
    for val in l2b.iter_mut() {
        *val = read_i32_le(&data, &mut pos) as f32 / SOURCE_L2B_MULT;
    }

    let outw_len = NUM_BUCKETS * L3 * (L2 * 2 + L2 * 2);
    let mut outw = vec![0.0f32; outw_len];
    for val in outw.iter_mut() {
        *val = read_i8(&data, &mut pos) as f32 / SOURCE_OUTW_MULT;
    }

    let outb_len = NUM_BUCKETS * L3;
    let mut outb = vec![0.0f32; outb_len];
    for val in outb.iter_mut() {
        *val = read_i32_le(&data, &mut pos) as f32 / SOURCE_OUTB_MULT;
    }

    let mut threat_weights = Vec::with_capacity(THREAT_DIMS * HIDDEN_SIZE);
    let mut threat_psqt = Vec::with_capacity(THREAT_DIMS * PSQT_BUCKETS);
    let mut psq_weights = Vec::with_capacity(HALFKA_DIMS * HIDDEN_SIZE);
    let mut psqt = Vec::with_capacity(HALFKA_DIMS * PSQT_BUCKETS);

    for feature in 0..THREAT_DIMS {
        for out in 0..HIDDEN_SIZE {
            threat_weights.push(ft_wt[out * THREAT_DIMS + feature]);
        }
        for bucket in 0..PSQT_BUCKETS {
            threat_psqt.push(ft_wt_psqt[feature * PSQT_BUCKETS + bucket]);
        }
    }
    for feature in 0..HALFKA_DIMS {
        for out in 0..HIDDEN_SIZE {
            psq_weights.push(ft_wh[out * HALFKA_DIMS + feature]);
        }
        for bucket in 0..PSQT_BUCKETS {
            psqt.push(ft_wh_psqt[feature * PSQT_BUCKETS + bucket]);
        }
    }

    let ft_bias = ft_b[..HIDDEN_SIZE].to_vec();

    let mut stacks = Vec::with_capacity(NUM_STACKS);
    for b in 0..NUM_STACKS {
        let base = b * L2;
        let mut fc0_bias = Vec::with_capacity(L2);
        let mut fc0_weights = Vec::with_capacity(L2 * HIDDEN_SIZE);
        let mut fc1_bias = Vec::with_capacity(L2);
        let mut fc1_weights = Vec::with_capacity(L2 * (L2 * 2));
        for h in 0..L2 {
            fc0_bias.push(l1b[base + h]);
            for col in 0..HIDDEN_SIZE {
                fc0_weights.push(l1w[base * HIDDEN_SIZE + h * HIDDEN_SIZE + col]);
            }
            fc1_bias.push(l2b[base + h]);
            for col in 0..(L2 * 2) {
                fc1_weights.push(l2w[base * (L2 * 2) + h * (L2 * 2) + col]);
            }
        }
        stacks.push(ExportedStack {
            fc0_bias,
            fc0_weights,
            fc1_bias,
            fc1_weights,
            fc2_bias: outb[b],
            fc2_weights: outw[b * (L2 * 2 + L2 * 2)..(b + 1) * (L2 * 2 + L2 * 2)].to_vec(),
        });
    }

    let weights = ExportedWeights {
        ft_bias,
        threat_weights,
        threat_psqt,
        psq_weights,
        psqt,
        stacks,
    };

    let mut buf = Vec::new();
    write_container(&mut buf, &weights, "Ember V2 net trained with bullet")
        .expect("Failed to write container");

    let parsed = parse_container(&buf).expect("Failed to re-parse container");
    println!(
        "Wrote {} bytes to {output_path} (arch {:#x}, stacks {})",
        buf.len(),
        parsed.arch_hash,
        parsed.num_stacks
    );
    fs::write(&output_path, &buf).unwrap_or_else(|e| {
        eprintln!("Failed to write {output_path}: {e}");
        std::process::exit(1);
    });
}
