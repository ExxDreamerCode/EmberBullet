use crate::model::{
    CLIP_FT_WEIGHT, CLIP_L1_WEIGHT, CLIP_L2_WEIGHT, CLIP_OUT_WEIGHT, COMBINED_INPUTS, FT_OUTPUTS,
    L1, L2, L3, NUM_BUCKETS,
};
use anyhow::{Context, Result, bail, ensure};

pub const EMBER_V2_VERSION: u32 = 0x6A448AFA;
pub const ARCH_HASH: u32 = 0x0256ACDF;
pub const FT_HEADER_HASH: u32 = 0x6165DDC9;
pub const STACK_HASH: u32 = 0x63337116;

pub const HIDDEN_SIZE: usize = 1024;
pub const PSQ_DIMS: usize = 22_528;
pub const THREAT_DIMS: usize = 60_720;
pub const NUM_STACKS: usize = 8;
pub const PSQT_BUCKETS: usize = 8;

pub const STACK_BYTES: usize = 4 + 32 * 4 + 32 * 1024 + 32 * 4 + 32 * 64 + 4 + 128;

const MAGIC: &[u8; 17] = b"COMPRESSED_LEB128";

const FT_BIAS_SCALE: f64 = 256.0;
const THREAT_W_SCALE: f64 = 256.0;
const THREAT_PSQT_SCALE: f64 = 9600.0;
const PSQ_W_SCALE: f64 = 256.0;
const PSQT_SCALE: f64 = 9600.0;
const FC0_BIAS_SCALE: f64 = 16384.0;
const FC0_W_SCALE: f64 = 128.0;
const FC1_BIAS_SCALE: f64 = 8192.0;
const FC1_W_SCALE: f64 = 64.0;
const FC2_BIAS_SCALE: f64 = 16384.0;
const FC2_W_SCALE: f64 = 128.0;

#[derive(Clone)]
pub struct ExportedWeights {
    pub ft_bias: Vec<f32>,
    pub threat_weights: Vec<f32>,
    pub threat_psqt: Vec<f32>,
    pub psq_weights: Vec<f32>,
    pub psqt: Vec<f32>,
    pub stacks: Vec<ExportedStack>,
}

#[derive(Clone)]
pub struct ExportedStack {
    pub fc0_bias: Vec<f32>,
    pub fc0_weights: Vec<f32>,
    pub fc1_bias: Vec<f32>,
    pub fc1_weights: Vec<f32>,
    pub fc2_bias: f32,
    pub fc2_weights: Vec<f32>,
}

pub fn read_model_weights(
    weights: &bullet_trainer::model::ModelWeights,
) -> Result<ExportedWeights> {
    let get = |name: &str| -> Result<Vec<f32>> {
        let v = weights.get(name);
        match &v.values {
            bullet_compiler::tensor::TValue::F32(f) => Ok(f.clone()),
            _ => bail!("weight {name} is not f32"),
        }
    };

    let ft_w = get("ft_w")?;
    let ft_b = get("ft_b")?;
    let l1_w = get("l1w")?;
    let l1_b = get("l1b")?;
    let l2_w = get("l2w")?;
    let l2_b = get("l2b")?;
    let out_w = get("outw")?;
    let out_b = get("outb")?;

    ensure!(
        ft_w.len() == FT_OUTPUTS * COMBINED_INPUTS,
        "ft_w has {} values, expected {}",
        ft_w.len(),
        FT_OUTPUTS * COMBINED_INPUTS
    );
    ensure!(ft_b.len() == FT_OUTPUTS, "ft_b has {} values", ft_b.len());
    ensure!(
        l1_w.len() == NUM_BUCKETS * L2 * L1,
        "l1w has {} values",
        l1_w.len()
    );
    ensure!(
        l1_b.len() == NUM_BUCKETS * L2,
        "l1b has {} values",
        l1_b.len()
    );
    ensure!(
        l2_w.len() == NUM_BUCKETS * L2 * (L2 * 2),
        "l2w has {} values",
        l2_w.len()
    );
    ensure!(
        l2_b.len() == NUM_BUCKETS * L2,
        "l2b has {} values",
        l2_b.len()
    );
    ensure!(
        out_w.len() == NUM_BUCKETS * L3 * (L2 * 2 + L2 * 2),
        "outw has {} values",
        out_w.len()
    );
    ensure!(
        out_b.len() == NUM_BUCKETS * L3,
        "outb has {} values",
        out_b.len()
    );

    let cols = COMBINED_INPUTS;
    let mut threat_weights = Vec::with_capacity(THREAT_DIMS * HIDDEN_SIZE);
    let mut threat_psqt = Vec::with_capacity(THREAT_DIMS * PSQT_BUCKETS);
    let mut psq_weights = Vec::with_capacity(PSQ_DIMS * HIDDEN_SIZE);
    let mut psqt = Vec::with_capacity(PSQ_DIMS * PSQT_BUCKETS);

    for feature in 0..COMBINED_INPUTS {
        for out in 0..HIDDEN_SIZE {
            let value = ft_w[out * cols + feature];
            if feature < THREAT_DIMS {
                threat_weights.push(value);
            } else {
                psq_weights.push(value);
            }
        }
        for bucket in 0..PSQT_BUCKETS {
            let value = ft_w[(HIDDEN_SIZE + bucket) * cols + feature];
            if feature < THREAT_DIMS {
                threat_psqt.push(value);
            } else {
                psqt.push(value);
            }
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
            fc0_bias.push(l1_b[base + h]);
            for col in 0..HIDDEN_SIZE {
                fc0_weights.push(l1_w[base * HIDDEN_SIZE + h * HIDDEN_SIZE + col]);
            }
            fc1_bias.push(l2_b[base + h]);
            for col in 0..(L2 * 2) {
                fc1_weights.push(l2_w[base * (L2 * 2) + h * (L2 * 2) + col]);
            }
        }
        stacks.push(ExportedStack {
            fc0_bias,
            fc0_weights,
            fc1_bias,
            fc1_weights,
            fc2_bias: out_b[b],
            fc2_weights: out_w[b * (L2 * 2 + L2 * 2)..(b + 1) * (L2 * 2 + L2 * 2)].to_vec(),
        });
    }

    Ok(ExportedWeights {
        ft_bias,
        threat_weights,
        threat_psqt,
        psq_weights,
        psqt,
        stacks,
    })
}

pub fn validate_ranges(weights: &ExportedWeights) -> Result<()> {
    for &w in &weights.threat_weights {
        let wf = w as f64;
        let in_range = (-CLIP_FT_WEIGHT as f64 - 1e-6..=CLIP_FT_WEIGHT as f64 + 1e-6).contains(&wf);
        ensure!(
            in_range,
            "threat weight {w} outside [-{},{}]",
            CLIP_FT_WEIGHT,
            CLIP_FT_WEIGHT
        );
    }
    for &w in &weights.psq_weights {
        let wf = w as f64;
        let in_range = (-CLIP_FT_WEIGHT as f64 - 1e-6..=CLIP_FT_WEIGHT as f64 + 1e-6).contains(&wf);
        ensure!(
            in_range,
            "psq weight {w} outside [-{},{}]",
            CLIP_FT_WEIGHT,
            CLIP_FT_WEIGHT
        );
    }
    for b in &weights.stacks {
        for &w in &b.fc0_weights {
            let wf = w as f64;
            let in_range =
                (-CLIP_L1_WEIGHT as f64 - 1e-6..=CLIP_L1_WEIGHT as f64 + 1e-6).contains(&wf);
            ensure!(
                in_range,
                "fc0 weight {w} outside [-{},{}]",
                CLIP_L1_WEIGHT,
                CLIP_L1_WEIGHT
            );
        }
        for &w in &b.fc1_weights {
            let wf = w as f64;
            let in_range =
                (-CLIP_L2_WEIGHT as f64 - 1e-6..=CLIP_L2_WEIGHT as f64 + 1e-6).contains(&wf);
            ensure!(
                in_range,
                "fc1 weight {w} outside [-{},{}]",
                CLIP_L2_WEIGHT,
                CLIP_L2_WEIGHT
            );
        }
        for &w in &b.fc2_weights {
            let wf = w as f64;
            let in_range =
                (-CLIP_OUT_WEIGHT as f64 - 1e-6..=CLIP_OUT_WEIGHT as f64 + 1e-6).contains(&wf);
            ensure!(
                in_range,
                "fc2 weight {w} outside [-{},{}]",
                CLIP_OUT_WEIGHT,
                CLIP_OUT_WEIGHT
            );
        }
    }
    Ok(())
}

pub fn write_container(
    out: &mut impl std::io::Write,
    weights: &ExportedWeights,
    description: &str,
) -> Result<()> {
    let desc_bytes = description.as_bytes();

    out.write_all(&EMBER_V2_VERSION.to_le_bytes())?;
    out.write_all(&ARCH_HASH.to_le_bytes())?;
    out.write_all(&(desc_bytes.len() as u32).to_le_bytes())?;
    out.write_all(desc_bytes)?;

    out.write_all(&FT_HEADER_HASH.to_le_bytes())?;

    write_leb_block_f32(out, &weights.ft_bias, FT_BIAS_SCALE)?;

    let mut threat_raw = Vec::with_capacity(weights.threat_weights.len());
    for &w in &weights.threat_weights {
        let q = (w as f64 * THREAT_W_SCALE).round() as i32;
        ensure!(
            q >= i8::MIN as i32 && q <= i8::MAX as i32,
            "threat weight {q} out of i8 range"
        );
        threat_raw.push(q as i8 as u8);
    }
    out.write_all(&threat_raw)?;

    write_leb_block_f32(out, &weights.threat_psqt, THREAT_PSQT_SCALE)?;

    write_leb_block_f32(out, &weights.psq_weights, PSQ_W_SCALE)?;

    write_leb_block_f32(out, &weights.psqt, PSQT_SCALE)?;

    for (i, stack) in weights.stacks.iter().enumerate() {
        write_stack(out, stack).with_context(|| format!("stack {i}"))?;
    }

    Ok(())
}

fn write_stack(out: &mut impl std::io::Write, s: &ExportedStack) -> Result<()> {
    out.write_all(&STACK_HASH.to_le_bytes())?;

    for &v in &s.fc0_bias {
        let q = (v as f64 * FC0_BIAS_SCALE).round() as i32;
        out.write_all(&q.to_le_bytes())?;
    }
    for &v in &s.fc0_weights {
        let q = (v as f64 * FC0_W_SCALE).round() as i32;
        ensure!(
            q >= i8::MIN as i32 && q <= i8::MAX as i32,
            "fc0 weight {q} out of i8 range"
        );
        out.write_all(&[q as i8 as u8])?;
    }
    for &v in &s.fc1_bias {
        let q = (v as f64 * FC1_BIAS_SCALE).round() as i32;
        out.write_all(&q.to_le_bytes())?;
    }
    for &v in &s.fc1_weights {
        let q = (v as f64 * FC1_W_SCALE).round() as i32;
        ensure!(
            q >= i8::MIN as i32 && q <= i8::MAX as i32,
            "fc1 weight {q} out of i8 range"
        );
        out.write_all(&[q as i8 as u8])?;
    }
    let q = (s.fc2_bias as f64 * FC2_BIAS_SCALE).round() as i32;
    out.write_all(&q.to_le_bytes())?;
    for &v in &s.fc2_weights {
        let q = (v as f64 * FC2_W_SCALE).round() as i32;
        ensure!(
            q >= i8::MIN as i32 && q <= i8::MAX as i32,
            "fc2 weight {q} out of i8 range"
        );
        out.write_all(&[q as i8 as u8])?;
    }
    Ok(())
}

fn write_leb_block_f32(out: &mut impl std::io::Write, values: &[f32], scale: f64) -> Result<()> {
    out.write_all(MAGIC)?;

    let mut encoded = Vec::new();
    for &v in values {
        let q = (v as f64 * scale).round() as i64;
        write_sleb128(q, &mut encoded);
    }

    out.write_all(&(encoded.len() as u32).to_le_bytes())?;
    out.write_all(&encoded)?;
    Ok(())
}

pub fn write_sleb128(mut value: i64, out: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        let more = if value == 0 {
            (byte & 0x40) != 0
        } else if value == -1 {
            (byte & 0x40) == 0
        } else {
            true
        };
        if !more {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

pub struct ParsedContainer {
    pub version: u32,
    pub arch_hash: u32,
    pub description: String,
    pub ft_bias: Vec<i16>,
    pub threat_weights: Vec<i8>,
    pub threat_psqt: Vec<i32>,
    pub psq_weights: Vec<i16>,
    pub psqt: Vec<i32>,
    pub num_stacks: usize,
}

pub fn parse_container(data: &[u8]) -> Result<ParsedContainer> {
    ensure!(data.len() >= 12, "container too short for header");
    let version = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    ensure!(
        version == EMBER_V2_VERSION,
        "unsupported version {version:#x}"
    );
    let arch_hash = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    ensure!(
        arch_hash == ARCH_HASH,
        "unsupported arch hash {arch_hash:#x}"
    );
    let desc_len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
    ensure!(
        12 + desc_len <= data.len(),
        "description overruns container"
    );
    let description = String::from_utf8(data[12..12 + desc_len].to_vec())?;

    let mut pos = 12 + desc_len;
    ensure!(pos + 4 <= data.len(), "body too short");
    let ft_hash = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
    ensure!(ft_hash == FT_HEADER_HASH, "unexpected ft hash {ft_hash:#x}");
    pos += 4;

    let (ft_bias_i32, next) = decode_leb_block(data, pos, HIDDEN_SIZE)?;
    pos = next;
    let ft_bias: Vec<i16> = ft_bias_i32
        .into_iter()
        .map(|v| i16::try_from(v).map_err(|e| anyhow::anyhow!("ft_bias {v}: {e}")))
        .collect::<anyhow::Result<_>>()?;
    ensure!(ft_bias.len() == HIDDEN_SIZE, "ft_bias length");

    let threat_start = find_magic_at_or_after(data, pos)
        .ok_or_else(|| anyhow::anyhow!("no threat_weights magic"))?;
    let threat_bytes = threat_start - pos;
    ensure!(
        threat_bytes == THREAT_DIMS * HIDDEN_SIZE,
        "threat_weights length {threat_bytes} != {}",
        THREAT_DIMS * HIDDEN_SIZE
    );
    let threat_weights: Vec<i8> = data[pos..threat_start].iter().map(|&b| b as i8).collect();
    pos = threat_start;

    let (threat_psqt, next) = decode_leb_block(data, pos, THREAT_DIMS * PSQT_BUCKETS)?;
    pos = next;
    ensure!(
        threat_psqt.len() == THREAT_DIMS * PSQT_BUCKETS,
        "threat_psqt length"
    );

    let (psq_weights_i32, next) = decode_leb_block(data, pos, PSQ_DIMS * HIDDEN_SIZE)?;
    pos = next;
    let psq_weights: Vec<i16> = psq_weights_i32
        .into_iter()
        .map(|v| i16::try_from(v).map_err(|e| anyhow::anyhow!("psq_weights {v}: {e}")))
        .collect::<anyhow::Result<_>>()?;
    ensure!(
        psq_weights.len() == PSQ_DIMS * HIDDEN_SIZE,
        "psq_weights length"
    );

    let (psqt, next) = decode_leb_block(data, pos, PSQ_DIMS * PSQT_BUCKETS)?;
    pos = next;
    ensure!(psqt.len() == PSQ_DIMS * PSQT_BUCKETS, "psqt length");

    let tail = &data[pos..];
    ensure!(
        tail.len().is_multiple_of(STACK_BYTES),
        "tail length not multiple of stack bytes"
    );
    let num_stacks = tail.len() / STACK_BYTES;
    ensure!(
        num_stacks == NUM_STACKS,
        "stack count {num_stacks} != {NUM_STACKS}"
    );
    for (i, chunk) in tail.chunks(STACK_BYTES).enumerate() {
        ensure!(
            u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) == STACK_HASH,
            "stack {i} bad hash"
        );
        let _ = chunk;
    }

    Ok(ParsedContainer {
        version,
        arch_hash,
        description,
        ft_bias,
        threat_weights,
        threat_psqt,
        psq_weights,
        psqt,
        num_stacks,
    })
}

fn find_magic_at_or_after(data: &[u8], mut start: usize) -> Option<usize> {
    if start + MAGIC.len() > data.len() {
        return None;
    }
    let limit = data.len() - MAGIC.len();
    while start <= limit {
        if &data[start..start + MAGIC.len()] == MAGIC {
            return Some(start);
        }
        start += 1;
    }
    None
}

fn decode_leb_block(data: &[u8], pos: usize, expected_count: usize) -> Result<(Vec<i32>, usize)> {
    ensure!(
        data.get(pos..pos + MAGIC.len()) == Some(MAGIC),
        "expected LEB magic"
    );
    let cf = pos + MAGIC.len();
    ensure!(cf + 4 <= data.len(), "leb count outside");
    let bc = u32::from_le_bytes([data[cf], data[cf + 1], data[cf + 2], data[cf + 3]]) as usize;
    let dp = cf + 4;
    ensure!(dp + bc <= data.len(), "leb block outside data");

    let mut out = Vec::with_capacity(expected_count);
    let mut p = dp;
    let end = dp + bc;
    while p < end {
        let (v, next) = decode_signed_leb(data, p)?;
        out.push(v as i32);
        p = next;
    }
    ensure!(p == end, "LEB block does not end at boundary");
    ensure!(
        out.len() == expected_count,
        "expected {} values, got {}",
        expected_count,
        out.len()
    );
    Ok((out, end))
}

fn decode_signed_leb(data: &[u8], mut pos: usize) -> Result<(i64, usize)> {
    let mut result: i64 = 0;
    let mut shift: u32 = 0;
    loop {
        let byte = *data
            .get(pos)
            .ok_or_else(|| anyhow::anyhow!("leb out of range"))?;
        pos += 1;
        result |= (byte as i64 & 0x7f) << shift;
        if byte & 0x80 == 0 {
            if byte & 0x40 != 0 && shift < 63 {
                result |= (-1i64) << (shift + 7);
            }
            return Ok((result, pos));
        }
        shift += 7;
        ensure!(shift <= 63, "leb overflow");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sleb128_roundtrip_known_vectors() {
        let mut buf = Vec::new();
        write_sleb128(300, &mut buf);
        assert_eq!(buf, vec![0xAC, 0x02]);
        buf.clear();
        write_sleb128(-1, &mut buf);
        assert_eq!(buf, vec![0x7F]);
        buf.clear();
        write_sleb128(-5, &mut buf);
        assert_eq!(buf, vec![0x7B]);
    }

    #[test]
    fn leb_block_roundtrip() {
        let values: Vec<f32> = vec![300.0, -1.0, -5.0, 0.0, 127.0, -128.0];
        let mut encoded = Vec::new();
        write_leb_block_f32(&mut encoded, &values, 1.0).unwrap();

        let (decoded, end) = decode_leb_block(&encoded, 0, values.len()).unwrap();
        assert_eq!(end, encoded.len());
        assert_eq!(decoded, vec![300, -1, -5, 0, 127, -128]);
    }

    #[test]
    fn stack_bytes_constant() {
        assert_eq!(STACK_BYTES, 35208);
    }

    #[test]
    fn sleb128_roundtrip_full_i64() {
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state as i64
        };

        let mut cases: Vec<i64> = vec![
            0,
            -1,
            1,
            i64::MIN,
            i64::MAX,
            -2,
            -(1i64 << 31),
            -(1i64 << 32),
            -(1i64 << 35),
            -(1i64 << 40),
            -(1i64 << 56),
            1i64 << 56,
            i64::MIN + 1,
        ];
        for _ in 0..100_000 {
            cases.push(next());
        }

        for v in cases {
            let mut buf = Vec::new();
            write_sleb128(v, &mut buf);
            assert!(buf.len() <= 10, "value {v} encoded to {} bytes", buf.len());
            let (decoded, end) = decode_signed_leb(&buf, 0).expect("decode {v}");
            assert_eq!(end, buf.len());
            assert_eq!(decoded, v, "sleb128 roundtrip failed for {v}");
        }
    }
}
