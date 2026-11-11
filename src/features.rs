use crate::attacks;
use crate::position::EmberPosition;
use bullet_lib::game::inputs::SparseInputType;

const THREAT_DIMS: usize = 60_720;
const PSQ_DIMS: usize = 22_528;

fn flip_bb(bb: u64) -> u64 {
    let mut out = 0u64;
    let mut b = bb;
    while b != 0 {
        let sq = b.trailing_zeros();
        b &= b - 1;
        out |= 1u64 << (sq ^ 56);
    }
    out
}

const NUM_VALID_TARGETS: [u32; 12] = [6, 10, 8, 8, 10, 0, 6, 10, 8, 8, 10, 0];

const THREAT_MAP: [[i32; 6]; 6] = [
    [0, 1, -1, 2, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [-1, -1, -1, -1, -1, -1],
];

pub const INACTIVE: usize = usize::MAX;

fn swap_piece_color(piece: u32) -> u32 {
    if piece < 6 { piece + 6 } else { piece - 6 }
}

fn halfka_piece_base(perspective: u32, piece: u32) -> u32 {
    let piece = if perspective == 0 {
        piece
    } else {
        swap_piece_color(piece)
    };
    match piece {
        0 => 0,
        1 => 128,
        2 => 256,
        3 => 384,
        4 => 512,
        5 => 640,
        6 => 64,
        7 => 192,
        8 => 320,
        9 => 448,
        10 => 576,
        11 => 640,
        _ => unreachable!("invalid Ember piece index"),
    }
}

fn halfka_orientation(king_square: u32) -> u32 {
    if king_square & 4 == 0 { 7 } else { 0 }
}

fn threat_orientation(king_square: u32) -> u32 {
    if king_square & 4 == 0 { 0 } else { 7 }
}

pub fn halfka_index(perspective: u32, square: u32, piece: u32, king_square: u32) -> usize {
    let square = square ^ 56;
    let king_square = king_square ^ 56;
    let flip = 56 * perspective;
    let oriented_king = king_square ^ flip;
    let king_rank = oriented_king / 8;
    let king_file = oriented_king % 8;
    let king_bucket = (7 - king_rank) * 4 + king_file.min(7 - king_file);
    ((square ^ halfka_orientation(king_square) ^ flip)
        + halfka_piece_base(perspective, piece)
        + king_bucket * 11 * 64) as usize
}

#[derive(Clone, Debug)]
pub struct ThreatLut {
    index_lut1: [u32; 12 * 12 * 2],
    offsets: [u32; 12 * 64],
    index_lut2: Vec<u16>,
}

impl ThreatLut {
    pub fn build() -> Self {
        let mut index_lut2 = vec![0u16; 12 * 64 * 64];
        for piece in 0..12u32 {
            for from in 0..64u32 {
                let attacks = if piece % 6 == 0 && !(1..=6).contains(&(from / 8)) {
                    0
                } else {
                    attacks::pseudo_attacks(piece, from)
                };
                for to in 0..64u32 {
                    let before = (1u64 << to).wrapping_sub(1);
                    index_lut2[((piece * 64 + from) * 64 + to) as usize] =
                        (before & attacks).count_ones() as u16;
                }
            }
        }

        let mut offsets = [0u32; 12 * 64];
        let mut cumulative = 0usize;
        for piece in 0..12u32 {
            let mut cumulative_piece = 0usize;
            for from in 0..64u32 {
                offsets[(piece * 64 + from) as usize] = cumulative_piece as u32;
                let count = if piece % 6 == 0 && !(1..=6).contains(&(from / 8)) {
                    0
                } else {
                    attacks::pseudo_attacks(piece, from).count_ones() as usize
                };
                cumulative_piece += count;
            }
            cumulative += NUM_VALID_TARGETS[piece as usize] as usize * cumulative_piece;
        }
        assert_eq!(cumulative, THREAT_DIMS);

        let mut piece_dimensions = [0usize; 12];
        for att in 0..12u32 {
            let mut cum = 0usize;
            for from in 0..64u32 {
                let count = if att % 6 == 0 && !(1..=6).contains(&(from / 8)) {
                    0
                } else {
                    attacks::pseudo_attacks(att, from).count_ones() as usize
                };
                cum += count;
            }
            piece_dimensions[att as usize] = cum;
        }
        let mut piece_offset = [0usize; 12];
        {
            let mut acc = 0usize;
            for att in 0..12u32 {
                piece_offset[att as usize] = acc;
                acc += piece_dimensions[att as usize];
            }
        }

        let mut index_lut1 = [THREAT_DIMS as u32; 12 * 12 * 2];
        for attacker in 0..12u32 {
            for attacked in 0..12u32 {
                let attacker_type = attacker % 6;
                let attacked_type = attacked % 6;
                let map = THREAT_MAP[attacker_type as usize][attacked_type as usize];
                let enemy = attacker / 6 != attacked / 6;
                let semi_excluded = attacker_type == attacked_type && (enemy || attacker_type != 0);
                if map < 0 {
                    continue;
                }

                let feature = piece_offset[attacker as usize]
                    + (attacked / 6 * (NUM_VALID_TARGETS[attacker as usize] / 2) + map as u32)
                        as usize
                        * piece_dimensions[attacker as usize];
                let base = (attacker as usize * 12 + attacked as usize) * 2;
                index_lut1[base] = feature as u32;
                if !semi_excluded {
                    index_lut1[base + 1] = feature as u32;
                }
            }
        }

        Self {
            index_lut1,
            offsets,
            index_lut2,
        }
    }

    pub fn make_index(
        &self,
        perspective: u32,
        attacker: u32,
        from: u32,
        to: u32,
        attacked: u32,
        king_square: u32,
    ) -> usize {
        let from = from ^ 56;
        let to = to ^ 56;
        let king_square = king_square ^ 56;
        let orientation = threat_orientation(king_square) ^ (56 * perspective);
        let from = from ^ orientation;
        let to = to ^ orientation;
        let attacker = if perspective == 0 {
            attacker
        } else {
            swap_piece_color(attacker)
        };
        let attacked = if perspective == 0 {
            attacked
        } else {
            swap_piece_color(attacked)
        };

        let lut1 = self.index_lut1
            [(attacker as usize * 12 + attacked as usize) * 2 + usize::from(from < to)]
            as usize;
        if lut1 >= THREAT_DIMS {
            return THREAT_DIMS;
        }
        lut1 + self.offsets[(attacker * 64 + from) as usize] as usize
            + self.index_lut2[((attacker * 64 + from) * 64 + to) as usize] as usize
    }
}

impl Default for ThreatLut {
    fn default() -> Self {
        Self::build()
    }
}

#[derive(Clone, Debug, Default)]
pub struct EmberFullThreats {
    lut: ThreatLut,
}

impl EmberFullThreats {
    pub fn new() -> Self {
        Self {
            lut: ThreatLut::build(),
        }
    }

    fn emit_threat_pairs(&self, pos: &EmberPosition, mut f: impl FnMut(usize, usize)) {
        let stm_persp = if pos.is_white_to_move { 0 } else { 1 };
        let occupancy_a1a0: u64 = pos.bb.iter().copied().fold(0, |a, b| a | b);
        let occupancy = flip_bb(occupancy_a1a0);
        let all_pawns_a1a0 = pos.bb[0] | pos.bb[6];
        let all_pawns = flip_bb(all_pawns_a1a0);
        let white_ksq_a1a0 = if pos.is_white_to_move {
            pos.us_ksq as u32
        } else {
            pos.them_ksq as u32
        };
        let black_ksq_a1a0 = if pos.is_white_to_move {
            pos.them_ksq as u32
        } else {
            pos.us_ksq as u32
        };
        let king_squares = [white_ksq_a1a0 ^ 56, black_ksq_a1a0 ^ 56];

        let emit = |perspective: u32,
                    piece: u32,
                    from_internal: u32,
                    to_internal: u32,
                    attacked: u32,
                    stm_out: &mut Vec<usize>,
                    ntm_out: &mut Vec<usize>| {
            let king = king_squares[perspective as usize];
            let idx = self.lut.make_index(
                perspective,
                piece,
                from_internal,
                to_internal,
                attacked,
                king,
            );
            if idx < THREAT_DIMS {
                if perspective == stm_persp {
                    stm_out.push(idx);
                } else {
                    ntm_out.push(idx);
                }
            }
        };

        let mut stm = Vec::new();
        let mut ntm = Vec::new();

        for color in 0..2u32 {
            let pawns_a1a0 = pos.bb[(color * 6) as usize];
            let mut pieces = pawns_a1a0;
            while pieces != 0 {
                let from_a1a0 = pieces.trailing_zeros();
                pieces &= pieces - 1;
                let from_internal = from_a1a0 ^ 56;

                let mut attacks = attacks::pawn_attacks_bb(color, from_internal) & occupancy;
                while attacks != 0 {
                    let to_internal = attacks.trailing_zeros();
                    attacks &= attacks - 1;
                    let to_a1a0 = to_internal ^ 56;
                    let attacked = pos.piece_at(to_a1a0 as u8).unwrap();
                    emit(
                        0,
                        color * 6,
                        from_internal,
                        to_internal,
                        attacked,
                        &mut stm,
                        &mut ntm,
                    );
                    emit(
                        1,
                        color * 6,
                        from_internal,
                        to_internal,
                        attacked,
                        &mut stm,
                        &mut ntm,
                    );
                }

                if let Some(to_internal) = attacks::pawn_forward_square(color, from_internal)
                    && all_pawns & (1u64 << to_internal) != 0
                {
                    let to_a1a0 = to_internal ^ 56;
                    let attacked = pos.piece_at(to_a1a0 as u8).unwrap();
                    emit(
                        0,
                        color * 6,
                        from_internal,
                        to_internal,
                        attacked,
                        &mut stm,
                        &mut ntm,
                    );
                    emit(
                        1,
                        color * 6,
                        from_internal,
                        to_internal,
                        attacked,
                        &mut stm,
                        &mut ntm,
                    );
                }
            }
        }

        for color in 0..2u32 {
            for piece_type in 1..5u32 {
                let piece = color * 6 + piece_type;
                let mut pieces = pos.bb[piece as usize];
                while pieces != 0 {
                    let from_a1a0 = pieces.trailing_zeros();
                    pieces &= pieces - 1;
                    let from_internal = from_a1a0 ^ 56;

                    let mut attacks =
                        attacks::slider_attacks(piece_type, from_internal, occupancy) & occupancy;
                    while attacks != 0 {
                        let to_internal = attacks.trailing_zeros();
                        attacks &= attacks - 1;
                        let to_a1a0 = to_internal ^ 56;
                        let attacked = pos.piece_at(to_a1a0 as u8).unwrap();
                        emit(
                            0,
                            piece,
                            from_internal,
                            to_internal,
                            attacked,
                            &mut stm,
                            &mut ntm,
                        );
                        emit(
                            1,
                            piece,
                            from_internal,
                            to_internal,
                            attacked,
                            &mut stm,
                            &mut ntm,
                        );
                    }
                }
            }
        }

        let len = stm.len().max(ntm.len());
        for i in 0..len {
            let s = stm.get(i).copied().unwrap_or(INACTIVE);
            let n = ntm.get(i).copied().unwrap_or(INACTIVE);
            f(s, n);
        }
    }
}

impl SparseInputType for EmberFullThreats {
    type RequiredDataType = EmberPosition;

    fn num_inputs(&self) -> usize {
        THREAT_DIMS
    }

    fn max_active(&self) -> usize {
        256
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        self.emit_threat_pairs(pos, f);
    }

    fn shorthand(&self) -> String {
        "EmberFullThreats".to_string()
    }

    fn description(&self) -> String {
        "Ember Full Threats (60720 inputs)".to_string()
    }
}

#[derive(Clone, Debug, Default)]
pub struct EmberHalfKAv2Hm;

impl EmberHalfKAv2Hm {
    pub fn new() -> Self {
        Self
    }
}

impl SparseInputType for EmberHalfKAv2Hm {
    type RequiredDataType = EmberPosition;

    fn num_inputs(&self) -> usize {
        PSQ_DIMS
    }

    fn max_active(&self) -> usize {
        32
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let stm_persp = if pos.is_white_to_move { 0 } else { 1 };
        let ntm_persp = 1 - stm_persp;
        let white_ksq_a1a0 = if pos.is_white_to_move {
            pos.us_ksq as u32
        } else {
            pos.them_ksq as u32
        };
        let black_ksq_a1a0 = if pos.is_white_to_move {
            pos.them_ksq as u32
        } else {
            pos.us_ksq as u32
        };
        let k_stm = if stm_persp == 0 {
            white_ksq_a1a0 ^ 56
        } else {
            black_ksq_a1a0 ^ 56
        };
        let k_ntm = if ntm_persp == 0 {
            white_ksq_a1a0 ^ 56
        } else {
            black_ksq_a1a0 ^ 56
        };

        for piece in 0..12u32 {
            let mut pieces = pos.bb[piece as usize];
            while pieces != 0 {
                let square_a1a0 = pieces.trailing_zeros();
                pieces &= pieces - 1;
                let square_internal = square_a1a0 ^ 56;

                let idx_stm = halfka_index(stm_persp, square_internal, piece, k_stm);
                let idx_ntm = halfka_index(ntm_persp, square_internal, piece, k_ntm);
                debug_assert!(idx_stm < PSQ_DIMS);
                debug_assert!(idx_ntm < PSQ_DIMS);
                f(idx_stm, idx_ntm);
            }
        }
    }

    fn shorthand(&self) -> String {
        "EmberHalfKAv2Hm".to_string()
    }

    fn description(&self) -> String {
        "Ember HalfKAv2_hm^ (22528 inputs)".to_string()
    }
}

pub struct EmberCombinedFeatures {
    threats: EmberFullThreats,
    halfka: EmberHalfKAv2Hm,
}

pub const COMBINED_DIMS: usize = THREAT_DIMS + PSQ_DIMS;

impl EmberCombinedFeatures {
    pub fn new() -> Self {
        Self {
            threats: EmberFullThreats::new(),
            halfka: EmberHalfKAv2Hm::new(),
        }
    }

    pub fn collect(&self, pos: &EmberPosition, stm: &mut Vec<usize>, ntm: &mut Vec<usize>) {
        stm.clear();
        ntm.clear();
        self.threats.emit_threat_pairs(pos, |s, n| {
            if s != INACTIVE {
                stm.push(s);
            }
            if n != INACTIVE {
                ntm.push(n);
            }
        });
        self.halfka.map_features(pos, |s, n| {
            stm.push(s);
            ntm.push(n);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halfka_index_in_range() {
        let idx = halfka_index(0, 36, 0, 60);
        assert!(idx < PSQ_DIMS);
    }

    #[test]
    fn threat_lut_builds() {
        let lut = ThreatLut::build();
        let idx = lut.make_index(0, 0, 0, 1, 0, 4);
        assert!(idx <= THREAT_DIMS);
    }
}
