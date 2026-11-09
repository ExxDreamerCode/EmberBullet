const THREAT_DIMS: usize = 60_720;
#[allow(dead_code)]
const PSQ_DIMS: usize = 22_528;

const NUM_VALID_TARGETS: [u32; 12] = [6, 10, 8, 8, 10, 0, 6, 10, 8, 8, 10, 0];

const THREAT_MAP: [[i32; 6]; 6] = [
    [0, 1, -1, 2, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [-1, -1, -1, -1, -1, -1],
];

#[derive(Clone, Debug, Default)]
pub struct OracleBoard {
    pub bb: [u64; 12],
    pub side_to_move: u32,
    pub castling: u32,
    pub ep: Option<u32>,
    pub halfmove: u32,
    pub fullmove: u32,
}

impl OracleBoard {
    pub fn piece_count(&self) -> u32 {
        self.bb.iter().map(|b| b.count_ones()).sum()
    }

    pub fn piece_at(&self, square: u32) -> Option<u32> {
        let bit = 1u64 << square;
        for (p, bb) in self.bb.iter().enumerate() {
            if bit & bb != 0 {
                return Some(p as u32);
            }
        }
        None
    }

    pub fn occupancy(&self) -> u64 {
        self.bb.iter().copied().fold(0, |a, b| a | b)
    }

    pub fn all_pawns(&self) -> u64 {
        self.bb[0] | self.bb[6]
    }

    pub fn find_king(&self, perspective: u32) -> u32 {
        self.bb[if perspective == 0 { 5 } else { 11 }].trailing_zeros()
    }
}

fn to_v2_square(square: u32) -> u32 {
    square ^ 56
}

fn halfka_orientation(king_square: u32) -> u32 {
    if king_square & 4 == 0 { 7 } else { 0 }
}

fn threat_orientation(king_square: u32) -> u32 {
    if king_square & 4 == 0 { 0 } else { 7 }
}

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
        _ => unreachable!(),
    }
}

pub fn halfka_index(perspective: u32, square: u32, piece: u32, king_square: u32) -> usize {
    let square = to_v2_square(square);
    let king_square = to_v2_square(king_square);
    let flip = 56 * perspective;
    let oriented_king = king_square ^ flip;
    let king_rank = oriented_king / 8;
    let king_file = oriented_king % 8;
    let king_bucket = (7 - king_rank) * 4 + king_file.min(7 - king_file);
    ((square ^ halfka_orientation(king_square) ^ flip)
        + halfka_piece_base(perspective, piece)
        + king_bucket * 11 * 64) as usize
}

fn pawn_attacks_bb(color: u32, square: u32) -> u64 {
    let bb = 1u64 << square;
    const A_FILE: u64 = 0x0101_0101_0101_0101;
    const H_FILE: u64 = 0x8080_8080_8080_8080;
    if color == 0 {
        ((bb & !H_FILE) >> 7) | ((bb & !A_FILE) >> 9)
    } else {
        ((bb & !H_FILE) << 9) | ((bb & !A_FILE) << 7)
    }
}

fn pawn_forward_square(color: u32, square: u32) -> Option<u32> {
    match color {
        0 if square >= 8 => Some(square - 8),
        1 if square < 56 => Some(square + 8),
        _ => None,
    }
}

fn attacks_bb(piece_type: u32, from: u32, occupancy: u64) -> u64 {
    match piece_type {
        1 => knight_attacks(from),
        2 => bishop_attacks(from, occupancy),
        3 => rook_attacks(from, occupancy),
        4 => bishop_attacks(from, occupancy) | rook_attacks(from, occupancy),
        5 => king_attacks(from),
        _ => 0,
    }
}

fn knight_attacks(sq: u32) -> u64 {
    let rank = sq / 8;
    let file = sq % 8;
    let mut m = 0u64;
    for (dr, df) in [
        (2, 1),
        (2, -1),
        (-2, 1),
        (-2, -1),
        (1, 2),
        (1, -2),
        (-1, 2),
        (-1, -2),
    ] {
        let nr = rank as i32 + dr;
        let nf = file as i32 + df;
        if (0..8i32).contains(&nr) && (0..8i32).contains(&nf) {
            m |= 1u64 << (nr * 8 + nf);
        }
    }
    m
}

fn king_attacks(sq: u32) -> u64 {
    let rank = sq / 8;
    let file = sq % 8;
    let mut m = 0u64;
    for dr in -1..=1 {
        for df in -1..=1 {
            if dr == 0 && df == 0 {
                continue;
            }
            let nr = rank as i32 + dr;
            let nf = file as i32 + df;
            if (0..8i32).contains(&nr) && (0..8i32).contains(&nf) {
                m |= 1u64 << (nr * 8 + nf);
            }
        }
    }
    m
}

fn slider_ray(sq: u32, dr: i32, df: i32, occupancy: u64) -> u64 {
    let rank = sq / 8;
    let file = sq % 8;
    let mut m = 0u64;
    let mut step = 1;
    loop {
        let nr = rank as i32 + dr * step;
        let nf = file as i32 + df * step;
        if !(0..8i32).contains(&nr) || !(0..8i32).contains(&nf) {
            break;
        }
        let bit = 1u64 << (nr * 8 + nf);
        m |= bit;
        if bit & occupancy != 0 {
            break;
        }
        step += 1;
    }
    m
}

fn bishop_attacks(sq: u32, occupancy: u64) -> u64 {
    slider_ray(sq, -1, -1, occupancy)
        | slider_ray(sq, -1, 1, occupancy)
        | slider_ray(sq, 1, -1, occupancy)
        | slider_ray(sq, 1, 1, occupancy)
}

fn rook_attacks(sq: u32, occupancy: u64) -> u64 {
    slider_ray(sq, -1, 0, occupancy)
        | slider_ray(sq, 1, 0, occupancy)
        | slider_ray(sq, 0, -1, occupancy)
        | slider_ray(sq, 0, 1, occupancy)
}

fn pseudo_attacks(piece: u32, from: u32) -> u64 {
    let piece_type = piece % 6;
    if piece_type == 0 {
        let bb = 1u64 << from;
        if piece / 6 == 0 {
            (bb >> 8) | ((bb & 0x7f7f_7f7f_7f7f_7f7f) >> 7) | ((bb & 0xfefefefefefefefe) >> 9)
        } else {
            (bb << 8) | ((bb & 0x7f7f_7f7f_7f7f_7f7f) << 9) | ((bb & 0xfefefefefefefefe) << 7)
        }
    } else {
        let ember_from = to_v2_square(from);
        attacks_bb(piece_type, ember_from, 0).swap_bytes()
    }
}

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
                    pseudo_attacks(piece, from)
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
                    pseudo_attacks(piece, from).count_ones() as usize
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
                    pseudo_attacks(att, from).count_ones() as usize
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
        let from = to_v2_square(from);
        let to = to_v2_square(to);
        let king_square = to_v2_square(king_square);
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

pub fn collect_threat_indices(board: &OracleBoard) -> (Vec<u16>, Vec<u16>) {
    let lut = ThreatLut::build();
    let king_squares = [board.find_king(0), board.find_king(1)];
    let occupancy = board.occupancy();
    let all_pawns = board.all_pawns();
    let mut out_white = Vec::new();
    let mut out_black = Vec::new();

    macro_rules! push {
        ($out:expr, $persp:expr, $piece:expr, $from:expr, $to:expr, $attacked:expr, $king:expr) => {
            let idx = lut.make_index($persp, $piece, $from, $to, $attacked, $king);
            if idx < THREAT_DIMS {
                $out.push(idx as u16);
            }
        };
    }

    for color in 0..2u32 {
        let pawns = board.bb[(color * 6) as usize];
        let mut pieces = pawns;
        while pieces != 0 {
            let from = pieces.trailing_zeros();
            pieces &= pieces - 1;
            let mut attacks = pawn_attacks_bb(color, from) & occupancy;
            while attacks != 0 {
                let to = attacks.trailing_zeros();
                attacks &= attacks - 1;
                let attacked = board.piece_at(to).unwrap();
                push!(out_white, 0, color * 6, from, to, attacked, king_squares[0]);
                push!(out_black, 1, color * 6, from, to, attacked, king_squares[1]);
            }
            if let Some(to) = pawn_forward_square(color, from)
                && all_pawns & (1u64 << to) != 0
            {
                let attacked = board.piece_at(to).unwrap();
                push!(out_white, 0, color * 6, from, to, attacked, king_squares[0]);
                push!(out_black, 1, color * 6, from, to, attacked, king_squares[1]);
            }
        }
        for piece_type in 1..5u32 {
            let piece = color * 6 + piece_type;
            let mut pieces = board.bb[piece as usize];
            while pieces != 0 {
                let from = pieces.trailing_zeros();
                pieces &= pieces - 1;
                let mut attacks = attacks_bb(piece_type, from, occupancy) & occupancy;
                while attacks != 0 {
                    let to = attacks.trailing_zeros();
                    attacks &= attacks - 1;
                    let attacked = board.piece_at(to).unwrap();
                    push!(out_white, 0, piece, from, to, attacked, king_squares[0]);
                    push!(out_black, 1, piece, from, to, attacked, king_squares[1]);
                }
            }
        }
    }
    (out_white, out_black)
}

pub fn collect_halfka_indices(board: &OracleBoard) -> (Vec<usize>, Vec<usize>) {
    let mut out_white = Vec::new();
    let mut out_black = Vec::new();
    for piece in 0..12u32 {
        let mut pieces = board.bb[piece as usize];
        while pieces != 0 {
            let square = pieces.trailing_zeros();
            pieces &= pieces - 1;
            let ksq_w = board.find_king(0);
            let ksq_b = board.find_king(1);
            out_white.push(halfka_index(0, square, piece, ksq_w));
            out_black.push(halfka_index(1, square, piece, ksq_b));
        }
    }
    (out_white, out_black)
}

pub fn parse_fen(fen: &str) -> Result<OracleBoard, String> {
    let parts: Vec<&str> = fen.split_whitespace().collect();
    if parts.len() < 2 {
        return Err("bad FEN".into());
    }
    let board_str = parts[0];
    let stm = parts[1];

    let mut board = OracleBoard {
        side_to_move: if stm == "w" { 0 } else { 1 },
        ..OracleBoard::default()
    };

    let mut rank = 0;
    let mut file = 0;
    for ch in board_str.chars() {
        if ch == '/' {
            rank += 1;
            file = 0;
        } else if ch.is_ascii_digit() {
            file += ch.to_digit(10).unwrap();
        } else {
            let color = if ch.is_uppercase() { 0 } else { 1 };
            let pt = match ch.to_ascii_lowercase() {
                'p' => 0,
                'n' => 1,
                'b' => 2,
                'r' => 3,
                'q' => 4,
                'k' => 5,
                _ => return Err(format!("bad piece {ch}")),
            };
            let piece = pt + color * 6;
            let square = rank * 8 + file;
            board.bb[piece as usize] |= 1u64 << square;
            file += 1;
        }
    }
    Ok(board)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_startpos() {
        let board = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        assert_eq!(board.side_to_move, 0);
        assert_eq!(board.piece_count(), 32);
        assert_eq!(board.find_king(0), 60);
        assert_eq!(board.find_king(1), 4);
    }
}
