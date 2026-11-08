const A_FILE: u64 = 0x0101_0101_0101_0101;
const H_FILE: u64 = 0x8080_8080_8080_8080;

pub const KNIGHT_ATTACKS: [u64; 64] = knight_table();

pub const KING_ATTACKS: [u64; 64] = king_table();

const fn knight_table() -> [u64; 64] {
    let mut table = [0u64; 64];
    let mut s = 0usize;
    while s < 64 {
        let rank = s / 8;
        let file = s % 8;
        let mut m = 0u64;
        if rank + 2 < 8 && file + 1 < 8 {
            m |= 1u64 << ((rank + 2) * 8 + (file + 1));
        }
        if rank + 2 < 8 && file >= 1 {
            m |= 1u64 << ((rank + 2) * 8 + (file - 1));
        }
        if rank >= 2 && file + 1 < 8 {
            m |= 1u64 << ((rank - 2) * 8 + (file + 1));
        }
        if rank >= 2 && file >= 1 {
            m |= 1u64 << ((rank - 2) * 8 + (file - 1));
        }
        if rank + 1 < 8 && file + 2 < 8 {
            m |= 1u64 << ((rank + 1) * 8 + (file + 2));
        }
        if rank + 1 < 8 && file >= 2 {
            m |= 1u64 << ((rank + 1) * 8 + (file - 2));
        }
        if rank >= 1 && file + 2 < 8 {
            m |= 1u64 << ((rank - 1) * 8 + (file + 2));
        }
        if rank >= 1 && file >= 2 {
            m |= 1u64 << ((rank - 1) * 8 + (file - 2));
        }
        table[s] = m;
        s += 1;
    }
    table
}

const fn king_table() -> [u64; 64] {
    let mut table = [0u64; 64];
    let mut s = 0usize;
    while s < 64 {
        let rank = s / 8;
        let file = s % 8;
        let mut m = 0u64;
        let mut dr = -1isize;
        while dr <= 1 {
            let mut df = -1isize;
            while df <= 1 {
                if dr != 0 || df != 0 {
                    let nr = rank as isize + dr;
                    let nf = file as isize + df;
                    if nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
                        m |= 1u64 << (nr * 8 + nf);
                    }
                }
                df += 1;
            }
            dr += 1;
        }
        table[s] = m;
        s += 1;
    }
    table
}

fn ray_mask(square: u32, dir: usize, occupancy: u64) -> u64 {
    let rank = square / 8;
    let file = square % 8;
    let mut blocked = 0u64;
    let mut step = 1u32;
    while step <= 7 {
        let bit = match dir {
            0 if rank >= step => 1u64 << ((rank - step) * 8 + file),
            1 if rank + step <= 7 => 1u64 << ((rank + step) * 8 + file),
            2 if file >= step => 1u64 << (rank * 8 + (file - step)),
            3 if file + step <= 7 => 1u64 << (rank * 8 + (file + step)),
            4 if rank >= step && file >= step => 1u64 << ((rank - step) * 8 + (file - step)),
            5 if rank + step <= 7 && file + step <= 7 => {
                1u64 << ((rank + step) * 8 + (file + step))
            }
            6 if rank >= step && file + step <= 7 => 1u64 << ((rank - step) * 8 + (file + step)),
            7 if rank + step <= 7 && file >= step => 1u64 << ((rank + step) * 8 + (file - step)),
            _ => break,
        };
        blocked |= bit;
        if bit & occupancy != 0 {
            break;
        }
        step += 1;
    }
    blocked
}

pub fn bishop_attacks(square: u32, occupancy: u64) -> u64 {
    ray_mask(square, 4, occupancy)
        | ray_mask(square, 5, occupancy)
        | ray_mask(square, 6, occupancy)
        | ray_mask(square, 7, occupancy)
}

pub fn rook_attacks(square: u32, occupancy: u64) -> u64 {
    ray_mask(square, 0, occupancy)
        | ray_mask(square, 1, occupancy)
        | ray_mask(square, 2, occupancy)
        | ray_mask(square, 3, occupancy)
}

pub fn queen_attacks(square: u32, occupancy: u64) -> u64 {
    bishop_attacks(square, occupancy) | rook_attacks(square, occupancy)
}

pub fn slider_attacks(piece_type: u32, from: u32, occupancy: u64) -> u64 {
    match piece_type {
        1 => KNIGHT_ATTACKS[from as usize],
        2 => bishop_attacks(from, occupancy),
        3 => rook_attacks(from, occupancy),
        4 => queen_attacks(from, occupancy),
        5 => KING_ATTACKS[from as usize],
        _ => 0,
    }
}

pub fn pawn_attacks_bb(color: u32, square: u32) -> u64 {
    let bb = 1u64 << square;
    if color == 0 {
        ((bb & !H_FILE) >> 7) | ((bb & !A_FILE) >> 9)
    } else {
        ((bb & !H_FILE) << 9) | ((bb & !A_FILE) << 7)
    }
}

pub fn pawn_forward_square(color: u32, square: u32) -> Option<u32> {
    match color {
        0 if square >= 8 => Some(square - 8),
        1 if square < 56 => Some(square + 8),
        _ => None,
    }
}

pub fn pseudo_attacks(piece: u32, from: u32) -> u64 {
    let piece_type = piece % 6;
    if piece_type == 0 {
        let bb = 1u64 << from;
        if piece / 6 == 0 {
            (bb >> 8) | ((bb & !H_FILE) >> 7) | ((bb & !A_FILE) >> 9)
        } else {
            (bb << 8) | ((bb & !H_FILE) << 9) | ((bb & !A_FILE) << 7)
        }
    } else {
        let ember_from = from ^ 56;
        slider_attacks(piece_type, ember_from, 0).swap_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knight_attacks_symmetric() {
        assert_eq!(KNIGHT_ATTACKS[0], (1u64 << 17) | (1u64 << 10));
    }

    #[test]
    fn white_pawn_attacks_internal() {
        assert_eq!(pawn_attacks_bb(0, 28), (1u64 << 19) | (1u64 << 21));
    }

    #[test]
    fn black_pawn_attacks_internal() {
        assert_eq!(pawn_attacks_bb(1, 28), (1u64 << 35) | (1u64 << 37));
    }

    #[test]
    fn pawn_pseudo_attacks_equals_attacks_plus_push() {
        for color in 0..2u32 {
            for sq in 0..64u32 {
                let bb = 1u64 << sq;
                let forward_bb = match color {
                    0 if sq >= 8 => bb >> 8,
                    1 if sq < 56 => bb << 8,
                    _ => 0,
                };
                let expected = pawn_attacks_bb(color, sq) | forward_bb;
                let actual = pseudo_attacks(color * 6, sq);
                assert_eq!(
                    actual, expected,
                    "color={color} sq={sq}: pseudo={actual:064b} expected={expected:064b}"
                );
            }
        }
    }
}
