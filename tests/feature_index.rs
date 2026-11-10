use std::collections::BTreeSet;

use bullet_lib::game::inputs::SparseInputType;
use ember_bullet_v2_trainer::features::{EmberFullThreats, EmberHalfKAv2Hm};
use ember_bullet_v2_trainer::oracle::{collect_halfka_indices, collect_threat_indices, parse_fen};
use ember_bullet_v2_trainer::position::{SimpleRng, entry_to_position};

fn playout_positions(_rng: &mut SimpleRng, count: usize) -> Vec<String> {
    let pool = [
        "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR w KQkq - 0 1",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK1R w KQkq - 0 1",
        "rnbqk2r/pppp1ppp/5n2/2b1p3/2P1P3/2N2N2/PP1P1PPP/R1BQKB1R w KQkq - 0 1",
        "rnb1kbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/ppppp1pp/5p2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK1R b KQkq - 0 1",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK1R b KQkq - 0 1",
        "rnbqk2r/pppp1ppp/5n2/2b1p3/2P1P3/2N2N2/PP1P1PPP/R1BQKB1R b KQkq - 0 1",
        "rnb1kbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "rnbqkbnr/ppppp1pp/5p2/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "rnbqkbnr/pppp1ppp/4n3/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
        "rnbqkbnr/pppp1ppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppp1ppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 0 1",
    ];
    let mut out = Vec::new();
    for i in 0..count {
        out.push(pool[(i * 7 + 3) % pool.len()].to_string());
    }
    out
}

fn entry_from_fen(fen: &str) -> sfbinpack::TrainingDataEntry {
    use sfbinpack::chess::position::Position;
    let pos = Position::from_fen(fen).unwrap_or_else(|e| panic!("bad FEN {fen}: {e:?}"));
    sfbinpack::TrainingDataEntry {
        pos,
        mv: Default::default(),
        score: 0,
        ply: 0,
        result: 1,
    }
}

fn trainer_threats(
    pos: &ember_bullet_v2_trainer::position::EmberPosition,
    threats: &EmberFullThreats,
) -> BTreeSet<usize> {
    let mut set = BTreeSet::new();
    threats.map_features(pos, |s, n| {
        if s != usize::MAX {
            set.insert(s);
        }
        if n != usize::MAX {
            set.insert(n);
        }
    });
    set
}

fn trainer_halfka(
    pos: &ember_bullet_v2_trainer::position::EmberPosition,
    halfka: &EmberHalfKAv2Hm,
) -> (BTreeSet<usize>, BTreeSet<usize>) {
    let mut stm = BTreeSet::new();
    let mut ntm = BTreeSet::new();
    halfka.map_features(pos, |s, n| {
        stm.insert(s);
        ntm.insert(n);
    });
    (stm, ntm)
}

#[test]
fn feature_index_bit_exact() {
    let threats = EmberFullThreats::new();
    let halfka = EmberHalfKAv2Hm::new();
    let mut rng = SimpleRng::new();

    let mut fens: Vec<String> = vec![
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1".to_string(),
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1".to_string(),
        "rnbqkbnr/pppppppp/8/8/4p3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1".to_string(),
        "4k3/8/8/8/8/8/P6P/K7 w - - 0 1".to_string(),
        "4K3/8/8/8/8/8/p6p/k7 b - - 0 1".to_string(),
        "4k3/P7/8/8/8/8/p7/K7 w - - 0 1".to_string(),
        "4k3/8/8/8/8/8/8/K6k w - - 0 1".to_string(),
        "4k3/8/8/8/8/8/8/K3k4 b - - 0 1".to_string(),
        "7k/8/8/8/8/8/8/7K w - - 0 1".to_string(),
        "7K/8/8/8/8/8/8/7k b - - 0 1".to_string(),
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2P1P3/2N2N2/PP1P1PPP/R1BQKB1R w KQkq - 0 1".to_string(),
        "rnbqkbnr/pppp1ppp/8/4p3/4Pp3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 1".to_string(),
        "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1".to_string(),
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
    ];
    fens.extend(playout_positions(&mut rng, 190));

    assert!(fens.len() >= 200, "corpus too small: {}", fens.len());

    for fen in [
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        "3k4/8/8/8/8/8/8/3K4 w - - 0 1",
        "2k5/8/8/8/8/8/8/2K5 w - - 0 1",
        "1k6/8/8/8/8/8/8/1K6 w - - 0 1",
        "k7/8/8/8/8/8/8/K7 w - - 0 1",
        "6k1/8/8/8/8/8/8/6K1 w - - 0 1",
        "5k2/8/8/8/8/8/8/5K2 w - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 b - - 0 1",
    ] {
        if !fens.iter().any(|f| f == fen) {
            fens.push(fen.to_string());
        }
    }

    for fen in &fens {
        let pos = entry_to_position(&entry_from_fen(fen));
        let oracle_board = parse_fen(fen).unwrap_or_else(|e| panic!("bad FEN {fen}: {e}"));
        let (oracle_white, oracle_black) = collect_threat_indices(&oracle_board);
        let (oracle_hk_white, oracle_hk_black) = collect_halfka_indices(&oracle_board);

        let trainer_threats_set = trainer_threats(&pos, &threats);
        let (trainer_hk_stm, trainer_hk_ntm) = trainer_halfka(&pos, &halfka);

        let exp_threats: BTreeSet<usize> = if pos.is_white_to_move {
            oracle_white
                .iter()
                .chain(oracle_black.iter())
                .map(|&x| x as usize)
                .collect()
        } else {
            oracle_black
                .iter()
                .chain(oracle_white.iter())
                .map(|&x| x as usize)
                .collect()
        };

        let exp_hk_stm: BTreeSet<usize> = if pos.is_white_to_move {
            oracle_hk_white.iter().cloned().collect()
        } else {
            oracle_hk_black.iter().cloned().collect()
        };
        let exp_hk_ntm: BTreeSet<usize> = if pos.is_white_to_move {
            oracle_hk_black.iter().cloned().collect()
        } else {
            oracle_hk_white.iter().cloned().collect()
        };

        assert_eq!(
            trainer_threats_set, exp_threats,
            "threat feature mismatch for FEN {fen}"
        );
        assert_eq!(
            trainer_hk_stm, exp_hk_stm,
            "halfka stm mismatch for FEN {fen}"
        );
        assert_eq!(
            trainer_hk_ntm, exp_hk_ntm,
            "halfka ntm mismatch for FEN {fen}"
        );
    }
}
