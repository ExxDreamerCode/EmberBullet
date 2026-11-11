use sfbinpack::{
    ChunkReader, TrainingDataEntry,
    chess::{color::Color, coords::Square, piecetype::PieceType},
};
use std::fs::File;
use std::sync::Arc;
use std::sync::mpsc::SyncSender;

#[derive(Clone, Copy, Debug, Default)]
pub struct EmberPosition {
    pub bb: [u64; 12],
    pub us_ksq: u8,
    pub them_ksq: u8,
    pub is_white_to_move: bool,
    pub score: i16,
    pub result: f32,
    pub ply: u16,
}

impl EmberPosition {
    pub fn piece_count(&self) -> u32 {
        self.bb.iter().map(|b| b.count_ones()).sum()
    }

    pub fn piece_at(&self, square: u8) -> Option<u32> {
        let bit = 1u64 << square;
        for (p, bb) in self.bb.iter().enumerate() {
            if bit & bb != 0 {
                return Some(p as u32);
            }
        }
        None
    }
}

pub fn entry_to_position(entry: &TrainingDataEntry) -> EmberPosition {
    let pos = &entry.pos;
    let is_white = pos.side_to_move() == Color::White;

    let mut bb = [0u64; 12];
    let occ = pos.occupied().bits();

    let mut us_ksq: u8 = 0;
    let mut them_ksq: u8 = 0;

    let mut occ2 = occ;
    while occ2 != 0 {
        let sq_internal = occ2.trailing_zeros() as u8;
        occ2 &= occ2 - 1;
        let sq = sq_internal;
        let bit = 1u64 << sq;
        let piece = pos.piece_at(Square::new(sq_internal as u32));
        let (color, pt) = (piece.color(), piece.piece_type());
        let ember_piece = (pt as u32) + if color == Color::White { 0 } else { 6 };
        bb[ember_piece as usize] |= bit;

        if pt == PieceType::King {
            let sq_a1a0 = sq;
            if color == Color::White {
                if is_white {
                    us_ksq = sq_a1a0;
                } else {
                    them_ksq = sq_a1a0;
                }
            } else {
                if is_white {
                    them_ksq = sq_a1a0;
                } else {
                    us_ksq = sq_a1a0;
                }
            }
        }
    }

    let score = entry.score;
    let result = f32::from(1 + entry.result) / 2.0;

    EmberPosition {
        bb,
        us_ksq,
        them_ksq,
        is_white_to_move: is_white,
        score,
        result,
        ply: entry.ply,
    }
}

pub type PositionFilter = Arc<dyn Fn(&TrainingDataEntry) -> bool + Send + Sync>;

#[derive(Clone)]
pub struct BinpackReader {
    file_paths: Vec<String>,
    buffer_size: usize,
    threads: usize,
    filter: PositionFilter,
}

impl BinpackReader {
    pub fn new(
        path: &str,
        buffer_size_mb: usize,
        threads: usize,
        filter: impl Fn(&TrainingDataEntry) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self::new_concat_multiple(&[path], buffer_size_mb, threads, filter)
    }

    pub fn new_concat_multiple(
        paths: &[&str],
        buffer_size_mb: usize,
        threads: usize,
        filter: impl Fn(&TrainingDataEntry) -> bool + Send + Sync + 'static,
    ) -> Self {
        let buffer_size = buffer_size_mb * 1024 * 1024 / std::mem::size_of::<EmberPosition>() / 2;
        Self {
            file_paths: paths.iter().map(|x| x.to_string()).collect(),
            buffer_size: buffer_size.max(1),
            threads: threads.max(1),
            filter: Arc::new(filter),
        }
    }

    pub fn read_chunks<F: FnMut(&[EmberPosition]) -> bool>(&self, _skip_count: usize, mut f: F) {
        use sfbinpack::read_chunk_into;
        use std::io::BufReader;
        use std::sync::mpsc;
        use std::thread;

        let file_paths = self.file_paths.clone();
        let buffer_size = self.buffer_size;
        let threads = self.threads;
        let filter = self.filter.clone();
        let reader_buffer_size = threads.max(1);

        let (reader_sender, reader_receiver) = mpsc::sync_channel::<Vec<Vec<u8>>>(4);
        let (reader_msg_sender, reader_msg_receiver) = mpsc::sync_channel::<bool>(1);

        thread::spawn(move || {
            'dataloading: loop {
                for file_path in &file_paths {
                    let file = match File::open(file_path.as_str()) {
                        Ok(f) => f,
                        Err(_) => panic!("binpack file must exist: {file_path}"),
                    };
                    let mut reader = BufReader::new(file);
                    let mut games = Vec::new();
                    let mut chunk = Vec::new();

                    while read_chunk_into(&mut reader, &mut chunk).unwrap_or(false) {
                        games.push(std::mem::take(&mut chunk));

                        if games.len() == reader_buffer_size {
                            if reader_msg_receiver.try_recv().unwrap_or(false)
                                || reader_sender.send(games).is_err()
                            {
                                break 'dataloading;
                            }

                            games = Vec::new();
                        }
                    }

                    if !games.is_empty()
                        && (reader_msg_receiver.try_recv().unwrap_or(false)
                            || reader_sender.send(games).is_err())
                    {
                        break 'dataloading;
                    }
                }
            }
        });

        let (converted_sender, converted_receiver) =
            mpsc::sync_channel::<Vec<EmberPosition>>(4 * threads);
        let (converted_msg_sender, converted_msg_receiver) = mpsc::sync_channel::<bool>(1);

        thread::spawn(move || {
            let filter = &filter;
            let mut should_break = false;
            'dataloading: while let Ok(chunks) = reader_receiver.recv() {
                if should_break || converted_msg_receiver.try_recv().unwrap_or(false) {
                    reader_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }

                should_break = convert_buffer(threads, &converted_sender, &chunks, filter);

                if should_break {
                    reader_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }
            }
        });

        let (buffer_sender, buffer_receiver) = mpsc::sync_channel::<Vec<EmberPosition>>(0);
        let (buffer_msg_sender, buffer_msg_receiver) = mpsc::sync_channel::<bool>(1);

        thread::spawn(move || {
            let mut shuffle_buffer = Vec::with_capacity(buffer_size);

            'dataloading: while let Ok(converted) = converted_receiver.recv() {
                for entry in converted {
                    shuffle_buffer.push(entry);

                    if shuffle_buffer.len() == buffer_size {
                        shuffle(&mut shuffle_buffer);

                        if buffer_msg_receiver.try_recv().unwrap_or(false)
                            || buffer_sender.send(shuffle_buffer).is_err()
                        {
                            converted_msg_sender.send(true).unwrap();
                            break 'dataloading;
                        }

                        shuffle_buffer = Vec::with_capacity(buffer_size);
                    }
                }
            }
        });

        'dataloading: while let Ok(shuffle_buffer) = buffer_receiver.recv() {
            if !f(&shuffle_buffer) {
                buffer_msg_sender.send(true).unwrap();
                break 'dataloading;
            }
        }
    }
}

fn shuffle(buffer: &mut [EmberPosition]) {
    let mut rng = SimpleRng::new();
    for i in (0..buffer.len()).rev() {
        let idx = rng.rand_range(0..=i as u64) as usize;
        buffer.swap(idx, i);
    }
}

fn convert_buffer(
    threads: usize,
    sender: &SyncSender<Vec<EmberPosition>>,
    chunks: &[Vec<u8>],
    filter: &PositionFilter,
) -> bool {
    let chunk_size = chunks.len().div_ceil(threads);
    let mut should_break = false;

    std::thread::scope(|s| {
        let mut handles = Vec::new();

        for chunk_group in chunks.chunks(chunk_size) {
            let this_sender = sender.clone();
            let handle = s.spawn(move || {
                let mut buffer = Vec::new();

                for chunk in chunk_group {
                    let mut reader = ChunkReader::default();

                    while reader.has_next(chunk) {
                        let entry = reader.next(chunk);
                        if filter(&entry) {
                            buffer.push(entry_to_position(&entry));
                        }
                    }
                }

                this_sender.send(buffer).is_err()
            });
            handles.push(handle);
        }

        for handle in handles {
            should_break |= handle.join().unwrap_or(true);
        }
    });

    should_break
}

#[derive(Clone)]
pub struct SimpleRng(u64);

impl SimpleRng {
    pub fn new() -> Self {
        Self(0x243f_6a88_85a3_08d3)
    }

    pub fn rand_range(&mut self, range: std::ops::RangeInclusive<u64>) -> u64 {
        let lo = *range.start();
        let hi = *range.end();
        let span = hi.saturating_sub(lo);
        if span == 0 {
            return lo;
        }
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let value = self.0 >> 33;
        lo + value % (span + 1)
    }
}

impl Default for SimpleRng {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sfbinpack::chess::r#move::Move;
    use sfbinpack::chess::position::Position;

    #[test]
    fn entry_to_position_startpos() {
        let pos =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let entry = TrainingDataEntry {
            pos,
            mv: Move::default(),
            score: 0,
            ply: 0,
            result: 1,
        };
        let p = entry_to_position(&entry);
        assert!(p.is_white_to_move);
        assert_eq!(p.piece_count(), 32);
        assert_eq!(p.us_ksq, 4);
        assert_eq!(p.them_ksq, 60);
        assert_eq!(p.bb[0], 0x00FF << 8);
        assert_eq!(p.bb[6], 0x00FF << 48);
    }

    #[test]
    fn entry_to_position_black_to_move() {
        let pos =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1").unwrap();
        let entry = TrainingDataEntry {
            pos,
            mv: Move::default(),
            score: 10,
            ply: 1,
            result: 0,
        };
        let p = entry_to_position(&entry);
        assert!(!p.is_white_to_move);
        assert_eq!(p.us_ksq, 60);
        assert_eq!(p.them_ksq, 4);
    }
}
