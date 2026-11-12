use std::sync::mpsc;

use bullet_trainer::run::DataLoader;

use ember_bullet_v2_trainer::position::EmberPosition;

fn write_synthetic_binpack(path: &str, entries: usize) {
    use sfbinpack::{
        CompressedTrainingDataEntryWriter, TrainingDataEntry, chess::r#move::Move,
        chess::position::Position,
    };

    let file = std::fs::File::create(path).unwrap();
    let mut writer = CompressedTrainingDataEntryWriter::new(file).unwrap();

    let pos =
        Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
    for _ in 0..entries {
        let entry = TrainingDataEntry {
            pos,
            mv: Move::default(),
            score: 0,
            ply: 0,
            result: 1,
        };
        writer.write_entry(&entry).unwrap();
    }
    writer.flush_and_end();
}

fn make_mapper(lambda: f32) -> bullet_trainer::model::ModelInputsMapper<EmberPosition> {
    use ember_bullet_v2_trainer::features::EmberCombinedFeatures;
    use ember_bullet_v2_trainer::model::{
        HALFKA_DIMS, L2, MAX_ACTIVE_HALFKA, MAX_ACTIVE_THREATS, NUM_BUCKETS, THREAT_DIMS,
    };

    let features = EmberCombinedFeatures::new();

    let inputs = bullet_trainer::model::ModelInputs::default()
        .add_sparse("ft_t_stm", (THREAT_DIMS, 1), MAX_ACTIVE_THREATS)
        .add_sparse("ft_t_ntm", (THREAT_DIMS, 1), MAX_ACTIVE_THREATS)
        .add_sparse("ft_h_stm", (HALFKA_DIMS, 1), MAX_ACTIVE_HALFKA)
        .add_sparse("ft_h_ntm", (HALFKA_DIMS, 1), MAX_ACTIVE_HALFKA)
        .add_sparse("buckets", (NUM_BUCKETS, 1), 1)
        .add_sparse("skip_idx_30", (L2 * NUM_BUCKETS, 1), 1)
        .add_sparse("skip_idx_31", (L2 * NUM_BUCKETS, 1), 1)
        .add_dense("us", (1, 1))
        .add_dense("score", (1, 1))
        .add_dense("outcome", (1, 1))
        .add_dense("lambda", (1, 1));

    bullet_trainer::model::ModelInputsMapper::build(
        &inputs,
        move |pos: &EmberPosition,
              _step: bullet_trainer::run::Step,
              (
            (
                ((((((((t_stm, t_ntm), h_stm), h_ntm), buckets), skip30), skip31), us), score),
                outcome,
            ),
            lambda_buf,
        )| {
            t_stm.fill(-1);
            t_ntm.fill(-1);
            h_stm.fill(-1);
            h_ntm.fill(-1);

            let mut t_stm_idx = Vec::with_capacity(MAX_ACTIVE_THREATS);
            let mut t_ntm_idx = Vec::with_capacity(MAX_ACTIVE_THREATS);
            let mut h_stm_idx = Vec::with_capacity(MAX_ACTIVE_HALFKA);
            let mut h_ntm_idx = Vec::with_capacity(MAX_ACTIVE_HALFKA);
            features.collect(
                pos,
                &mut t_stm_idx,
                &mut t_ntm_idx,
                &mut h_stm_idx,
                &mut h_ntm_idx,
            );

            for (buf, idx) in [
                (t_stm, &t_stm_idx),
                (t_ntm, &t_ntm_idx),
                (h_stm, &h_stm_idx),
                (h_ntm, &h_ntm_idx),
            ] {
                for (dst, src) in buf.iter_mut().zip(idx.iter()) {
                    *dst = *src as i32;
                }
            }

            let bucket = ((pos.piece_count() as i32 - 1) / 4).clamp(0, 7) as usize;
            buckets[0] = bucket as i32;
            skip30[0] = (bucket * L2 + 30) as i32;
            skip31[0] = (bucket * L2 + 31) as i32;

            us[0] = 1.0;
            score[0] = f32::from(pos.score);
            outcome[0] = pos.result;
            lambda_buf[0] = lambda;
        },
    )
}

#[test]
fn dataloader_sends_exactly_scheduled_batches() {
    use bullet_trainer::reader::ReadMapLoader;
    use ember_bullet_v2_trainer::position::BinpackReader;

    let dir = std::env::temp_dir().join("ember_bullet_dataloader_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("smoke.binpack");
    let path_str = path.to_string_lossy().into_owned();
    let entries = 2 * 8 * 1024 + 4096;
    write_synthetic_binpack(&path_str, entries);

    let steps = bullet_trainer::run::TrainingSteps {
        batch_size: 1024,
        batches_per_superbatch: 8,
        start_superbatch: 1,
        end_superbatch: 2,
    };

    let device = bullet_gpu::runtime::Device::<bullet_gpu::runtime::mock::MockGpu>::new(0).unwrap();
    let pool = bullet_trainer::run::HostPool::new(device);

    let reader = BinpackReader::new(&path_str, 8, 2, |_| true);
    let loader = ReadMapLoader::new(reader, make_mapper(1.0), 2);

    let (sender, receiver) = mpsc::sync_channel::<()>(32);

    let step = bullet_trainer::run::Step::from(steps);
    let handle = std::thread::spawn(move || {
        let mut step = step;
        loader.map_batches(&pool, step, steps.batch_size, |batch| {
            drop(batch);
            if sender.send(()).is_err() {
                return true;
            }
            step.step();
            step.finished()
        })
    });

    let mut received = 0usize;
    while receiver.recv().is_ok() {
        received += 1;
    }
    let _ = handle.join();

    assert_eq!(received, 16, "loader sent {received} batches, expected 16");
}
