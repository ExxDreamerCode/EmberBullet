use ember_bullet_v2_trainer::export::{
    ExportedStack, ExportedWeights, parse_container, write_container,
};
use ember_bullet_v2_trainer::model::init_weights;

#[test]
fn leb128_known_vectors() {
    let mut buf = Vec::new();
    ember_bullet_v2_trainer::export::write_sleb128(300i64, &mut buf);
    assert_eq!(buf, vec![0xAC, 0x02]);
    buf.clear();
    ember_bullet_v2_trainer::export::write_sleb128(-1i64, &mut buf);
    assert_eq!(buf, vec![0x7F]);
    buf.clear();
    ember_bullet_v2_trainer::export::write_sleb128(-5i64, &mut buf);
    assert_eq!(buf, vec![0x7B]);
}

#[test]
fn container_self_test() {
    let exported = ExportedWeights {
        ft_bias: vec![0.0; 1024],
        threat_weights: vec![0.0; 60_720 * 1024],
        threat_psqt: vec![0.0; 60_720 * 8],
        psq_weights: vec![0.0; 22_528 * 1024],
        psqt: vec![0.0; 22_528 * 8],
        stacks: (0..8)
            .map(|_| ExportedStack {
                fc0_bias: vec![0.0; 32],
                fc0_weights: vec![0.0; 32 * 1024],
                fc1_bias: vec![0.0; 32],
                fc1_weights: vec![0.0; 32 * 64],
                fc2_bias: 0.0,
                fc2_weights: vec![0.0; 128],
            })
            .collect(),
    };

    let mut buf = Vec::new();
    write_container(&mut buf, &exported, "untrained init net").expect("writing container");

    let parsed = parse_container(&buf).expect("parsing container");

    assert_eq!(parsed.version, 0x6A448AFA);
    assert_eq!(parsed.arch_hash, 0x0256ACDF);
    assert_eq!(parsed.num_stacks, 8);
    assert_eq!(parsed.ft_bias.len(), 1024);
    assert_eq!(parsed.threat_weights.len(), 60_720 * 1024);
    assert_eq!(parsed.psq_weights.len(), 22_528 * 1024);
    assert_eq!(parsed.psqt.len(), 22_528 * 8);
}

#[test]
fn read_model_weights_compiles() {
    let model = ember_bullet_v2_trainer::model::build_model();
    let _weights = init_weights(&model.definition, 12345);
    let _ = model;
}
