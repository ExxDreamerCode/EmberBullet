use clap::Parser;
use ember_bullet_v2_trainer::cli::{Cli, run};

fn main() {
    let cli = Cli::parse();
    if let Err(err) = run(cli) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
