fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--stm") {
        gubas_rs::run_stm();
    } else {
        gubas_rs::run_simulation();
    }
}
