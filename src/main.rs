use std::time::Instant;

use prng_cracker::{JavaNextDoubleCracker, JavaNextDoubleLCG};

fn main() {
    let seed = 3454324;
    let mut my_lcg = JavaNextDoubleLCG::new(seed);
    let amount = 10;
    let error = 1.0 / 64.0;

    println!(
        "Starting with configuration: (Amount of measurements: {}, Error: +-{})",
        amount, error
    );

    println!("Creating cracker...");
    let instant = Instant::now();
    let cracker = JavaNextDoubleCracker::new(amount);
    let elapsed = instant.elapsed();
    println!("Done creating cracker. Took {}ms.", elapsed.as_millis());

    println!("Seed: {}, Measurements:", seed);
    let mut m = Vec::with_capacity(amount);

    for i in 0..amount {
        let val = my_lcg.next();

        let min = (val - error).max(0_f64);
        let max = (val + error).min(1_f64);

        println!("[{}]: {} ({} - {})", i + 1, val, min, max);

        m.push((min, max));
    }

    println!("Starting cracking process...");
    let iter = cracker.crack(&m, true);
    println!("Searching the space...");

    let mut counter = 0;
    for mut lcg in iter {
        counter += 1;
        println!(
            "(LCG-{}) Found matching LCG! Seed: {}, Results:",
            counter,
            lcg.get_seed()
        );
        for i in 0..(amount + 2) {
            println!("(LCG-{}) [{}] {}", counter, i + 1, lcg.next());
        }
    }

    println!("Done cracking. {} result(s) found.", counter);
}
