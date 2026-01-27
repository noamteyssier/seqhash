use std::{
    fs::File,
    io::{BufRead, BufReader},
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::Result;
use binseq::{BinseqReader, ParallelProcessor, ParallelReader};
use clap::Parser;
use log::info;
use seqhash::SeqHash;

#[derive(Clone)]
struct Counter {
    /// Guide library
    library: Arc<SeqHash>,

    /// local counts of guides
    t_counts: Vec<usize>,

    /// global counts of guides
    counts: Arc<Mutex<Vec<usize>>>,
}
impl Counter {
    pub fn new(guides: &[Vec<u8>]) -> Result<Self> {
        let library = SeqHash::new(guides)?;
        let t_counts = vec![0; library.num_parents()];
        Ok(Self {
            counts: Arc::new(Mutex::new(t_counts.clone())),
            library: Arc::new(library),
            t_counts,
        })
    }

    pub fn find_parent(&self, seq: &[u8]) -> Option<usize> {
        self.library
            .query_sliding(seq)
            .map(|(m, _pos)| m.parent_idx())
    }

    pub fn pprint(&self) {
        self.library
            .iter_parents()
            .zip(self.counts.lock().unwrap().iter())
            .for_each(|(parent, count)| {
                println!("{}\t{}", std::str::from_utf8(parent).unwrap(), count);
            });
    }
}
impl ParallelProcessor for Counter {
    fn process_record<R: binseq::BinseqRecord>(&mut self, record: R) -> binseq::Result<()> {
        if let Some(parent_idx) = self.find_parent(record.sseq()) {
            self.t_counts[parent_idx] += 1;
        }
        if record.is_paired() {
            if let Some(parent_idx) = self.find_parent(record.xseq()) {
                self.t_counts[parent_idx] += 1;
            }
        }
        Ok(())
    }

    fn on_batch_complete(&mut self) -> binseq::Result<()> {
        for (global, local) in self
            .counts
            .lock()
            .unwrap()
            .iter_mut()
            .zip(self.t_counts.iter_mut())
        {
            *global += *local;
            *local = 0;
        }
        Ok(())
    }
}

#[derive(Parser, Debug)]
struct Cli {
    /// Input BINSEQ to process
    #[clap(required = true)]
    input: String,

    /// Input file with guide sequences (one per line)
    #[clap(required = true)]
    guides: String,

    /// Number of threads to use for processing (0: all)
    #[clap(short = 'T', long, default_value_t = 0)]
    threads: usize,
}

/// Load guide sequences from a file
fn load_guides(path: &str) -> Result<Vec<Vec<u8>>> {
    let reader = File::open(path).map(BufReader::new)?;
    let mut guides = Vec::new();
    for line in reader.lines() {
        let line = line?;
        guides.push(line.into_bytes());
    }
    Ok(guides)
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::builder()
        .format_timestamp_millis()
        .filter_level(log::LevelFilter::Info)
        .init();

    // Parse command line arguments
    let args = Cli::parse();

    info!("Loading guides from path: {}", args.guides);
    let guides = load_guides(&args.guides)?;
    info!("Loaded {} guides", guides.len());

    info!("Initializing counter");
    let proc = Counter::new(&guides)?;
    info!("Counter initialized");

    info!("Loading reader from path: {}", args.input);
    let reader = BinseqReader::new(&args.input)?;
    let num_reads = reader.num_records()?;

    info!("Processing input");
    let start = Instant::now();
    reader.process_parallel(proc.clone(), args.threads)?;
    let elapsed = start.elapsed();
    info!(
        "Processed {} reads in {:?}: {}M/s",
        num_reads,
        elapsed,
        num_reads as f64 / 1_000_000.0 / elapsed.as_secs_f64(),
    );

    info!("Processing complete");
    proc.pprint();
    Ok(())
}
