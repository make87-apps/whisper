use anyhow::{Context, Result};
use make87::config::load_config_from_default_env;
use make87::encodings::{Encoder, ProtobufEncoder};
use make87::interfaces::zenoh::{ConfiguredSubscriber, ZenohInterface};
use make87_messages::audio::{frame_pcm_s16le, FramePcmS16le};
use tokio::sync::mpsc;
use whisper_rs::{
    convert_integer_to_float_audio, FullParams, SamplingStrategy, WhisperContext,
    WhisperContextParameters,
};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    // --- Whisper init
    let model = std::env::var("WHISPER_MODEL")
        .unwrap_or("/models/ggml-tiny.en.bin".into());
    let ctx = WhisperContext::new_with_params(&model, WhisperContextParameters::default())
        .with_context(|| format!("loading model {model}"))?;
    let mut state = ctx.create_state().context("create whisper state")?;

    // Greedy for lowest latency
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    // tiny.en is English-only; don't auto-detect
    params.set_detect_language(false);
    params.set_language(Some("en"));
    params.set_translate(false);
    params.set_no_context(true);
    params.set_temperature(0.0);
    params.set_n_threads(std::cmp::min(num_cpus::get(), 4) as i32); // avoid contention on little boards
    params.set_print_realtime(false);
    params.set_print_progress(false);

    // --- App config with low-latency defaults
    let cfg = load_config_from_default_env().expect("load config");
    let sr_fallback = cfg.config.get("sr").map(|v| v.as_u64().unwrap_or(16_000) as usize).unwrap_or(16_000);
    let gap_ms      = cfg.config.get("gap_ms").map(|v| v.as_i64().unwrap_or(200)).unwrap_or(200);
    let min_utter_ms= cfg.config.get("min_utter_ms").map(|v| v.as_i64().unwrap_or(250)).unwrap_or(250);
    let max_utter_ms= cfg.config.get("max_utter_ms").and_then(|v| v.as_i64()).unwrap_or(5000);
    let vad_rms_th  = cfg.config.get("vad_rms_th").map(|v| v.as_f64().unwrap_or(0.010) as f32).unwrap_or(0.010);

    // --- Zenoh
    let zenoh = ZenohInterface::from_default_env("zenoh").expect("missing zenoh config");
    let session = zenoh.get_session().await.expect("zenoh session");
    let sub = zenoh.get_subscriber(&session, "incoming_message").await.expect("subscriber");
    let pub_text = zenoh.get_publisher(&session, "transcript_text").await.expect("publisher");
    let enc_in = ProtobufEncoder::<FramePcmS16le>::new();

    // --- worker channel (don’t block subscriber) — pass (samples, input_sr)
    let (tx, mut rx) = mpsc::channel::<(Vec<i16>, usize)>(8);

    // Whisper worker
    let mut worker_state = state;
    let worker_pub_text = pub_text;
    let worker_params = params.clone();
    let (pub_tx, mut pub_rx) = mpsc::channel::<String>(100);
    tokio::spawn(async move {
        while let Some(s) = pub_rx.recv().await {
            if s.is_empty() || s == "[BLANK_AUDIO]" { continue; }
            println!("Detected: {}", &s);
            if let Err(e) = worker_pub_text.put(s.into_bytes()).await {
                eprintln!("zenoh publish error: {e:?}");
            }
        }
    });

    tokio::task::spawn_blocking(move || -> Result<()> {
        while let Some((mono_i16, in_sr)) = rx.blocking_recv() {
            if mono_i16.is_empty() { continue; }

            // Convert to f32 [-1,1]
            let mut audio_f32 = vec![0.0f32; mono_i16.len()];
            if let Err(e) = convert_integer_to_float_audio(&mono_i16, &mut audio_f32) {
                eprintln!("convert_integer_to_float_audio error: {e:?}");
                continue;
            }

            // Resample to 16 kHz for Whisper
            let audio_f32 = resample_linear_to_16k(&audio_f32, in_sr);

            // Inference
            worker_state.full(worker_params.clone(), &audio_f32).context("whisper full")?;

            // Collect segments
            let n = worker_state.full_n_segments();
            let mut out = String::new();
            for i in 0..n {
                if let Some(s) = worker_state.get_segment(i) {
                    out.push_str(s.to_str().unwrap());
                }
            }
            let out = out.trim();
            let _ = pub_tx.try_send(out.to_string());
        }
        Ok(())
    });

    // --- low-latency gap/VAD aggregator
    let mut buf: Vec<i16> = Vec::with_capacity(sr_fallback * 3);
    let mut utter_start_pts: Option<i64> = None;
    let mut prev_pts: Option<i64> = None;
    let mut current_in_sr: usize = sr_fallback;

    let mut handle_frame = |f: FramePcmS16le| -> Result<()> {
        // SR from time_base (if 1 tick == 1 sample: sr ≈ den/num), else fallback
        current_in_sr = infer_sample_rate(f.time_base.as_ref(), sr_fallback);

        // PTS -> ms
        let (tb_num, tb_den) = f.time_base.as_ref().map(|t| (t.num as i64, t.den as i64)).unwrap_or((1, 1000));
        let to_ms = |pts: i64| -> i64 { (pts * 1000 * tb_num) / tb_den };
        let pts_ms = to_ms(f.pts);

        // bytes -> i16 (interleaved)
        let mut i16s: Vec<i16> = f.data.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();

        // downmix to mono if needed
        if f.channels > 1 {
            let ch = f.channels as usize;
            let frames = i16s.len() / ch;
            let mut mono = Vec::with_capacity(frames);
            for i in 0..frames {
                let mut acc = 0i32;
                for c in 0..ch { acc += i16s[i*ch + c] as i32; }
                mono.push((acc / ch as i32) as i16);
            }
            i16s = mono;
        }

        // latency-minded flush logic
        let mut flush = false;
        if let Some(prev) = prev_pts {
            if pts_ms - prev >= gap_ms { flush = true; } // silence gap
        }
        if let Some(start) = utter_start_pts {
            if pts_ms - start >= max_utter_ms { flush = true; } // hard cap
        }

        // append samples BEFORE optional VAD check
        buf.extend_from_slice(&i16s);
        let dur_ms = ((buf.len() as f32) * 1000.0 / current_in_sr as f32) as i64;
        let quiet = rms_i16(&buf) < vad_rms_th;

        // VAD-assisted flush: flush if longish chunk or quiet after ~0.8s
        if dur_ms >= 2000 || (dur_ms >= 800 && quiet) {
            flush = true;
        }

        if flush {
            if let (Some(start), Some(prev)) = (utter_start_pts, prev_pts) {
                let span_ms = prev - start;
                if span_ms >= min_utter_ms && !buf.is_empty() {
                    let _ = tx.try_send((std::mem::take(&mut buf), current_in_sr));
                } else {
                    buf.clear();
                }
            }
            utter_start_pts = Some(pts_ms);
        }

        if utter_start_pts.is_none() { utter_start_pts = Some(pts_ms); }
        prev_pts = Some(pts_ms);
        Ok(())
    };

    // drain subscriber
    match sub {
        ConfiguredSubscriber::Fifo(s) => {
            while let Ok(sample) = s.recv_async().await {
                if let Ok(msg) = enc_in.decode(&sample.payload().to_bytes()) {
                    if let Err(e) = handle_frame(msg) { log::error!("{e:?}"); }
                }
            }
        }
        ConfiguredSubscriber::Ring(s) => {
            while let Ok(sample) = s.recv_async().await {
                if let Ok(msg) = enc_in.decode(&sample.payload().to_bytes()) {
                    if let Err(e) = handle_frame(msg) { log::error!("{e:?}"); }
                }
            }
        }
    }

    Ok(())
}

fn infer_sample_rate(tb: Option<&frame_pcm_s16le::Fraction>, fallback: usize) -> usize {
    if let Some(t) = tb {
        let num = t.num.max(1);
        let den = t.den.max(1);
        let sr = (den as f64 / num as f64).round() as usize; // seconds per tick = num/den
        if (8_000..=192_000).contains(&sr) { return sr; }
    }
    fallback
}

/// Linear resampler to 16 kHz.
fn resample_linear_to_16k(input: &[f32], in_sr: usize) -> Vec<f32> {
    if in_sr == 16_000 { return input.to_vec(); }
    if in_sr == 0 || input.is_empty() { return Vec::new(); }
    let ratio = 16_000.0 / in_sr as f32;
    let out_len = ((input.len() as f32) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let t = i as f32 / ratio;
        let j = t.floor() as usize;
        let a = t - j as f32;
        let s0 = *input.get(j).unwrap_or(&0.0);
        let s1 = *input.get(j + 1).unwrap_or(&s0);
        out.push(s0 + a * (s1 - s0));
    }
    out
}

/// Simple RMS on i16 buffer scaled to [-1,1]
fn rms_i16(a: &[i16]) -> f32 {
    if a.is_empty() { return 0.0; }
    let sum: f64 = a.iter().map(|&x| {
        let v = x as f64 / 32768.0;
        v * v
    }).sum();
    (sum / (a.len() as f64)).sqrt() as f32
}
