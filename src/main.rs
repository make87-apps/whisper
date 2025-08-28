use anyhow::{Context, Result};
use make87::config::{load_config_from_default_env};
use make87::encodings::{Encoder, ProtobufEncoder};
use make87::interfaces::zenoh::{ConfiguredSubscriber, ZenohInterface};
use make87_messages::audio::FramePcmS16le;
use tokio::sync::mpsc;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, convert_integer_to_float_audio};

// const SR: usize = 16_000;
// const GAP_MS: i64 = 600;         // flush if gap between frames ≥ 600 ms
// const MAX_UTTER_MS: i64 = 10_000;// hard cutoff per utterance
// const MIN_UTTER_MS: i64 = 300;   // ignore very short blips

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    // --- Whisper init
    let model = std::env::var("WHISPER_MODEL").unwrap_or("/app/models/ggml-tiny.en.bin".into());
    let ctx = WhisperContext::new_with_params(&model, WhisperContextParameters::default())
        .with_context(|| format!("loading model {model}"))?;
    let mut state = ctx.create_state().context("create whisper state")?;

    let mut params = FullParams::new(SamplingStrategy::BeamSearch { beam_size: 5, patience: -1.0 });
    params.set_print_realtime(false);
    params.set_print_progress(false);
    params.set_detect_language(true);

    let config = load_config_from_default_env().expect("load config");
    let sr = config.config.get("sr").map(|v| v.as_u64().unwrap_or(16_000) as usize).unwrap_or(16_000);
    let gap_ms = config.config.get("gap_ms").map(|v| v.as_i64().unwrap_or(600)).unwrap_or(600);
    let min_utter_ms = config.config.get("min_utter_ms").map(|v| v.as_i64().unwrap_or(300)).unwrap_or(300);

    // --- Zenoh
    let zenoh = ZenohInterface::from_default_env("zenoh").expect("missing zenoh config");
    let session = zenoh.get_session().await.expect("zenoh session");
    let sub = zenoh.get_subscriber(&session, "incoming_message").await.expect("subscriber");
    let pub_text = zenoh.get_publisher(&session, "transcript_text").await.expect("publisher");
    let enc_in = ProtobufEncoder::<FramePcmS16le>::new();
    // --- worker channel (don’t block subscriber)
    let (tx, mut rx) = mpsc::channel::<Vec<i16>>(8);

    // Whisper worker
    tokio::task::spawn_blocking(move || -> Result<()> {
        while let Some(mono_i16) = rx.blocking_recv() {
            if mono_i16.is_empty() { continue; }
            let mut audio_f32= vec![0.0f32; mono_i16.len()];
            let res = convert_integer_to_float_audio(&mono_i16, audio_f32.as_mut_slice());
            if let Err(e) = res {
                log::error!("convert_integer_to_float_audio: {e:?}");
                continue;
            }

            state.full(params.clone(), &audio_f32).context("whisper full")?;

            // collect segments into one UTF-8 string
            let mut out = String::new();
            for seg in state.as_iter() {
                out.push_str(&seg.to_string());
            }
            let out = out.trim();
            if !out.is_empty() {
                // publish directly as UTF-8 bytes
                let _ = pub_text.put(out.as_bytes().to_vec());
            }
        }
        Ok(())
    });

    // --- simple gap-based aggregator
    let mut buf: Vec<i16> = Vec::with_capacity(sr * 12);
    let mut utter_start_pts: Option<i64> = None;
    let mut prev_pts: Option<i64> = None;

    let mut handle_frame = |f: FramePcmS16le| -> Result<()> {
        let tb = f.time_base.as_ref().map(|t| (t.num, t.den)).unwrap_or((1, 1000)); // default ms
        let to_ms = |pts: i64| -> i64 { (pts * 1000 * tb.0 as i64) / tb.1 as i64 }; // (pts * num/den) in seconds -> ms
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

        // new utterance?
        let mut flush = false;
        if let Some(prev) = prev_pts {
            if pts_ms - prev >= gap_ms {
                flush = true;
            }
        }
        if let Some(start) = utter_start_pts {
            if pts_ms - start >= min_utter_ms {
                flush = true;
            }
        }

        // flush if needed
        if flush {
            if let (Some(start), Some(prev)) = (utter_start_pts, prev_pts) {
                let dur_ms = prev - start;
                if dur_ms >= min_utter_ms && !buf.is_empty() {
                    let _ = tx.try_send(std::mem::take(&mut buf)); // send and clear
                } else {
                    buf.clear();
                }
            }
            utter_start_pts = Some(pts_ms);
        }

        // start if empty
        if utter_start_pts.is_none() { utter_start_pts = Some(pts_ms); }

        // append samples
        buf.extend_from_slice(&i16s);
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
