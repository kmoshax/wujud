//! `RoomEngine` against `research/live.py::RoomEngine` on the empty -> walk -> still excerpts.

mod common;

use common::*;
use serde_json::Value;

/// Python rounds probabilities to 3 and distances/activity to 2 decimals; a last-bit difference can flip one digit.
fn numbers_agree(actual: &Value, expected: &Value, path: &str) -> bool {
    agree_within(actual, expected, path, 1.0)
}

fn identical(actual: &Value, expected: &Value) -> bool {
    agree_within(actual, expected, "", 0.0)
}

fn agree_within(actual: &Value, expected: &Value, path: &str, rounding_steps: f64) -> bool {
    match (actual, expected) {
        (Value::Number(a), Value::Number(e)) => {
            let (a, e) = (a.as_f64().unwrap(), e.as_f64().unwrap());
            let last_digit = if path.contains("probs") {
                1.001e-3
            } else {
                1.001e-2
            };
            (a - e).abs() <= rounding_steps * last_digit
        }
        (Value::Array(a), Value::Array(e)) => {
            a.len() == e.len()
                && a.iter()
                    .zip(e)
                    .all(|(a, e)| agree_within(a, e, path, rounding_steps))
        }
        (Value::Object(a), Value::Object(e)) => {
            a.len() == e.len()
                && a.iter().all(|(k, v)| {
                    e.get(k).is_some_and(|ev| {
                        agree_within(v, ev, &format!("{path}.{k}"), rounding_steps)
                    })
                })
        }
        (a, e) => a == e,
    }
}

#[test]
fn room_engine_matches_python_every_second() {
    let golden = read_json("room_engine_ticks");
    let expected = golden["ticks"].as_array().unwrap();
    let recordings: Vec<Recording> = ["c3_empty", "c3_motion", "c3_still"].map(recording).into();
    let segments: Vec<(&str, &Recording)> = ["c3_empty", "c3_motion", "c3_still"]
        .into_iter()
        .zip(&recordings)
        .collect();
    let mut engine = replay_engine();
    let mut ticks = Vec::new();
    replay_room(&mut engine, &segments, |segment, now, state| {
        let mut json = serde_json::to_value(&state).unwrap();
        let history = json["history"].as_array().unwrap();
        json["history"] = Value::Array(history[history.len() - 1..].to_vec());
        ticks.push((segment.to_string(), now, json));
    });
    assert_eq!(ticks.len(), expected.len());
    let mut exact_states = 0;
    for ((segment, now, actual), expected) in ticks.iter().zip(expected) {
        assert_eq!(segment, expected["segment"].as_str().unwrap());
        assert_eq!(*now, expected["now"].as_f64().unwrap());
        let room_expected = &expected["state"]["room"];
        assert_eq!(
            actual["room"]["state3"], room_expected["state3"],
            "t = {now}"
        );
        assert_eq!(actual["room"]["state"], room_expected["state"], "t = {now}");
        assert!(
            numbers_agree(actual, &expected["state"], ""),
            "t = {now}:\n{actual}\n{}",
            expected["state"]
        );
        exact_states += identical(actual, &expected["state"]) as usize;
    }
    eprintln!(
        "room engine: {exact_states} of {} published states identical to Python",
        ticks.len()
    );
    assert!(
        exact_states * 100 >= ticks.len() * 95,
        "{exact_states} of {} ticks identical",
        ticks.len()
    );
}

struct LinkPackets {
    times: Vec<f64>,
    iq: ndarray::Array2<i8>,
    rssi: Vec<f64>,
}

fn three_link_packets() -> Vec<LinkPackets> {
    (0..3)
        .map(|k| LinkPackets {
            times: read_1d::<f64>(&format!("room3_link{k}_t")).to_vec(),
            iq: read_2d::<i8>(&format!("room3_link{k}_iq")),
            rssi: read_1d::<f64>(&format!("room3_link{k}_rssi")).to_vec(),
        })
        .collect()
}

fn three_link_engine(golden: &Value) -> wujud_model::room::RoomEngine {
    use wujud_model::{gbdt::Gbdt, room, scene::Scene};
    let nodes: Vec<[f64; 3]> = golden["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| [0, 1, 2].map(|c| n[c].as_f64().unwrap()))
        .collect();
    let links: Vec<(usize, usize)> = golden["links"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| {
            (
                l[0].as_u64().unwrap() as usize,
                l[1].as_u64().unwrap() as usize,
            )
        })
        .collect();
    let roles = ["router", "esp", "esp", "esp"].map(String::from).to_vec();
    let size = |key: &str| golden[key].as_f64().unwrap();
    let scene = Scene::room(size("W"), size("D"), size("H"), &nodes, vec![], roles);
    let model = Gbdt::load(&model_path(room::activity_model_file(links.len()))).unwrap();
    room::RoomEngine::new(scene, links, "REPLAY", model, nodes)
}

/// A simulated 3-board room (research/live.py demo-room, short): mean/max/min over links, the latch over the
/// concatenated profiles, per-link activity.
#[test]
fn room_engine_with_three_links_matches_python_every_second() {
    let golden = read_json("room_engine_three_links");
    let expected = golden["ticks"].as_array().unwrap();
    let packets = three_link_packets();
    let mut engine = three_link_engine(&golden);
    let mut next_packet = [0; 3];
    let mut exact_states = 0;
    for tick in expected {
        let now = tick["now"].as_f64().unwrap();
        for (link, link_packets) in packets.iter().enumerate() {
            while next_packet[link] < link_packets.times.len()
                && link_packets.times[next_packet[link]] < now
            {
                let p = next_packet[link];
                let iq = link_packets.iq.row(p);
                engine.add(
                    link,
                    link_packets.times[p],
                    iq.as_slice().unwrap(),
                    link_packets.rssi[p],
                );
                next_packet[link] += 1;
            }
        }
        let mut actual = serde_json::to_value(engine.tick(now).unwrap()).unwrap();
        let history = actual["history"].as_array().unwrap();
        actual["history"] = Value::Array(history[history.len() - 1..].to_vec());
        let state = &tick["state"];
        assert_eq!(
            actual["room"]["state3"], state["room"]["state3"],
            "t = {now}"
        );
        assert_eq!(actual["room"]["state"], state["room"]["state"], "t = {now}");
        assert_eq!(actual["alive"], state["alive"], "t = {now}");
        assert!(
            numbers_agree(&actual, state, ""),
            "t = {now}:\n{actual}\n{state}"
        );
        exact_states += identical(&actual, state) as usize;
    }
    eprintln!(
        "three-link room engine: {exact_states} of {} published states identical to Python",
        expected.len()
    );
    assert!(exact_states * 100 >= expected.len() * 95);
}
