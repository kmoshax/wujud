//! Evaluation helpers: ROC AUC and the room benchmark metrics of `research/bench/phase_exp.py::metrics`.

use serde::Serialize;

use crate::gbdt::Gbdt;
use crate::Result;

/// Area under the ROC curve (sklearn `roc_auc_score`): tied scores count half. NaN without both classes.
pub fn roc_auc(positive: &[bool], scores: &[f64]) -> f64 {
    let positives = positive.iter().filter(|&&p| p).count();
    let negatives = positive.len() - positives;
    if positives == 0 || negatives == 0 {
        return f64::NAN;
    }
    let ranks = average_ranks(scores);
    let positive_rank_sum: f64 = ranks
        .iter()
        .zip(positive)
        .filter(|(_, &p)| p)
        .map(|(r, _)| r)
        .sum();
    let pairs_won = positive_rank_sum - (positives * (positives + 1)) as f64 / 2.0;
    pairs_won / (positives as f64 * negatives as f64)
}

/// 1-based ranks, ties sharing their average rank.
fn average_ranks(scores: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    let mut ranks = vec![0.0; scores.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && scores[order[end]] == scores[order[start]] {
            end += 1;
        }
        let average = (start + 1 + end) as f64 / 2.0;
        for &i in &order[start..end] {
            ranks[i] = average;
        }
        start = end;
    }
    ranks
}

/// Window feature rows of one recording and its label (0 empty, 1 still person, 2 motion).
#[derive(Debug, Clone)]
pub struct LabelledWindows {
    pub rows: Vec<Vec<f64>>,
    pub label: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RoomMetrics {
    pub motion_recall: f64,
    pub motion_false_alarm: f64,
    pub still_vs_empty_auc: f64,
    pub still_vs_empty_auc_recording: f64,
    pub n_windows: usize,
    pub n_recordings: usize,
}

const MOTION: i64 = 2;
/// Recording-level score: the first ~minute (9 windows of 20 s every 5 s).
const RECORDING_WINDOWS: usize = 9;

fn still_share(probabilities: &[f64]) -> f64 {
    probabilities[1] / (probabilities[0] + probabilities[1] + 1e-9)
}

fn fraction(hits: usize, total: usize) -> f64 {
    hits as f64 / total as f64
}

/// Motion recall / false alarms per window and still-vs-empty AUC per window and per recording,
/// for a 3-class (empty, still, motion) model reading `columns` of each row.
pub fn room_metrics(
    model: &Gbdt,
    recordings: &[LabelledWindows],
    columns: &[usize],
) -> Result<RoomMetrics> {
    let select = |row: &Vec<f64>| columns.iter().map(|&c| row[c]).collect::<Vec<f64>>();
    let (mut motion_hits, mut motion_total, mut false_alarms, mut quiet_total) = (0, 0, 0, 0);
    let (mut window_is_still, mut window_score) = (Vec::new(), Vec::new());
    let (mut recording_is_still, mut recording_score) = (Vec::new(), Vec::new());
    for recording in recordings {
        let probabilities: Vec<Vec<f64>> = recording
            .rows
            .iter()
            .map(|row| model.predict_proba(&select(row)))
            .collect::<Result<_>>()?;
        for p in &probabilities {
            let says_motion = model.classes[crate::numeric::argmax(p)] == MOTION;
            if recording.label == MOTION {
                motion_total += 1;
                motion_hits += says_motion as usize;
            } else {
                quiet_total += 1;
                false_alarms += says_motion as usize;
                window_is_still.push(recording.label == 1);
                window_score.push(still_share(p));
            }
        }
        if recording.label != MOTION {
            let first: Vec<f64> = probabilities
                .iter()
                .take(RECORDING_WINDOWS)
                .map(|p| still_share(p))
                .collect();
            recording_is_still.push(recording.label == 1);
            recording_score.push(first.iter().sum::<f64>() / first.len() as f64);
        }
    }
    Ok(RoomMetrics {
        motion_recall: fraction(motion_hits, motion_total),
        motion_false_alarm: fraction(false_alarms, quiet_total),
        still_vs_empty_auc: roc_auc(&window_is_still, &window_score),
        still_vs_empty_auc_recording: roc_auc(&recording_is_still, &recording_score),
        n_windows: motion_total + quiet_total,
        n_recordings: recordings.len(),
    })
}
