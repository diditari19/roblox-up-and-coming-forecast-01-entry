// Roblox Up-and-Coming forecaster (entry) — analogue conditional occupancy.
// Node 22, standard library only. Deterministic (no randomness at all).
// Outside the grader's container:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Algorithm hyper-parameters (per DESIGN.md).
const ANALOGUE_RANK_TOL = 4;   // |rank - r| <= 4
const ANALOGUE_VEL_TOL = 0.5;  // |velocity - v| <= 0.5
const SHARPEN = 1.15;          // per-place exponent before renormalization
const MIN_P = 0.002;           // emit rows only above this (floored to 4 dp)

interface ChartRow {
  rank: number;
  universe_id: number;
  playing: number;
  root_place_id: number;
  name: string;
}

interface Snapshot {
  at: string;
  up_and_coming: ChartRow[];
}

interface Obs {
  timeSec: number;
  rank: number;
  playing: number;
}

// Horizons are uniform every 6h over 0..72h, scored hourly in the final window.
const HORIZONS = [...Array(13)].map((_, i) => 6 * i); // 0,6,...,72

const taskJson = readFileSync(join(INPUT_DIR, 'task.json'), 'utf8');
const task = JSON.parse(taskJson) as { places: number; output: string; window?: { from: string; until: string } };
const places = task.places || 10;

const rawLines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((l) => l.trim() !== '');

const snapshots: Snapshot[] = rawLines.map((l) => JSON.parse(l) as Snapshot);

// Convert each snapshot to per-game observations keyed by universe_id.
const series = new Map<number, Obs[]>();
for (const snap of snapshots) {
  const t = Date.parse(snap.at) / 1000;
  for (const row of snap.up_and_coming) {
    let arr = series.get(row.universe_id);
    if (!arr) {
      arr = [];
      series.set(row.universe_id, arr);
    }
    arr.push({ timeSec: t, rank: row.rank, playing: row.playing });
  }
}

// Sort each series by time (stable) and dedupe: keep the last-seen value per read.
for (const arr of series.values()) {
  arr.sort((a, b) => (a.timeSec === b.timeSec ? 0 : a.timeSec - b.timeSec));
  const out: Obs[] = [];
  for (const o of arr) {
    if (out.length && out[out.length - 1].timeSec === o.timeSec) {
      out[out.length - 1] = o;
    } else {
      out.push(o);
    }
  }
}

// Velocity: rank change per hour over the last interval, blended 50/50 with the
// previous interval's (relative to the observation's own two predecessors).
// 0 if fewer than 2 observations.
function velocityAt(obs: Obs[], i: number): number {
  const n = obs.length;
  if (n < 2) return 0;
  const last = Math.min(i, n - 1);
  const second = Math.max(0, last - 1);
  const third = Math.max(0, last - 2);
  // rank change is signed so upward moves (worse rank) are positive.
  const v1 = obs[last].rank - obs[second].rank;
  const dt1 = obs[last].timeSec - obs[second].timeSec;
  const v1h = dt1 > 0 ? v1 / dt1 : 0;
  if (third === second) return v1h;
  const v2 = obs[second].rank - obs[third].rank;
  const dt2 = obs[second].timeSec - obs[third].timeSec;
  const v2h = dt2 > 0 ? v2 / dt2 : 0;
  return (v1h + v2h) / 2;
}

// For every (game, read) record (rank, velocity) and the game's rank at each
// horizon H: nearest read within 12h of now + H, else null.
type FutureRecord = {
  rank: number;
  velocity: number;
  horizons: (number | null)[];
};
const records = new Map<number, FutureRecord[]>();
for (const [id, obs] of series) {
  const recs: FutureRecord[] = [];
  for (let i = 0; i < obs.length; i++) {
    const now = obs[i].timeSec;
    const hor = HORIZONS.map((H) => {
      const target = now + H * 3600;
      const cand = obs
        .slice(i + 1)
        .reduce(
          (best, o, idx) => {
            const d = Math.abs(o.timeSec - target);
            if (best === null) return { o, d, idx };
            return d < best.d ? { o, d, idx } : best;
          },
          null as { o: Obs; d: number; idx: number } | null
        );
      return cand !== null && cand.d <= 12 * 3600 ? cand.o.rank : null;
    });
    recs.push({ rank: obs[i].rank, velocity: velocityAt(obs, i), horizons: hor });
  }
  records.set(id, recs);
}

// The last chart as of "now" (the final read).
const lastSnap = snapshots[snapshots.length - 1];
const nowSeries = series.get(lastSnap.up_and_coming[0].universe_id);
// Build the last-chart series properly from the last snapshot.
const nowObsByRank = new Map<number, Obs>();
const lastChart: { id: number; rank: number; name: string }[] = lastSnap.up_and_coming
  .map((row) => ({ id: row.universe_id, rank: row.rank, name: row.name }))
  .sort((a, b) => a.rank - b.rank);

// Step 2 (velocity) is inside recs; build analogue records on demand.
// Step 3 already materialized future records for every game.

// Step 4: per last-chart game, find analogues among ALL past observations with
// |rank - r| <= 4 and |velocity - v| <= 0.5; average their future-rank
// distributions over 13 horizons -> occupancy per place 1..10.
const occ = new Map<number, number[]>(); // universe_id -> array[1..10] occupancy
const allIds = Array.from(records.keys());
for (const game of lastChart) {
  const id = game.id;
  const gObs = series.get(id);
  if (!gObs) continue;
  const n = gObs.length;
  const r = gObs[n - 1].rank;
  const v = velocityAt(gObs, n - 1);
  const sum = new Array(11).fill(0); // index 1..10
  let count = 0;
  const recs = records.get(id);
  // analogues = every (game, read) in history including this game's past self
  for (const otherId of allIds) {
    const otherRecs = records.get(otherId)!;
    for (const rec of otherRecs) {
      if (Math.abs(rec.rank - r) <= ANALOGUE_RANK_TOL && Math.abs(rec.velocity - v) <= ANALOGUE_VEL_TOL) {
        for (const hr of rec.horizons) {
          if (hr !== null && hr >= 1 && hr <= 10) {
            sum[hr] += 1 / 13;
          }
        }
        count += 1;
      }
    }
  }
  if (count === 0) {
    occ.set(id, new Array(11).fill(0));
    continue;
  }
  const o = new Array(11).fill(0);
  for (let p = 1; p <= 10; p++) o[p] = sum[p] / count;
  occ.set(id, o);
}

// Step 5: new-entrant dilution. For each horizon H, fraction of top-10 slots held
// by games absent at "now". Average over horizons.
// Use the analogue records' horizon-ranks to estimate how often the top 10 at now
// is occupied by a game that is NOT in the last chart. Count across all analogue
// (game, horizon) pairs whose horizon-rank is in 1..10.
let newInTotal = 0;
let newInCount = 0;
for (const otherId of allIds) {
  const otherRecs = records.get(otherId)!;
  for (const rec of otherRecs) {
    for (const hr of rec.horizons) {
      if (hr !== null && hr >= 1 && hr <= 10) {
        newInCount += 1;
        if (!lastChart.some((g) => g.id === otherId)) {
          newInTotal += 1;
        }
      }
    }
  }
}
const avgNew = newInCount > 0 ? newInTotal / newInCount : 0;

// Step 6: sharpening + per-place renormalization, then new-entrant dilution.
// Order per DESIGN.md: dilute first, then sharpen to alpha and renormalize
// that place's total back to its pre-sharpening sum.
const finalOcc = new Map<number, number[]>();
for (const [id, o] of occ) {
  const dilute = new Array(11).fill(0);
  for (let p = 1; p <= 10; p++) dilute[p] = o[p] * (1 - avgNew);
  const preSum = dilute.slice(1).reduce((s, x) => s + x, 0);
  if (preSum <= 0) {
    finalOcc.set(id, new Array(11).fill(0));
    continue;
  }
  const sharpened = dilute.map((x) => Math.max(x, 0) ** SHARPEN);
  const sharpSum = sharpened.slice(1).reduce((s, x) => s + x, 0);
  const renorm = new Array(11).fill(0);
  for (let p = 1; p <= 10; p++) renorm[p] = preSum * (sharpened[p] / sharpSum);
  finalOcc.set(id, renorm);
}

// Step 7: emit rows with p > MIN_P, floored to 4 decimals so per-place sums
// never exceed 1. Group by place 1..places.
const byPlace = new Array(places + 1).fill(null).map(() => new Map<number, number>());
for (const [id, o] of finalOcc) {
  for (let p = 1; p <= places; p++) {
    const pval = Math.floor(o[p] * 10000) / 10000;
    if (pval > MIN_P) {
      byPlace[p].set(id, pval);
    }
  }
}


// top10: 10 games with highest total top-10 occupancy, ordered by that total.
const totals = new Array(lastChart.length);
for (let i = 0; i < lastChart.length; i++) {
  const id = lastChart[i].id;
  totals[i] = { id, total: finalOcc.get(id)!.slice(1, places + 1).reduce((s, x) => s + x, 0), name: lastChart[i].name };
}
totals.sort((a, b) => b.total - a.total);
const top10Ids = totals.slice(0, 10).map((t) => t.id);

const probabilities: Array<{ universe_id: number; place: number; p: number }> = [];
for (let p = 1; p <= places; p++) {
  for (const [id, val] of byPlace[p]) {
    probabilities.push({ universe_id: id, place: p, p: val });
  }
}
// Sort for determinism: by place, then by total-occupancy descending, then id.
const totalOcc = new Map<number, number>();
for (const t of totals) totalOcc.set(t.id, t.total);
probabilities.sort((a, b) => {
  if (a.place !== b.place) return a.place - b.place;
  const ta = totalOcc.get(a.universe_id)!;
  const tb = totalOcc.get(b.universe_id)!;
  if (tb !== ta) return tb - ta;
  return a.universe_id - b.universe_id;
});

const out = {
  probabilities,
  top10: top10Ids,
};

writeFileSync(join(OUTPUT_DIR, task.output || 'forecast.json'), JSON.stringify(out, null, 2) + '\n');
