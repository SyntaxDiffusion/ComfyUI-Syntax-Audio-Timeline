import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "SyntaxTimeline";
const CANVAS_H = 230;
const PAD = { l: 10, r: 10, t: 4, b: 22 };
const HIT_LANE = 20;
const SHOT_STRIP = 26;   // action: the shots sit in a strip along the bottom, under the drawing
const ACTION_H = 290;
const GRAB_PX = 10;
const MIN_SHOT = 0.25;
const SHOT_SECONDS = 5.5;
const COLORS = {
    bg: "#15171c", lane: "#1d2027", wave: "#5b6475", text: "#9aa1b0", beat: "rgba(255,255,255,0.06)", bar: "rgba(255,255,255,0.16)",
    cut: "#e8ecf3", range: "#ffd633", hover: "#ffd633", kick: "#ff4050", shots: ["rgba(77,158,255,0.22)", "rgba(255,140,77,0.22)"],
};
const HIT_SOUNDS = { thump: "#ff4050", crash: "#ffd04d", boom: "#ff8a3d" };
const CURVE_SOUNDS = { flow: "#4dc3ff", motor: "#b07cff", pulse: "#5cf2a0", mumble: "#ff7ad1" };
const CURVE_USES = { flow: "wind, water, light", motor: "an engine revving", pulse: "magic, sci-fi", mumble: "someone talking" };
const HELP = [
    "Shots: drag an edge to move it. Double-click a shot to split it. Right-click an edge to merge the two shots.",
    "Range: drag the yellow handles to set where the video starts and ends.",
    "music: the KICKS lane shows the kick drum hits the last run found. Only they guide the shots. Cuts snap to bars: hold Alt for beats, Shift to move freely.",
    "action, HITS lane: double-click to add a hit with the hit_sound sound, double-click a hit to change its sound, drag to move it, right-click to delete it.",
    "action, drawing: pick a sound above the timeline (flow: wind, water, light; motor: an engine revving; pulse: magic, sci-fi; mumble: someone talking), then drag in the big area to draw it. Higher is louder or faster. Draw over a part to redraw it, right-drag to erase, x on a sound clears it.",
    "action, shots: they sit in the strip along the bottom. Double-click the strip to split a shot.",
    "No track: action on a blank timeline of duration seconds; what you draw is the sound.",
    "View: mouse wheel zooms, drag an empty part of the timeline to scroll, drag past an edge to scroll while moving something.",
    "One shot: a single shot over the whole range. By images: one shot per image again. More shots than images reuse the images in order.",
    "Analyse: runs just this node (and what feeds it) to find the beat, the kicks, the waveform and the exact cuts.",
];
const HINTS = {
    cut: "drag to move this cut · right-click to merge the two shots",
    start: "drag to set where the video starts",
    end: "drag to set where the video ends",
    kicks: "the kick drum hits found by the last run: the only sound that guides the shots",
    lane: "double-click to add a hit here",
    draw: "drag to draw · right-drag to erase · wheel to zoom",
    point: "drag to move this point, higher is louder or faster · right-click to delete it",

    shot: "double-click to split here · drag to scroll · wheel to zoom",
    pan: "scrolling the view",
    none: "hover the timeline for what you can do · ? for all controls",
};

// seconds as "12.34 s", or "2:45.33" past a minute
const fmt = (t) => (t < 60 ? `${t.toFixed(2)} s` : `${Math.floor(t / 60)}:${(t % 60).toFixed(2).padStart(5, "0")}`);

const peaksCache = new Map();

// peaks of an input file, one max-abs value per bucket, decoded in the browser
async function loadPeaks(name) {
    if (peaksCache.has(name)) return peaksCache.get(name);
    const promise = (async () => {
        const i = name.lastIndexOf("/");
        const query = `filename=${encodeURIComponent(name.slice(i + 1))}&subfolder=${encodeURIComponent(i >= 0 ? name.slice(0, i) : "")}&type=input`;
        const buf = await (await api.fetchApi(`/view?${query}`)).arrayBuffer();
        const audio = await new OfflineAudioContext(1, 1, 44100).decodeAudioData(buf);
        const data = audio.getChannelData(0), n = 4000, step = Math.max(1, Math.floor(data.length / n));
        const peaks = new Float32Array(Math.ceil(data.length / step));
        for (let b = 0; b < peaks.length; b++) {
            let m = 0;
            for (let j = b * step, e = Math.min(data.length, j + step); j < e; j++) m = Math.max(m, Math.abs(data[j]));
            peaks[b] = m;
        }
        const top = Math.max(...peaks) || 1;
        return { peaks: peaks.map((p) => p / top), duration: audio.duration };
    })();
    peaksCache.set(name, promise);
    promise.catch(() => peaksCache.delete(name));
    return promise;
}

class TimelineEditor {
    constructor(node, container) {
        this.node = node;
        this.w = (name) => node.widgets.find((w) => w.name === name);
        this.data = null;          // last run: {duration, mode, cuts, hits, images, bpm, downbeat, key, source}
        this.wave = null;          // {peaks, duration, name}
        this.viewMode = "fit";     // fit: the shots, whole: the track, manual: zoomed or scrolled by hand
        this.manual = null;
        this.frozen = null;        // the view held still while something is dragged
        this.pan = null;
        this.help = false;
        this.drag = null;          // {kind, i}
        this.hover = { kind: "none" };
        this.hoverT = null;

        container.style.cssText = "display:flex;flex-direction:column;gap:4px;width:100%;font:11px sans-serif;color:#9aa1b0;";
        this.label = document.createElement("div");
        this.label.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
        const bar = document.createElement("div");
        bar.style.cssText = "display:flex;gap:4px;flex-wrap:wrap;";
        const button = (text, title, fn) => {
            const b = document.createElement("button");
            b.textContent = text;
            b.title = title;
            b.style.cssText = "background:#222;color:#ddd;border:1px solid #444;border-radius:4px;padding:1px 7px;cursor:pointer;";
            b.onclick = fn;
            bar.append(b);
            return b;
        };
        button("Analyse", "Run just this node to find the beat and the exact cuts", () => this.analyse());
        button("One shot", "One shot over the whole range", () => { const c = this.layout().cuts; this.setEdits({ cuts: [c[0], c[c.length - 1]] }); });
        button("By images", "One shot per image, cut automatically", () => this.setEdits({ cuts: null }));
        this.clearButton = button("Clear drawing", "Remove every hit and curve", () => this.setEdits({ hits: null, curves: null }));
        button("Fit", "Zoom to the shots", () => { this.viewMode = "fit"; this.draw(); });
        button("Whole track", "Show the whole track", () => { this.viewMode = "whole"; this.draw(); });
        this.helpButton = button("?", "How to edit the timeline", () => { this.help = !this.help; this.draw(); });

        this.canvas = document.createElement("canvas");
        this.canvas.style.cssText = `width:100%;height:${CANVAS_H}px;border-radius:6px;touch-action:none;`;
        // the frontend leaves the wheel to a focused element marked like this, so the wheel zooms the timeline, not the graph
        this.canvas.dataset.captureWheel = "true";
        this.canvas.tabIndex = -1;
        this.canvas.style.outline = "none";
        this.canvas.addEventListener("pointerenter", () => this.canvas.focus({ preventScroll: true }));
        this.hint = document.createElement("div");
        this.hint.style.cssText = "min-height:14px;color:#c9ced8;";
        this.chips = document.createElement("div");      // action: the sound you draw with, and x to clear one
        this.chips.style.cssText = "display:flex;gap:4px;flex-wrap:wrap;align-items:center;";
        this.brush = "motor";
        this.paint = null;                                // {erase, last, from, to} while drawing
        container.append(this.label, bar, this.chips, this.canvas, this.hint);
        this.canvas.addEventListener("pointerdown", (e) => this.onDown(e));
        this.canvas.addEventListener("pointermove", (e) => this.onMove(e));
        this.canvas.addEventListener("pointerup", (e) => this.onUp(e));
        // the node can redraw under a stroke and take the pointer away: finish the stroke then too
        this.canvas.addEventListener("lostpointercapture", () => { if (this.paint) { this.endStroke(); this.draw(); } });
        this.canvas.addEventListener("pointerleave", () => { this.hoverT = null; this.hover = { kind: "none" }; this.draw(); });
        this.canvas.addEventListener("dblclick", (e) => this.onDouble(e));
        this.canvas.addEventListener("contextmenu", (e) => this.onRight(e));
        this.canvas.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
        new ResizeObserver(() => this.draw()).observe(this.canvas);
        this.timer = setInterval(() => this.checkSource(), 1000);
    }

    // the Load Audio file wired into `track`, if any
    sourceName() {
        const slot = this.node.findInputSlot("track");
        const src = slot >= 0 ? this.node.getInputNode(slot) : null;
        return src?.type === "LoadAudio" ? src.widgets?.find((w) => w.name === "audio")?.value : null;
    }

    checkSource() {
        const name = this.sourceName();
        if (!name || this.wave?.name === name || this.loading === name) return;
        this.loading = name;
        loadPeaks(name).then((w) => { this.wave = { ...w, name }; this.draw(); }).catch(() => {}).finally(() => { this.loading = null; });
    }

    hasTrack() {
        const slot = this.node.findInputSlot("track");
        return slot >= 0 && this.node.inputs[slot].link != null;
    }

    imageCount() {
        return Math.max(1, this.node.inputs.filter((i) => i.name.startsWith("images.") && i.link != null).length);
    }

    // with no track the node runs the action timeline, whatever the mode
    get mode() {
        return this.hasTrack() ? this.w("mode").value : "action";
    }

    // what the last run's cuts depend on; when it changes, the editor shows a preview until the next run
    runKey() {
        return JSON.stringify([this.mode, this.w("start_at").value, this.w("duration").value, this.w("bpm").value, this.imageCount(), this.sourceName()]);
    }

    get stored() {
        try {
            return JSON.parse(this.w("edits").value || "{}");
        } catch {
            return {};
        }
    }

    // [[seconds, sound]]; a bare time is a hit_sound hit
    get hits() {
        return (this.stored.hits ?? []).map((h) => (typeof h === "number" ? [h, this.w("hit_sound").value] : h));
    }

    // [{sound, points: [[seconds, 0..1]]}]
    get curves() {
        return this.stored.curves ?? [];
    }

    // merge {cuts, hits, curves} into the stored edits; null removes a key
    setEdits(change) {
        const next = { ...this.stored, hits: this.hits, ...change };
        const r3 = (t) => +t.toFixed(3);
        if (next.cuts) next.cuts = next.cuts.map(r3).sort((x, y) => x - y);
        if (next.hits) next.hits = next.hits.map(([t, sound]) => [r3(t), sound]).sort((x, y) => x[0] - y[0]);
        if (next.curves) next.curves = next.curves.map((c) => ({ sound: c.sound, points: c.points.map(([t, v]) => [r3(t), r3(v)]).sort((x, y) => x[0] - y[0]) }));
        for (const k of Object.keys(next)) if (!next[k]?.length) delete next[k];
        const w = this.w("edits");
        w.value = Object.keys(next).length ? JSON.stringify(next) : "";
        w.callback?.(w.value);
        if (next.cuts && this.hasTrack()) {
            // keep start_at and duration in step with hand-made shots, without the callback that resets them
            this.w("start_at").value = next.cuts[0];
            this.w("duration").value = +(next.cuts[next.cuts.length - 1] - next.cuts[0]).toFixed(3);
        }
        this.node.setDirtyCanvas(true, true);
        this.draw();
    }

    // the lane above the shots: the kicks in music, the hits in action
    lanes() {
        return this.mode === "music" ? [{ kind: "kicks", top: PAD.t, h: HIT_LANE }]
            : this.mode === "action" ? [{ kind: "hits", top: PAD.t, h: HIT_LANE }] : [];
    }

    height() {
        return this.mode === "action" ? ACTION_H : CANVAS_H;
    }

    // action: where the curves are drawn, between the hits and the shot strip
    area() {
        const top = PAD.t + HIT_LANE;
        return { top, h: this.canvas.clientHeight - PAD.b - SHOT_STRIP - top };
    }

    // the curve the picked sound draws on
    get activeCurve() {
        return this.curves.findIndex((c) => c.sound === this.brush);
    }

    // draw (or erase) from the last pointer position to this one
    stroke(e) {
        const p = this.paint, r = this.canvas.getBoundingClientRect(), area = this.area();
        const t = Math.max(0, this.tOf(e)), v = this.vOf(e.clientY - r.top, area);
        const lo = Math.min(p.last, t), hi = Math.max(p.last, t), inside = (x) => x > lo + 1e-4 && x < hi - 1e-4;
        const curves = this.curves;
        if (p.erase) {
            if (hi - lo < 0.01) return;
            for (const c of curves) {
                const pts = c.points;
                if (pts.length < 2 || pts[0][0] > hi || pts.at(-1)[0] < lo) continue;
                // the level just outside the erased stretch, so the curve keeps its shape up to the eraser
                const level = (t) => {
                    const j = pts.findIndex(([x]) => x >= t);
                    if (j <= 0) return j === 0 && pts[0][0] === t ? pts[0][1] : 0;
                    const [[t0, v0], [t1, v1]] = [pts[j - 1], pts[j]];
                    return v0 + ((v1 - v0) * (t - t0)) / Math.max(t1 - t0, 1e-6);
                };
                const edges = [[lo - 0.01, level(lo - 0.01)], [lo, 0], [hi, 0], [hi + 0.01, level(hi + 0.01)]].filter(([, v], i) => i === 1 || i === 2 || v > 0);
                c.points = [...pts.filter(([x]) => x < lo - 0.01 || x > hi + 0.01), ...edges];
            }
        } else {
            let c = curves.find((x) => x.sound === this.brush);
            if (!c) curves.push((c = { sound: this.brush, points: [] }));
            c.points = [...c.points.filter(([x]) => !inside(x) && Math.abs(x - t) > 1e-4), [t, v]];
        }
        p.last = t;
        p.from = Math.min(p.from, t);
        p.to = Math.max(p.to, t);
        this.setEdits({ curves });
    }

    // a finished stroke: fewer points, and silence just outside it unless it joins what was there
    endStroke() {
        const p = this.paint, curves = this.curves, c = curves.find((x) => x.sound === this.brush);
        this.paint = null;
        // silent stretches need only their two ends
        const tidy = (pts) => pts.filter(([, v], i) => v > 0 || !(pts[i - 1]?.[1] === 0 && pts[i + 1]?.[1] === 0));
        if (p.erase || !c) {
            for (const x of curves) x.points = tidy(x.points);
            return this.setEdits({ curves });
        }
        const keep = (pts) => {
            // Douglas-Peucker: drop points within 1.5% of the line through their neighbours
            if (pts.length < 3) return pts;
            const [a, b] = [pts[0], pts.at(-1)];
            let worst = 0, k = 0;
            pts.forEach(([t, v], i) => {
                const d = Math.abs(v - (a[1] + ((b[1] - a[1]) * (t - a[0])) / Math.max(b[0] - a[0], 1e-6)));
                if (d > worst) { worst = d; k = i; }
            });
            return worst < 0.015 ? [a, b] : [...keep(pts.slice(0, k + 1)).slice(0, -1), ...keep(pts.slice(k))];
        };
        // stored times are rounded to the millisecond: compare with that slack
        const from = p.from - 0.002, to = p.to + 0.002;
        const before = c.points.filter(([t]) => t < from), after = c.points.filter(([t]) => t > to);
        const drawn = keep(c.points.filter(([t]) => t >= from && t <= to));
        if (!before.length || p.from - before.at(-1)[0] > 0.05) before.push([Math.max(0, p.from - 0.02), 0]);
        if (!after.length || after[0][0] - p.to > 0.05) after.unshift([p.to + 0.02, 0]);
        c.points = tidy([...before, ...drawn, ...after]);
        this.setEdits({ curves });
    }

    resize() {
        this.canvas.style.height = `${this.height()}px`;
        this.node.setSize([this.node.size[0], this.node.computeSize()[1]]);
    }

    // a curve's level (0..1) at a height in `lane`, and back
    yOf(v, lane) {
        return lane.top + 4 + (1 - v) * (lane.h - 10);
    }

    vOf(y, lane) {
        return Math.min(Math.max(1 - (y - lane.top - 4) / (lane.h - 10), 0), 1);
    }

    // the waveform: decoded from a Load Audio wired straight in, else sent by the last run
    get waveform() {
        if (this.wave) return this.wave;
        const d = this.data;
        return d?.peaks && this.hasTrack() ? { peaks: d.peaks, duration: d.duration } : null;
    }

    // the kicks the last run found on this track
    get kicks() {
        const d = this.data;
        return d?.mode === "music" && d.source === this.sourceName() ? d.kicks ?? [] : null;
    }

    setWidget(name, value) {
        const w = this.w(name);
        w.value = value;
        w.callback?.(value);
    }

    get duration() {
        if (!this.hasTrack()) return this.w("start_at").value + (this.w("duration").value || 10);
        return this.wave?.duration ?? this.data?.duration ?? 0;
    }

    // the beat grid from the last run on this track
    get grid() {
        const d = this.data;
        return d?.bpm && d.source === this.sourceName() && this.mode === "music" ? d : null;
    }

    // shots to show: hand-made, else the last run's, else a preview from the widgets
    layout() {
        const ed = this.stored.cuts;
        if (ed?.length > 1) return { cuts: ed, edited: true, exact: true };
        if (this.data?.cuts && this.data.key === this.runKey()) return { cuts: this.data.cuts, edited: false, exact: true };
        const n = this.imageCount(), duration = this.w("duration").value, g = this.grid;
        let start = this.w("start_at").value;
        if (g) {
            const bar = 240 / g.bpm;
            start = g.downbeat + Math.ceil((start - g.downbeat) / bar - 1e-6) * bar;
            const bars = duration ? Math.max(n, Math.round(duration / bar)) : n * Math.max(1, Math.round(SHOT_SECONDS / bar));
            const cuts = [start];
            for (let i = 0, at = 0; i < n; i++) {
                at += Math.floor(bars / n) + (i < bars % n ? 1 : 0);
                cuts.push(start + at * bar);
            }
            return { cuts, edited: false, exact: false };
        }
        let end = !this.hasTrack() ? this.duration : duration ? start + duration : this.mode === "music" ? start + n * SHOT_SECONDS : this.duration;
        if (end <= start) end = start + n * SHOT_SECONDS;
        return { cuts: Array.from({ length: n + 1 }, (_, i) => start + (i * (end - start)) / n), edited: false, exact: false };
    }

    get end() {
        return Math.max(this.duration, this.layout().cuts.at(-1));
    }

    view() {
        if (this.frozen) return this.frozen;
        if (this.viewMode === "manual" && this.manual) return this.manual;
        if (this.viewMode === "whole") return [0, this.end];
        const cuts = this.layout().cuts, margin = Math.max(1.5, 0.08 * (cuts.at(-1) - cuts[0]));
        return [Math.max(0, cuts[0] - margin), Math.min(this.end, cuts.at(-1) + margin)];
    }

    // move the view to [a, b], kept inside the track and at least half a second wide
    setView(a, b) {
        const span = Math.min(Math.max(b - a, 0.5), this.end + 2);
        a = Math.min(Math.max(a, 0), this.end + 2 - span);
        this.manual = [a, a + span];
        this.viewMode = "manual";
        if (this.frozen) this.frozen = this.manual;
    }

    onWheel(e) {
        e.preventDefault();
        e.stopPropagation();
        const [a, b] = this.view(), t = this.tOf(e), k = Math.exp(Math.sign(e.deltaY) * 0.15);
        this.setView(t - (t - a) * k, t + (b - t) * k);
        this.draw();
    }

    xOf(t) {
        const [a, b] = this.view(), w = this.canvas.clientWidth - PAD.l - PAD.r;
        return PAD.l + ((t - a) / Math.max(b - a, 1e-6)) * w;
    }

    tOf(e) {
        const r = this.canvas.getBoundingClientRect(), [a, b] = this.view(), w = r.width - PAD.l - PAD.r;
        return Math.max(a + ((e.clientX - r.left - PAD.l) / w) * (b - a), 0);
    }

    // music snaps cuts to bars; Alt: beats, Shift: free
    snap(t, e) {
        const g = this.grid;
        if (e.shiftKey || !g) return t;
        const step = (e.altKey ? 60 : 240) / g.bpm;
        return g.downbeat + Math.round((t - g.downbeat) / step) * step;
    }

    // what is under the pointer
    pick(e) {
        const r = this.canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
        const near = (t) => Math.abs(this.xOf(t) - mx);
        const lane = this.lanes().find((l) => my >= l.top && my < l.top + l.h);
        if (lane?.kind === "kicks") return { kind: "kicks" };
        if (lane?.kind === "hits") {
            let best = -1, dist = GRAB_PX;
            this.hits.forEach(([h], i) => { if (near(h) <= dist) { best = i; dist = near(h); } });
            return best >= 0 ? { kind: "hit", i: best } : { kind: "lane" };
        }
        const area = this.area();
        if (this.mode === "action" && my >= area.top && my < area.top + area.h) return { kind: "draw" };
        const cuts = this.layout().cuts, last = cuts.length - 1;
        let best = -1, dist = GRAB_PX;
        cuts.forEach((c, i) => { if (near(c) <= dist) { best = i; dist = near(c); } });
        if (best === 0) return { kind: "start" };
        if (best === last) return { kind: "end" };
        if (best > 0) return { kind: "cut", i: best };
        const t = this.tOf(e);
        const i = cuts.findIndex((c, k) => k < last && t >= c && t < cuts[k + 1]);
        return i >= 0 ? { kind: "shot", i } : { kind: "none" };
    }

    // run just this node and what feeds it, never the renders after it
    async analyse() {
        this.notice = "analysing…";
        this.draw();
        try {
            const prompt = await app.graphToPrompt();
            await api.queuePrompt(0, prompt, { partialExecutionTargets: [String(this.node.id)] });
        } catch (err) {
            this.notice = `could not analyse: ${err?.message ?? err}`;
            this.draw();
        }
    }

    onDouble(e) {
        e.stopPropagation();
        const p = this.pick(e), t = this.tOf(e);
        if (p.kind === "lane") return this.setEdits({ hits: [...this.hits, [t, this.w("hit_sound").value]] });
        if (p.kind === "hit") {
            const names = Object.keys(HIT_SOUNDS), hits = this.hits;
            hits[p.i] = [hits[p.i][0], names[(names.indexOf(hits[p.i][1]) + 1) % names.length]];
            return this.setEdits({ hits });
        }
        if (p.kind !== "shot") return;
        const cuts = this.layout().cuts, s = this.snap(t, e);
        if (cuts.some((c) => Math.abs(c - s) < MIN_SHOT) || s <= cuts[0] || s >= cuts[cuts.length - 1]) return;
        this.setEdits({ cuts: [...cuts, s] });
    }

    onRight(e) {
        e.preventDefault();
        e.stopPropagation();
        const p = this.pick(e);
        if (p.kind === "draw") return;
        if (p.kind === "hit") this.setEdits({ hits: this.hits.filter((_, j) => j !== p.i) });
        else if (p.kind === "cut") this.setEdits({ cuts: this.layout().cuts.filter((_, j) => j !== p.i) });

    }

    onDown(e) {
        e.stopPropagation();
        const p = this.pick(e);
        if (p.kind === "draw" && (e.button === 0 || e.button === 2)) {
            this.canvas.setPointerCapture(e.pointerId);
            const t = Math.max(0, this.tOf(e));
            this.paint = { erase: e.button === 2, last: t, from: t, to: t };
            this.frozen = [...this.view()];
            return this.stroke(e);
        }
        if (e.button !== 0) return;
        this.canvas.setPointerCapture(e.pointerId);
        if (["hit", "cut", "start", "end"].includes(p.kind)) {
            this.drag = p;
            this.frozen = [...this.view()];
        } else {
            this.pan = { x: e.clientX, view: [...this.view()] };
        }
    }

    onMove(e) {
        e.stopPropagation();
        this.notice = null;
        if (this.pan) {
            const [a, b] = this.pan.view, dt = ((e.clientX - this.pan.x) / (this.canvas.clientWidth - PAD.l - PAD.r)) * (b - a);
            if (Math.abs(e.clientX - this.pan.x) > 3) {
                this.setView(a - dt, b - dt);
                this.canvas.style.cursor = "grabbing";
                this.hover = { kind: "pan" };
            }
            this.hoverT = this.tOf(e);
            return this.draw();
        }
        this.hoverY = e.clientY;
        if (this.paint) {
            this.hoverT = this.tOf(e);
            return this.stroke(e);
        }
        if (!this.drag) {
            this.hoverT = this.tOf(e);
            this.hover = this.pick(e);
            const k = this.hover.kind;
            this.canvas.style.cursor = ["hit", "cut", "start", "end"].includes(k) ? "ew-resize" : k === "lane" ? "copy"
                : k === "draw" ? "crosshair" : k === "kicks" ? "default" : "grab";
            return this.draw();
        }
        const r = this.canvas.getBoundingClientRect();
        this.edge = e.clientX > r.right - 24 ? 1 : e.clientX < r.left + 24 ? -1 : 0;
        this.lastDrag = e;
        if (this.edge && !this.scrolling) this.scrollLoop();
        this.applyDrag(e);
    }

    // while a drag is held past an edge, keep scrolling that way, with the dragged item at the edge
    scrollLoop() {
        this.scrolling = requestAnimationFrame(() => {
            this.scrolling = null;
            if (!this.drag || !this.edge) return;
            const [a, b] = this.frozen, step = (b - a) * 0.015 * this.edge;
            this.setView(a + step, b + step);
            this.applyDrag(this.lastDrag);
            this.scrollLoop();
        });
    }

    applyDrag(e) {
        const p = this.drag, [a, b] = this.view();
        this.hoverT = Math.min(Math.max(this.tOf(e), a), b);
        if (p.kind === "hit") {
            const hits = this.hits, moved = +this.hoverT.toFixed(3), sound = hits[p.i][1];
            hits[p.i] = [moved, sound];
            this.setEdits({ hits });
            p.i = this.hits.findIndex(([t, s]) => t === moved && s === sound);
            p.t = moved;
            return;
        }
        const { cuts, edited } = this.layout(), last = cuts.length - 1;
        const t = this.snap(this.hoverT, e);
        if (p.kind === "cut" || edited) {
            const i = p.kind === "start" ? 0 : p.kind === "end" ? last : p.i;
            const next = [...cuts];
            const lo = i > 0 ? next[i - 1] + MIN_SHOT : 0;
            const hi = i < last ? next[i + 1] - MIN_SHOT : this.hasTrack() ? Math.max(this.duration, next[last]) : 3600;
            next[i] = Math.min(Math.max(t, lo), hi);
            p.t = next[i];
            return this.setEdits({ cuts: next });
        }
        // automatic shots: the handles set start_at and duration
        if (p.kind === "start") {
            const start = Math.max(0, Math.min(t, cuts[last] - MIN_SHOT));
            if (this.w("duration").value) this.setWidget("duration", +(cuts[last] - start).toFixed(2));
            this.setWidget("start_at", +start.toFixed(2));
            p.t = start;
        } else {
            const dur = Math.max(MIN_SHOT, t - this.w("start_at").value);
            this.setWidget("duration", +dur.toFixed(2));
            p.t = this.w("start_at").value + dur;
        }
        this.draw();
    }

    onUp(e) {
        e.stopPropagation();
        if (this.paint) this.endStroke();
        if (this.frozen) {
            // keep what is on screen: no jump when the drop changes the shots
            this.manual = this.frozen;
            this.viewMode = "manual";
        }
        this.drag = null;
        this.frozen = null;
        this.pan = null;
        this.edge = 0;
        this.canvas.releasePointerCapture(e.pointerId);
        this.draw();
    }

    // action: a button per sound to draw with; a drawn sound gets an x to clear it. Rebuilt when that changes.
    drawChips(show) {
        const drawn = this.curves.filter((c) => c.points.length).map((c) => c.sound);
        const key = show ? JSON.stringify([this.brush, drawn]) : "";
        if (key === this.chipsKey) return;
        this.chipsKey = key;
        this.chips.replaceChildren();
        this.chips.style.display = key ? "flex" : "none";
        if (!key) return;
        const title = document.createElement("span");
        title.textContent = "Draw:";
        this.chips.append(title);
        for (const [sound, color] of Object.entries(CURVE_SOUNDS)) {
            const picked = sound === this.brush;
            const chip = document.createElement("span");
            chip.title = `Draw ${sound}: ${CURVE_USES[sound]}`;
            chip.style.cssText = `display:inline-flex;gap:6px;align-items:center;padding:1px 8px;border-radius:10px;cursor:pointer;border:1px solid ${color};`
                + `color:${picked ? "#15171c" : color};background:${picked ? color : "transparent"};font-weight:${picked ? "bold" : "normal"};`;
            chip.textContent = sound;
            chip.onclick = () => { this.brush = sound; this.draw(); };
            if (drawn.includes(sound)) {
                const x = document.createElement("span");
                x.textContent = "✕";
                x.title = `Clear ${sound}`;
                x.style.cssText = "opacity:0.75;";
                x.onclick = (e) => {
                    e.stopPropagation();
                    this.setEdits({ curves: this.curves.filter((c) => c.sound !== sound) });
                };
                chip.append(x);
            }
            this.chips.append(chip);
        }
    }

    draw() {
        const c = this.canvas, dpr = window.devicePixelRatio || 1;
        const W = c.clientWidth, H = c.clientHeight;
        if (!W || !H) return;
        if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) {
            c.width = Math.round(W * dpr);
            c.height = Math.round(H * dpr);
        }
        const ctx = c.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.fillStyle = COLORS.bg;
        ctx.fillRect(0, 0, W, H);
        const lanes = this.lanes(), laneTop = PAD.t, bottom = H - PAD.b;
        const top = lanes.length ? lanes.at(-1).top + lanes.at(-1).h : PAD.t;
        const [a, b] = this.view(), { cuts, edited, exact } = this.layout(), g = this.grid, n = this.imageCount();
        ctx.font = "10px sans-serif";

        for (const lane of lanes) {
            ctx.fillStyle = COLORS.lane;
            ctx.fillRect(PAD.l, lane.top, W - PAD.l - PAD.r, lane.h - 2);
            ctx.fillStyle = COLORS.text;
            ctx.textAlign = "left";
            ctx.fillText(lane.kind === "kicks" ? (this.kicks ? "KICKS" : "KICKS: Analyse to find them") : "HITS", PAD.l + 4, lane.top + 13);
        }
        const kicks = this.mode === "music" ? this.kicks ?? [] : [];
        if (kicks.length) {
            ctx.strokeStyle = COLORS.kick;
            for (const k of kicks) {
                if (k < a || k > b) continue;
                const x = this.xOf(k);
                ctx.beginPath(); ctx.moveTo(x, laneTop + 3); ctx.lineTo(x, laneTop + HIT_LANE - 5); ctx.stroke();
            }
            ctx.fillStyle = COLORS.lane;
            ctx.fillRect(PAD.l, laneTop, 44, HIT_LANE - 2);
            ctx.fillStyle = COLORS.text;
            ctx.textAlign = "left";
            ctx.fillText("KICKS", PAD.l + 4, laneTop + 13);
        }

        // shots
        for (let i = 0; i < cuts.length - 1; i++) {
            const x0 = this.xOf(cuts[i]), x1 = this.xOf(cuts[i + 1]);
            ctx.fillStyle = COLORS.shots[i % 2];
            ctx.fillRect(x0, top, x1 - x0, bottom - top);
        }

        // beat grid
        if (g) {
            const beat = 60 / g.bpm;
            const k0 = Math.ceil((a - g.downbeat) / beat), k1 = Math.floor((b - g.downbeat) / beat);
            if (k1 - k0 < W / 3) {
                for (let k = k0; k <= k1; k++) {
                    const x = this.xOf(g.downbeat + k * beat);
                    ctx.strokeStyle = ((k % 4) + 4) % 4 === 0 ? COLORS.bar : COLORS.beat;
                    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
                }
            }
        }

        // waveform: below the shot labels, or under the drawing in action
        const wave = this.waveform, action = this.mode === "action", area = this.area(), strip = bottom - SHOT_STRIP;
        if (wave) {
            const wtop = action ? area.top + 2 : top + 34, half = ((action ? strip : bottom) - wtop) / 2 - 2, wmid = wtop + half + 2;
            const { peaks, duration } = wave;
            ctx.fillStyle = COLORS.wave;
            for (let px = 0; px < W - PAD.l - PAD.r; px++) {
                const t0 = a + (px / (W - PAD.l - PAD.r)) * (b - a), t1 = a + ((px + 1) / (W - PAD.l - PAD.r)) * (b - a);
                const i0 = Math.floor((t0 / duration) * peaks.length), i1 = Math.max(i0 + 1, Math.ceil((t1 / duration) * peaks.length));
                let m = 0;
                for (let i = i0; i < i1 && i < peaks.length; i++) m = Math.max(m, peaks[i]);
                ctx.fillRect(PAD.l + px, wmid - m * half, 1, Math.max(1, 2 * m * half));
            }
        }

        // action: the curves, the picked one on top
        if (action) {
            ctx.fillStyle = "rgba(10,11,14,0.45)";
            ctx.fillRect(PAD.l, strip, W - PAD.l - PAD.r, SHOT_STRIP);
            const order = this.curves.map((_, c) => c).filter((c) => c !== this.activeCurve).concat(this.activeCurve >= 0 ? [this.activeCurve] : []);
            for (const c of order) {
                const { sound, points } = this.curves[c], color = CURVE_SOUNDS[sound], picked = c === this.activeCurve;
                if (!points.length) continue;
                ctx.beginPath();
                ctx.moveTo(this.xOf(points[0][0]), this.yOf(0, area));
                for (const [t, v] of points) ctx.lineTo(this.xOf(t), this.yOf(v, area));
                ctx.lineTo(this.xOf(points.at(-1)[0]), this.yOf(0, area));
                ctx.fillStyle = color + (picked ? "40" : "1c");
                ctx.fill();
                ctx.strokeStyle = picked ? color : color + "99";
                ctx.lineWidth = picked ? 2 : 1.2;
                ctx.beginPath();
                points.forEach(([t, v], j) => (j ? ctx.lineTo : ctx.moveTo).call(ctx, this.xOf(t), this.yOf(v, area)));
                ctx.stroke();
                ctx.lineWidth = 1;
            }
        }

        // shot labels: at the top of the shots, or in the strip in action
        const labelTop = action ? strip + 4 : top + 4;
        for (let i = 0; i < cuts.length - 1; i++) {
            const x0 = Math.max(this.xOf(cuts[i]), PAD.l), x1 = Math.min(this.xOf(cuts[i + 1]), W - PAD.r);
            if (x1 - x0 < 46) continue;
            const text = `${i + 1} · img ${(i % n) + 1} · ${(cuts[i + 1] - cuts[i]).toFixed(1)} s`;
            ctx.font = "11px sans-serif";
            const tw = Math.min(ctx.measureText(text).width + 10, x1 - x0 - 4);
            ctx.fillStyle = "rgba(10,11,14,0.7)";
            ctx.fillRect((x0 + x1) / 2 - tw / 2, labelTop, tw, 18);
            ctx.fillStyle = "#e8ecf3";
            ctx.textAlign = "center";
            ctx.fillText(text, (x0 + x1) / 2, labelTop + 13, tw - 6);
        }
        ctx.font = "10px sans-serif";

        // cuts (white) and the range handles (yellow), thicker when hovered or dragged
        cuts.forEach((t, i) => {
            const x = this.xOf(t), handle = i === 0 || i === cuts.length - 1;
            const active = [this.drag, this.hover].some((p) => p && ((p.kind === "cut" && p.i === i) || (p.kind === "start" && i === 0) || (p.kind === "end" && i === cuts.length - 1)));
            ctx.strokeStyle = handle ? COLORS.range : COLORS.cut;
            ctx.lineWidth = active ? 3 : 1.5;
            ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
            ctx.fillStyle = ctx.strokeStyle;
            ctx.beginPath(); ctx.moveTo(x - 6, bottom + 8); ctx.lineTo(x + 6, bottom + 8); ctx.lineTo(x, bottom); ctx.fill();
        });
        ctx.lineWidth = 1;

        // hits: a diamond in the lane, coloured by its sound, and a line through the shots
        if (this.mode === "action") {
            this.hits.forEach(([h, sound], i) => {
                const x = this.xOf(h), active = this.hover.kind === "hit" && this.hover.i === i, color = HIT_SOUNDS[sound] ?? HIT_SOUNDS.thump;
                ctx.strokeStyle = color;
                ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
                ctx.fillStyle = color;
                const r = active ? 7 : 5;
                ctx.beginPath(); ctx.moveTo(x, laneTop + 9 - r); ctx.lineTo(x + r, laneTop + 9); ctx.lineTo(x, laneTop + 9 + r); ctx.lineTo(x - r, laneTop + 9); ctx.fill();
            });
        }

        // nothing drawn with the picked sound yet: say how, over everything in the middle of the area
        if (action && this.activeCurve < 0 && !this.paint && !this.help) {
            const y = area.top + area.h / 2, title = `Drag here to draw ${this.brush} (${CURVE_USES[this.brush]})`;
            const sub = "higher is louder or faster · right-drag to erase · pick another sound above";
            ctx.font = "11px sans-serif";
            const bw = Math.min(Math.max(ctx.measureText(sub).width, 260) + 28, W - 2 * PAD.l);
            ctx.fillStyle = "rgba(10,11,14,0.85)";
            ctx.fillRect(W / 2 - bw / 2, y - 26, bw, 44);
            ctx.textAlign = "center";
            ctx.fillStyle = CURVE_SOUNDS[this.brush];
            ctx.font = "bold 13px sans-serif";
            ctx.fillText(title, W / 2, y - 8, bw - 12);
            ctx.font = "11px sans-serif";
            ctx.fillStyle = "#c9ced8";
            ctx.fillText(sub, W / 2, y + 10, bw - 12);
            ctx.font = "10px sans-serif";
        }

        // time axis and hover
        ctx.fillStyle = COLORS.text;
        ctx.textAlign = "left";
        ctx.fillText(fmt(a), PAD.l, H - 4);
        ctx.textAlign = "right";
        ctx.fillText(fmt(b), W - PAD.r, H - 4);
        if (this.drag?.t !== undefined) {
            const x = Math.min(Math.max(this.xOf(this.drag.t), 40), W - 40), text = fmt(this.drag.t);
            ctx.font = "bold 11px sans-serif";
            const tw = ctx.measureText(text).width + 10;
            ctx.fillStyle = COLORS.range;
            ctx.fillRect(x - tw / 2, bottom - 22, tw, 16);
            ctx.fillStyle = "#15171c";
            ctx.textAlign = "center";
            ctx.fillText(text, x, bottom - 10);
            ctx.font = "10px sans-serif";
        }
        if (this.hoverT !== null && !this.help && !this.drag && !this.paint) {
            const x = this.xOf(this.hoverT);
            ctx.strokeStyle = COLORS.hover;
            ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(x, laneTop); ctx.lineTo(x, bottom); ctx.stroke();
            ctx.setLineDash([]);
            if (this.hover.kind === "draw") {
                const r = this.canvas.getBoundingClientRect();
                ctx.fillStyle = COLORS.hover;
                ctx.textAlign = "left";
                ctx.fillText(`${Math.round(this.vOf(this.hoverY - r.top, area) * 100)}%`, Math.min(x + 4, W - 40), area.top + area.h - 4);
            }
            ctx.fillStyle = COLORS.hover;
            ctx.textAlign = "center";
            ctx.fillText(fmt(this.hoverT), Math.min(Math.max(x, 30), W - 30), H - 4);
        }

        if (this.help) {
            ctx.fillStyle = "rgba(10,11,14,0.92)";
            ctx.fillRect(0, 0, W, H);
            ctx.fillStyle = "#dfe3ea";
            ctx.textAlign = "left";
            ctx.font = "11px sans-serif";
            const lines = [];
            for (const text of HELP) {
                let line = "";
                for (const word of text.split(" ")) {
                    if (ctx.measureText(line + word).width > W - 24 && line) { lines.push(line); line = ""; }
                    line += word + " ";
                }
                lines.push(line, "");
            }
            lines.slice(0, Math.floor((H - 12) / 13)).forEach((l, i) => ctx.fillText(l, 12, 16 + i * 13));
        }

        this.helpButton.textContent = this.help ? "Close help" : "?";
        this.clearButton.style.display = action ? "" : "none";
        this.drawChips(action);
        if (this.canvas.style.height !== `${this.height()}px`) this.canvas.style.height = `${this.height()}px`;
        const parts = [this.mode];
        if (g) parts.push(`${g.bpm.toFixed(2)} BPM`);
        parts.push(`${cuts.length - 1} shot${cuts.length > 2 ? "s" : ""}, ${(cuts[cuts.length - 1] - cuts[0]).toFixed(1)} s`);
        parts.push(edited ? "edited by hand" : exact ? "automatic" : "preview: Analyse for exact cuts");
        if (this.mode === "music" && this.kicks) parts.push(`${this.kicks.filter((k) => k >= cuts[0] && k < cuts.at(-1)).length} kicks`);
        if (action && this.hits.length) parts.push(`${this.hits.length} hits`);
        if (action && this.curves.length) parts.push(`${this.curves.length} curve${this.curves.length > 1 ? "s" : ""}`);
        if (!this.hasTrack()) parts.push(this.hits.length || this.curves.length ? "no track: what you draw is the sound" : "no track: draw hits or curves");
        else if (!this.waveform) parts.push("Analyse to see the waveform");
        this.label.textContent = parts.join(" · ");
        const kind = this.drag?.kind ?? this.hover.kind, hit = kind === "hit" && this.hits[(this.drag ?? this.hover).i];
        this.hint.textContent = this.notice ?? (hit ? `${hit[1]} hit · drag to move · double-click to change the sound · right-click to delete`
            : kind === "lane" ? `double-click to add a ${this.w("hit_sound").value} hit here (hit_sound sets the sound)` : HINTS[kind] ?? HINTS.none);
    }
}

app.registerExtension({
    name: "Syntax.Timeline",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const edits = this.widgets.find((w) => w.name === "edits");
            edits.hidden = true;
            edits.options.hidden = true;
            edits.computeSize = () => [0, 0];

            const container = document.createElement("div");
            const widget = this.addDOMWidget("timeline_editor", "syntax_timeline", container, { serialize: false, hideOnZoom: false });
            widget.computeSize = (width) => [width, (this.syntaxTimeline?.height() ?? CANVAS_H) + 74];
            this.syntaxTimeline = new TimelineEditor(this, container);
            // typing a new mode, start or length goes back to automatic shots
            for (const name of ["mode", "start_at", "duration", "bpm", "hit_sound"]) {
                const w = this.widgets.find((x) => x.name === name);
                const cb = w.callback;
                w.callback = (...args) => {
                    const out = cb?.apply(w, args);
                    const tl = this.syntaxTimeline;
                    if (name !== "hit_sound" && !tl.drag && tl.stored.cuts) tl.setEdits({ cuts: null });
                    if (name === "mode") tl.resize();
                    tl.draw();
                    return out;
                };
            }
            this.setSize([Math.max(this.size[0], 760), this.computeSize()[1]]);
            return r;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const r = onConnectionsChange?.apply(this, arguments);
            requestAnimationFrame(() => { this.syntaxTimeline?.resize(); this.syntaxTimeline?.draw(); });
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted?.apply(this, arguments);
            const tl = this.syntaxTimeline;
            if (message?.timeline?.[0]) {
                tl.notice = null;
                tl.data = { ...message.timeline[0], key: tl.runKey(), source: tl.sourceName() };
                tl.draw();
            }
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            // workflows saved before music_guide existed load a stray value into it
            const guide = this.widgets.find((w) => w.name === "music_guide");
            if (!guide.options.values.includes(guide.value)) guide.value = "drums";
            requestAnimationFrame(() => { this.syntaxTimeline?.checkSource(); this.syntaxTimeline?.resize(); this.syntaxTimeline?.draw(); });
            return r;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            clearInterval(this.syntaxTimeline?.timer);
            return onRemoved?.apply(this, arguments);
        };
    },
});
