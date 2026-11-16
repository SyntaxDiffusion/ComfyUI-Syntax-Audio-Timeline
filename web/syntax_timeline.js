import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "SyntaxTimeline";
const CANVAS_H = 170;
const PAD = { l: 8, r: 8, t: 18, b: 20 };
const GRAB_PX = 7;
const MIN_SHOT = 0.25;
const COLORS = {
    bg: "#15171c", wave: "#5b6475", text: "#9aa1b0", beat: "rgba(255,255,255,0.06)", bar: "rgba(255,255,255,0.16)",
    cut: "#e8ecf3", start: "#ffd633", hover: "#ffd633", hit: "#ff4050", shots: ["rgba(77,158,255,0.22)", "rgba(255,140,77,0.22)"],
};

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
        this.data = null;          // last run: {duration, mode, cuts, bpm, downbeat}
        this.wave = null;          // {peaks, duration, name}
        this.whole = false;
        this.drag = -1;
        this.hoverT = null;

        container.style.cssText = "display:flex;flex-direction:column;gap:4px;width:100%;";
        const bar = document.createElement("div");
        bar.style.cssText = "display:flex;gap:4px;align-items:center;font:11px sans-serif;color:#9aa1b0;";
        this.label = document.createElement("span");
        this.label.style.cssText = "flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
        const button = (text, title, fn) => {
            const b = document.createElement("button");
            b.textContent = text;
            b.title = title;
            b.style.cssText = "background:#222;color:#ddd;border:1px solid #444;border-radius:4px;padding:1px 6px;cursor:pointer;";
            b.onclick = fn;
            return b;
        };
        this.viewButton = button("Whole track", "Show the whole track or zoom to the shots", () => { this.whole = !this.whole; this.draw(); });
        bar.append(this.label, this.viewButton,
            button("Reset cuts", "Forget moved cuts and use the automatic ones", () => this.setEdits({ cuts: null })),
            button("Clear hits", "Remove every marked hit", () => this.setEdits({ hits: null })));

        this.canvas = document.createElement("canvas");
        this.canvas.style.cssText = `width:100%;height:${CANVAS_H}px;border-radius:6px;touch-action:none;`;
        container.append(bar, this.canvas);
        this.canvas.addEventListener("pointerdown", (e) => this.onDown(e));
        this.canvas.addEventListener("pointermove", (e) => this.onMove(e));
        this.canvas.addEventListener("pointerup", (e) => this.onUp(e));
        this.canvas.addEventListener("pointerleave", () => { this.hoverT = null; this.draw(); });
        this.canvas.addEventListener("dblclick", (e) => this.onDouble(e));
        this.canvas.addEventListener("contextmenu", (e) => this.onRight(e));
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

    get stored() {
        try {
            return JSON.parse(this.w("edits").value || "{}");
        } catch {
            return {};
        }
    }

    get edits() {
        return this.stored.cuts ?? null;
    }

    get hits() {
        return this.stored.hits ?? [];
    }

    // merge {cuts, hits} into the stored edits; null removes a key
    setEdits(change) {
        const next = { ...this.stored, ...change };
        for (const k of Object.keys(next)) {
            if (!next[k]?.length) delete next[k];
            else next[k] = next[k].map((t) => +t.toFixed(3)).sort((x, y) => x - y);
        }
        const w = this.w("edits");
        w.value = Object.keys(next).length ? JSON.stringify(next) : "";
        w.callback?.(w.value);
        this.node.setDirtyCanvas(true, true);
        this.draw();
    }

    hasTrack() {
        const slot = this.node.findInputSlot("track");
        return slot >= 0 && this.node.inputs[slot].link != null;
    }

    // cuts to show: moved ones if they fit the last run's shot count, else the last run's
    get cuts() {
        const run = this.data?.cuts ?? null, ed = this.edits;
        if (ed && (!run || ed.length === run.length)) return ed;
        return run;
    }

    get duration() {
        if (!this.hasTrack()) return this.w("duration").value;
        return this.wave?.duration ?? this.data?.duration ?? 0;
    }

    view() {
        const d = this.duration, cuts = this.cuts;
        if (this.whole || !cuts) return [0, d];
        return [Math.max(0, cuts[0] - 1.5), Math.min(d, cuts[cuts.length - 1] + 1.5)];
    }

    xOf(t) {
        const [a, b] = this.view(), w = this.canvas.clientWidth - PAD.l - PAD.r;
        return PAD.l + ((t - a) / Math.max(b - a, 1e-6)) * w;
    }

    tOf(e) {
        const r = this.canvas.getBoundingClientRect(), [a, b] = this.view(), w = r.width - PAD.l - PAD.r;
        return Math.min(Math.max(a + ((e.clientX - r.left - PAD.l) / w) * (b - a), 0), this.duration);
    }

    snap(t, e) {
        const d = this.data;
        if (e.shiftKey || this.w("cut_on").value !== "beats" || !d?.bpm) return t;
        const step = (e.altKey ? 60 : 240) / d.bpm;
        return d.downbeat + Math.round((t - d.downbeat) / step) * step;
    }

    // index of the hit marker under the pointer (they sit in the top lane), or -1
    hitAt(e) {
        const r = this.canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
        if (my > PAD.t + 16) return -1;
        let best = -1, dist = GRAB_PX;
        this.hits.forEach((h, i) => { const dd = Math.abs(this.xOf(h) - mx); if (dd <= dist) { best = i; dist = dd; } });
        return best;
    }

    onDouble(e) {
        e.stopPropagation();
        this.setEdits({ hits: [...this.hits, this.snap(this.tOf(e), e)] });
    }

    onRight(e) {
        e.preventDefault();
        e.stopPropagation();
        const i = this.hitAt(e);
        if (i >= 0) this.setEdits({ hits: this.hits.filter((_, j) => j !== i) });
    }

    onDown(e) {
        e.stopPropagation();
        if (e.button !== 0) return;
        const mx = e.clientX - this.canvas.getBoundingClientRect().left;
        const cuts = this.cuts;
        this.dragHit = this.hitAt(e);
        if (this.dragHit >= 0) {
            this.drag = -1;
            this.canvas.setPointerCapture(e.pointerId);
            return;
        }
        if (cuts) {
            let best = -1, dist = GRAB_PX;
            cuts.forEach((c, i) => { const dd = Math.abs(this.xOf(c) - mx); if (dd <= dist) { best = i; dist = dd; } });
            this.drag = best;
        } else {
            this.drag = Math.abs(this.xOf(this.w("start_at").value) - mx) <= GRAB_PX ? 0 : -1;
        }
        if (this.drag >= 0) this.canvas.setPointerCapture(e.pointerId);
    }

    onMove(e) {
        e.stopPropagation();
        this.hoverT = this.tOf(e);
        if (this.dragHit >= 0) {
            const hits = [...this.hits];
            hits[this.dragHit] = this.snap(this.tOf(e), e);
            const moved = hits[this.dragHit];
            this.setEdits({ hits });
            this.dragHit = this.hits.indexOf(+moved.toFixed(3));
            return;
        }
        if (this.drag < 0) return this.draw();
        const t = this.snap(this.tOf(e), e);
        const cuts = this.cuts;
        if (!cuts) {
            const w = this.w("start_at");
            w.value = +Math.max(0, t).toFixed(2);
            w.callback?.(w.value);
            return this.draw();
        }
        const next = [...cuts];
        const lo = this.drag > 0 ? next[this.drag - 1] + MIN_SHOT : 0;
        const hi = this.drag < next.length - 1 ? next[this.drag + 1] - MIN_SHOT : this.duration;
        next[this.drag] = Math.min(Math.max(t, lo), hi);
        this.setEdits({ cuts: next });
    }

    onUp(e) {
        e.stopPropagation();
        this.drag = -1;
        this.dragHit = -1;
        this.canvas.releasePointerCapture(e.pointerId);
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
        const top = PAD.t, bottom = H - PAD.b, mid = (top + bottom) / 2;
        const [a, b] = this.view(), cuts = this.cuts, d = this.data;
        ctx.font = "10px sans-serif";

        if (!this.duration) {
            ctx.fillStyle = COLORS.text;
            ctx.textAlign = "center";
            ctx.fillText("Connect a Load Audio to track, or run once, to see the timeline", W / 2, mid);
            this.label.textContent = "";
            return;
        }

        // shots
        if (cuts) {
            for (let i = 0; i < cuts.length - 1; i++) {
                const x0 = this.xOf(cuts[i]), x1 = this.xOf(cuts[i + 1]);
                ctx.fillStyle = COLORS.shots[i % 2];
                ctx.fillRect(x0, top, x1 - x0, bottom - top);
                ctx.fillStyle = COLORS.text;
                ctx.textAlign = "center";
                ctx.fillText(`${i + 1} · ${(cuts[i + 1] - cuts[i]).toFixed(1)} s`, (x0 + x1) / 2, top - 5);
            }
        }

        // beat grid
        if (d?.bpm && this.w("cut_on").value === "beats") {
            const beat = 60 / d.bpm;
            const k0 = Math.ceil((a - d.downbeat) / beat), k1 = Math.floor((b - d.downbeat) / beat);
            if (k1 - k0 < W / 3) {
                for (let k = k0; k <= k1; k++) {
                    const x = this.xOf(d.downbeat + k * beat);
                    ctx.strokeStyle = ((k % 4) + 4) % 4 === 0 ? COLORS.bar : COLORS.beat;
                    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
                }
            }
        }

        // waveform
        if (this.wave) {
            const { peaks, duration } = this.wave, half = (bottom - top) / 2 - 2;
            ctx.fillStyle = COLORS.wave;
            for (let px = 0; px < W - PAD.l - PAD.r; px++) {
                const t0 = a + (px / (W - PAD.l - PAD.r)) * (b - a), t1 = a + ((px + 1) / (W - PAD.l - PAD.r)) * (b - a);
                const i0 = Math.floor((t0 / duration) * peaks.length), i1 = Math.max(i0 + 1, Math.ceil((t1 / duration) * peaks.length));
                let m = 0;
                for (let i = i0; i < i1 && i < peaks.length; i++) m = Math.max(m, peaks[i]);
                ctx.fillRect(PAD.l + px, mid - m * half, 1, Math.max(1, 2 * m * half));
            }
        }

        // cuts, or the start marker before the first run
        const marks = cuts ?? [this.w("start_at").value];
        marks.forEach((t, i) => {
            const x = this.xOf(t);
            ctx.strokeStyle = cuts ? COLORS.cut : COLORS.start;
            ctx.lineWidth = i === this.drag ? 2.5 : 1.5;
            ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
            ctx.fillStyle = ctx.strokeStyle;
            ctx.beginPath(); ctx.moveTo(x - 5, bottom); ctx.lineTo(x + 5, bottom); ctx.lineTo(x, bottom - 7); ctx.fill();
        });
        ctx.lineWidth = 1;

        // hits: a line and a diamond in the top lane
        for (const h of this.hits) {
            const x = this.xOf(h);
            ctx.strokeStyle = COLORS.hit;
            ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
            ctx.fillStyle = COLORS.hit;
            ctx.beginPath(); ctx.moveTo(x, top + 1); ctx.lineTo(x + 5, top + 7); ctx.lineTo(x, top + 13); ctx.lineTo(x - 5, top + 7); ctx.fill();
        }
        if (!this.hasTrack() && !this.hits.length) {
            ctx.fillStyle = COLORS.text;
            ctx.textAlign = "center";
            ctx.fillText("No track: double-click to mark where the action hits", W / 2, mid);
        }

        // time axis and hover
        ctx.fillStyle = COLORS.text;
        ctx.textAlign = "left";
        ctx.fillText(`${a.toFixed(1)} s`, PAD.l, H - 5);
        ctx.textAlign = "right";
        ctx.fillText(`${b.toFixed(1)} s`, W - PAD.r, H - 5);
        if (this.hoverT !== null) {
            const x = this.xOf(this.hoverT);
            ctx.strokeStyle = COLORS.hover;
            ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = COLORS.hover;
            ctx.textAlign = "center";
            ctx.fillText(`${this.hoverT.toFixed(2)} s`, Math.min(Math.max(x, 30), W - 30), H - 5);
        }

        this.viewButton.textContent = this.whole ? "Shots" : "Whole track";
        const mode = this.w("cut_on").value;
        const parts = [mode];
        if (d?.bpm && mode === "beats") parts.push(`${d.bpm.toFixed(2)} BPM`);
        if (this.hits.length) parts.push(`${this.hits.length} hits`);
        if (cuts) parts.push(`${cuts.length - 1} shots`, this.edits ? "cuts moved by hand" : "automatic cuts");
        else parts.push("run once to see the cuts; drag the yellow marker to set start_at");
        this.label.textContent = parts.join(" · ");
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
            widget.computeSize = (width) => [width, CANVAS_H + 30];
            this.syntaxTimeline = new TimelineEditor(this, container);
            for (const name of ["cut_on", "start_at", "duration"]) {
                const w = this.widgets.find((x) => x.name === name);
                const cb = w.callback;
                w.callback = (...args) => { const out = cb?.apply(w, args); this.syntaxTimeline.draw(); return out; };
            }
            this.setSize([Math.max(this.size[0], 460), this.computeSize()[1]]);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted?.apply(this, arguments);
            if (message?.timeline?.[0]) {
                this.syntaxTimeline.data = message.timeline[0];
                this.syntaxTimeline.draw();
            }
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            requestAnimationFrame(() => { this.syntaxTimeline?.checkSource(); this.syntaxTimeline?.draw(); });
            return r;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            clearInterval(this.syntaxTimeline?.timer);
            return onRemoved?.apply(this, arguments);
        };
    },
});
