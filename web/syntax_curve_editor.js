import { app } from "../../scripts/app.js";

const NODE = "SyntaxActionCurveEditor";
const CANVAS_H = 220;
const PAD = { l: 30, r: 12, t: 12, b: 22 };
const HIT_R = 8;
const COLORS = { bg: "#15171c", grid: "#262a33", text: "#7d8494", curve: "#4d9eff", fill: "rgba(77,158,255,0.16)", point: "#e8ecf3", playhead: "#ffd633" };

const PRESETS = {
    "Impulses x4": [[0, 0], [0.148, 0], [0.15, 1], [0.152, 0], [0.368, 0], [0.37, 1], [0.372, 0], [0.588, 0], [0.59, 1], [0.592, 0], [0.808, 0], [0.81, 1], [0.812, 0], [1, 0]],
    "Spikes x3": [[0, 0], [0.12, 0], [0.125, 1], [0.25, 0.05], [0.37, 0], [0.375, 1], [0.5, 0.05], [0.62, 0], [0.625, 1], [0.75, 0.05], [1, 0]],
    "Hits x4": [[0, 0], [0.15, 0], [0.155, 1], [0.21, 1], [0.215, 0], [0.35, 0], [0.355, 1], [0.41, 1], [0.415, 0], [0.55, 0], [0.555, 1], [0.61, 1], [0.615, 0], [0.75, 0], [0.755, 1], [0.81, 1], [0.815, 0], [1, 0]],
    "Step on": [[0, 0], [0.55, 0], [0.56, 1], [1, 1]],
    "Build + cut": [[0, 0], [0.6, 1], [0.61, 0], [1, 0]],
    "Pulse x3": [[0, 0.05], [0.1667, 1], [0.3333, 0.05], [0.5, 1], [0.6667, 0.05], [0.8333, 1], [1, 0.05]],
    "Ramp + peak": [[0, 0], [0.6, 0.45], [0.7, 1], [0.78, 0.35], [1, 0.15]],
    "Gallop": [[0, 0.3], [0.125, 0.9], [0.25, 0.3], [0.375, 0.9], [0.5, 0.3], [0.625, 0.9], [0.75, 0.3], [0.875, 0.9], [1, 0.3]],
    "Single hit": [[0, 0.05], [0.45, 0.05], [0.5, 1], [0.6, 0.1], [1, 0.05]],
    "Ease in": [[0, 0], [0.6, 0.15], [1, 1]],
    "Flat": [[0, 0.5], [1, 0.5]],
};

const clamp01 = (v) => Math.min(Math.max(v, 0), 1);

// Same math as curves.py: monotone cubic (Fritsch-Carlson), linear, step
function tangents(xs, ys) {
    const n = xs.length;
    const d = [];
    for (let i = 0; i < n - 1; i++) d.push(xs[i + 1] > xs[i] ? (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) : 0);
    const m = [d[0]];
    for (let i = 1; i < n - 1; i++) m.push(d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2);
    m.push(d[n - 2]);
    for (let i = 0; i < n - 1; i++) {
        if (d[i] === 0) { m[i] = m[i + 1] = 0; continue; }
        const a = m[i] / d[i], b = m[i + 1] / d[i], s = a * a + b * b;
        if (s > 9) { const t = 3 / Math.sqrt(s); m[i] = t * a * d[i]; m[i + 1] = t * b * d[i]; }
    }
    return m;
}

function evaluate(points, interpolation, x) {
    const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
    let seg = 0;
    while (xs[seg + 1] < x) seg++;
    const x0 = xs[seg], x1 = xs[seg + 1], y0 = ys[seg], y1 = ys[seg + 1], h = x1 - x0;
    const t = h > 0 ? (x - x0) / h : 1;
    if (interpolation === "step") return t < 1 ? y0 : y1;
    if (interpolation === "linear") return y0 + (y1 - y0) * t;
    const m = tangents(xs, ys), t2 = t * t, t3 = t2 * t;
    return clamp01((2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * h * m[seg] + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * h * m[seg + 1]);
}

function parsePoints(value) {
    try {
        const pts = JSON.parse(value).points.map(([x, y]) => [clamp01(+x), clamp01(+y)]);
        return pts.length ? pts.sort((a, b) => a[0] - b[0]) : null;
    } catch {
        return null;
    }
}

class CurveEditor {
    constructor(node, container) {
        this.node = node;
        this.dataWidget = node.widgets.find((w) => w.name === "curve_data");
        this.interpWidget = node.widgets.find((w) => w.name === "interpolation");
        this.lengthWidget = node.widgets.find((w) => w.name === "length");
        this.drag = -1;
        this.hover = null;

        container.style.cssText = "display:flex;flex-direction:column;gap:4px;width:100%;";
        const bar = document.createElement("div");
        bar.style.cssText = "display:flex;gap:4px;align-items:center;font:11px sans-serif;";
        const preset = document.createElement("select");
        preset.style.cssText = "flex:1;min-width:0;background:#222;color:#ddd;border:1px solid #444;border-radius:4px;padding:1px 2px;";
        preset.append(new Option("Preset…", ""));
        for (const name of Object.keys(PRESETS)) preset.append(new Option(name, name));
        preset.onchange = () => {
            if (preset.value) this.setPoints(PRESETS[preset.value].map((p) => [...p]));
            preset.value = "";
        };
        const button = (label, title, fn) => {
            const b = document.createElement("button");
            b.textContent = label;
            b.title = title;
            b.style.cssText = "background:#222;color:#ddd;border:1px solid #444;border-radius:4px;padding:1px 6px;cursor:pointer;";
            b.onclick = fn;
            return b;
        };
        bar.append(
            preset,
            button("Copy", "Copy curve JSON to the clipboard", () => navigator.clipboard.writeText(this.dataWidget.value)),
            button("Paste", "Load curve JSON from the clipboard", async () => {
                const pts = parsePoints(await navigator.clipboard.readText());
                if (pts) this.setPoints(pts);
            }),
        );

        this.canvas = document.createElement("canvas");
        this.canvas.style.cssText = `width:100%;height:${CANVAS_H}px;border-radius:6px;cursor:crosshair;touch-action:none;`;
        container.append(bar, this.canvas);

        this.canvas.addEventListener("pointerdown", (e) => this.onDown(e));
        this.canvas.addEventListener("pointermove", (e) => this.onMove(e));
        this.canvas.addEventListener("pointerup", (e) => this.onUp(e));
        this.canvas.addEventListener("pointerleave", () => { this.hover = null; this.draw(); });
        this.canvas.addEventListener("contextmenu", (e) => this.onDelete(e));
        new ResizeObserver(() => this.draw()).observe(this.canvas);
    }

    get points() {
        return parsePoints(this.dataWidget.value) ?? [[0, 0.5], [1, 0.5]];
    }

    setPoints(points) {
        const rounded = points.map(([x, y]) => [+x.toFixed(4), +y.toFixed(4)]);
        this.dataWidget.value = JSON.stringify({ points: rounded });
        this.dataWidget.callback?.(this.dataWidget.value);
        this.node.setDirtyCanvas(true, true);
        this.draw();
    }

    toCurve(e) {
        const r = this.canvas.getBoundingClientRect();
        const w = r.width - PAD.l - PAD.r, h = r.height - PAD.t - PAD.b;
        return [clamp01((e.clientX - r.left - PAD.l) / w), clamp01(1 - (e.clientY - r.top - PAD.t) / h)];
    }

    hit(e) {
        const r = this.canvas.getBoundingClientRect();
        const w = r.width - PAD.l - PAD.r, h = r.height - PAD.t - PAD.b;
        const mx = e.clientX - r.left, my = e.clientY - r.top;
        return this.points.findIndex(([x, y]) => Math.hypot(PAD.l + x * w - mx, PAD.t + (1 - y) * h - my) <= HIT_R);
    }

    onDown(e) {
        e.stopPropagation();
        if (e.button !== 0) return;
        const pts = this.points;
        let i = this.hit(e);
        if (i < 0) {
            const added = this.toCurve(e);
            pts.push(added);
            pts.sort((a, b) => a[0] - b[0]);
            i = pts.indexOf(added);
            this.setPoints(pts);
        }
        this.drag = i;
        this.canvas.setPointerCapture(e.pointerId);
    }

    onMove(e) {
        e.stopPropagation();
        this.hover = this.toCurve(e)[0];
        if (this.drag < 0) return this.draw();
        const pts = this.points;
        const [x, y] = this.toCurve(e);
        const lo = this.drag > 0 ? pts[this.drag - 1][0] : 0;
        const hi = this.drag < pts.length - 1 ? pts[this.drag + 1][0] : 1;
        pts[this.drag] = [Math.min(Math.max(x, lo), hi), y];
        this.setPoints(pts);
    }

    onUp(e) {
        e.stopPropagation();
        this.drag = -1;
        this.canvas.releasePointerCapture(e.pointerId);
    }

    onDelete(e) {
        e.preventDefault();
        e.stopPropagation();
        const pts = this.points;
        const i = this.hit(e);
        if (i >= 0 && pts.length > 1) {
            pts.splice(i, 1);
            this.setPoints(pts);
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
        const w = W - PAD.l - PAD.r, h = H - PAD.t - PAD.b;
        const X = (x) => PAD.l + x * w, Y = (y) => PAD.t + (1 - y) * h;
        const pts = this.points, interp = this.interpWidget.value, frames = this.lengthWidget.value;

        ctx.fillStyle = COLORS.bg;
        ctx.fillRect(0, 0, W, H);
        ctx.strokeStyle = COLORS.grid;
        ctx.lineWidth = 1;
        ctx.fillStyle = COLORS.text;
        ctx.font = "10px sans-serif";
        for (let i = 0; i <= 4; i++) {
            const y = Y(i / 4);
            ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(PAD.l + w, y); ctx.stroke();
            ctx.textAlign = "right"; ctx.textBaseline = "middle";
            ctx.fillText((i / 4).toFixed(2).replace(/0$/, ""), PAD.l - 4, y);
        }
        for (let i = 0; i <= 8; i++) {
            const x = X(i / 8);
            ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, PAD.t + h); ctx.stroke();
            ctx.textAlign = "center"; ctx.textBaseline = "top";
            ctx.fillText(String(Math.round((i / 8) * (frames - 1))), x, PAD.t + h + 5);
        }

        // sample every pixel plus every key, so spikes narrower than a pixel still draw
        const xs = Array.from({ length: w + 1 }, (_, px) => px / w).concat(pts.map((p) => p[0])).sort((a, b) => a - b);
        ctx.beginPath();
        for (const [i, x] of xs.entries()) {
            const y = Y(evaluate(pts, interp, x));
            i === 0 ? ctx.moveTo(X(x), y) : ctx.lineTo(X(x), y);
        }
        ctx.strokeStyle = COLORS.curve;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.lineTo(PAD.l + w, PAD.t + h);
        ctx.lineTo(PAD.l, PAD.t + h);
        ctx.closePath();
        ctx.fillStyle = COLORS.fill;
        ctx.fill();

        for (const [i, [x, y]] of pts.entries()) {
            ctx.beginPath();
            ctx.arc(X(x), Y(y), i === this.drag ? 6 : 4.5, 0, Math.PI * 2);
            ctx.fillStyle = COLORS.point;
            ctx.fill();
            ctx.strokeStyle = COLORS.curve;
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        if (this.hover !== null) {
            const frame = Math.round(this.hover * (frames - 1));
            const fx = frames > 1 ? frame / (frames - 1) : 0;
            // keys land on their nearest frame, as in curves.py
            const v = Math.max(evaluate(pts, interp, fx), ...pts.filter((p) => Math.round(p[0] * (frames - 1)) === frame).map((p) => p[1]));
            ctx.strokeStyle = COLORS.playhead;
            ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.moveTo(X(fx), PAD.t); ctx.lineTo(X(fx), PAD.t + h); ctx.stroke();
            ctx.fillStyle = COLORS.playhead;
            ctx.textAlign = fx > 0.8 ? "right" : "left";
            ctx.textBaseline = "top";
            ctx.fillText(`frame ${frame} · ${v.toFixed(2)}`, X(fx) + (fx > 0.8 ? -5 : 5), PAD.t + 2);
        }
    }
}

app.registerExtension({
    name: "Syntax.ActionCurves",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const data = this.widgets.find((w) => w.name === "curve_data");
            data.hidden = true;
            data.options.hidden = true;
            data.computeSize = () => [0, 0];

            const container = document.createElement("div");
            const widget = this.addDOMWidget("curve_editor", "syntax_curve_editor", container, { serialize: false, hideOnZoom: false });
            widget.computeSize = (width) => [width, CANVAS_H + 34];
            const editor = new CurveEditor(this, container);
            this.syntaxCurveEditor = editor;

            for (const w of [editor.interpWidget, editor.lengthWidget]) {
                const cb = w.callback;
                w.callback = function () {
                    const out = cb?.apply(this, arguments);
                    editor.draw();
                    return out;
                };
            }
            this.setSize([Math.max(this.size[0], 360), this.computeSize()[1]]);
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            requestAnimationFrame(() => this.syntaxCurveEditor?.draw());
            return r;
        };
    },
});
