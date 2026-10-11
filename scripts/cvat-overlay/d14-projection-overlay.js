/*
 * Day 14 projection overlay: draws the job's 3D cuboids onto calibrated CVAT
 * context images (camera panel). Plain JS, no build step.
 *
 * Browser: loaded by cvat_ui via a <script> tag, registers through the official
 * window.cvatUI.registerComponent + core.plugins.register APIs. Geometry comes
 * from cvat-core's own ObjectStates (already interpolated for the frame), so the
 * overlay matches what the 3D view shows, including unsaved edits.
 * Node: require() returns the pure projection functions for parity tests.
 *
 * Projection mirrors cvat3d_fusion.py: R = Rx@Ry@Rz (three.js 'XYZ'), .pcd ->
 * ego (sensor2ego when pcd_frame is "lidar") -> camera (ego2cam), pinhole +
 * Brown-Conrady. Each edge is clipped to the projectable region (Z > NEAR_Z and
 * inside the cone where the lens polynomial is still monotonic, see radialFoldR2)
 * and drawn as a sampled curve, since distortion bends straight edges.
 */
(function (root) {
    'use strict';

    const NEAR_Z = 0.1;
    const EDGE_STEPS = 16;
    // corner order: 0-3 bottom (z-), 4-7 top (z+); 0,1,5,4 = front face (x+)
    const UNIT = [
        [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5],
    ];
    const EDGES = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]];
    const FRONT = [0, 1, 5, 4];
    const FRONT_EDGES = new Set(['0-1', '1-5', '5-4', '4-0', '1-0', '5-1', '4-5', '0-4']);

    function eulerXYZ(rx, ry, rz) {
        const [cx, sx, cy, sy, cz, sz] = [Math.cos(rx), Math.sin(rx), Math.cos(ry),
            Math.sin(ry), Math.cos(rz), Math.sin(rz)];
        // Rx @ Ry @ Rz expanded
        return [
            [cy * cz, -cy * sz, sy],
            [sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy],
            [-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy],
        ];
    }

    function boxCorners(position, rotation, scale) {
        const R = eulerXYZ(rotation[0], rotation[1], rotation[2]);
        return UNIT.map((u) => {
            const p = [u[0] * scale[0], u[1] * scale[1], u[2] * scale[2]];
            return [0, 1, 2].map((r) => R[r][0] * p[0] + R[r][1] * p[1] + R[r][2] * p[2] + position[r]);
        });
    }

    function transform(p, T) {
        return [0, 1, 2].map((r) => T[r][0] * p[0] + T[r][1] * p[1] + T[r][2] * p[2] + T[r][3]);
    }

    function lidarToCam(p, calib, camera) {
        const ego = calib.pcd_frame === 'ego' ? p : transform(p, calib.lidar.sensor2ego);
        return transform(ego, calib.cameras[camera].ego2cam);
    }

    function distCoeffs(cam) {
        const d = [0, 0, 0, 0, 0];
        (cam.dist || []).slice(0, 5).forEach((v, i) => { d[i] = Number(v); });
        return d;
    }

    // Largest undistorted r^2 where r_d = r (1 + k1 r^2 + k2 r^4 + k3 r^6) still grows.
    // Past it the fitted polynomial folds back and pulls off-image points into the
    // frame, so those points count as unprojectable. Same algorithm as
    // cvat3d_fusion.radial_fold_r2 (only + - * / sqrt, so both agree bit for bit).
    function radialFoldR2(dist) {
        const d = distCoeffs({ dist });
        const c = [1.0, 3.0 * d[0], 5.0 * d[1], 7.0 * d[4]];
        let deg = 3;
        while (deg > 0 && c[deg] === 0.0) deg -= 1;
        if (deg === 0) return Infinity;
        const g = (s) => c[0] + s * (c[1] + s * (c[2] + s * c[3]));
        let top = 0.0;
        for (let i = 0; i < deg; i += 1) top = Math.max(top, Math.abs(c[i]));
        const bound = 1.0 + top / Math.abs(c[deg]);
        let crit = [];
        const [qa, qb, qc] = [3.0 * c[3], 2.0 * c[2], c[1]];
        if (qa !== 0.0) {
            const disc = qb * qb - 4.0 * qa * qc;
            if (disc >= 0.0) {
                const sq = Math.sqrt(disc);
                crit = [(-qb - sq) / (2.0 * qa), (-qb + sq) / (2.0 * qa)];
            }
        } else if (qb !== 0.0) {
            crit = [-qc / qb];
        }
        const knots = [0.0, bound, ...crit.filter((s) => s > 0.0 && s < bound)].sort((x, y) => x - y);
        for (let k = 0; k + 1 < knots.length; k += 1) {
            let [lo, hi] = [knots[k], knots[k + 1]];
            if (g(hi) > 0.0) continue;
            for (let it = 0; it < 200; it += 1) {
                const mid = 0.5 * (lo + hi);
                if (mid <= lo || mid >= hi) break;
                if (g(mid) > 0.0) lo = mid; else hi = mid;
            }
            return lo;
        }
        return Infinity;
    }

    const foldCache = new WeakMap(); // camera calib object -> radialFoldR2
    function maxR2(cam) {
        if (!foldCache.has(cam)) foldCache.set(cam, radialFoldR2(cam.dist || []));
        return foldCache.get(cam);
    }

    function distortToPixel(xn, yn, cam) {
        const [k1, k2, p1, p2, k3] = distCoeffs(cam);
        const r2 = xn * xn + yn * yn;
        const radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
        const xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn);
        const yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn;
        return [cam.K[0][0] * xd + cam.K[0][2], cam.K[1][1] * yd + cam.K[1][2]];
    }

    function project(pc, cam) {
        const front = pc[2] > NEAR_Z;
        const z = front ? pc[2] : 1.0;
        const xn = pc[0] / z;
        const yn = pc[1] / z;
        const [u, v] = distortToPixel(xn, yn, cam);
        return { u, v, valid: front && xn * xn + yn * yn <= maxR2(cam) };
    }

    // [t0, t1] of edge a + t (b - a) inside Z > NEAR_Z and X^2 + Y^2 <= m Z^2. The
    // region is convex, so the result is a single interval or null.
    function clipEdgeView(a, b, m) {
        const [ax, ay, az] = a;
        const [bx, by, bz] = b;
        if (az <= NEAR_Z && bz <= NEAR_Z) return null;
        let [t0, t1] = [0.0, 1.0];
        if (az <= NEAR_Z || bz <= NEAR_Z) {
            const t = (NEAR_Z - az) / (bz - az);
            if (az > NEAR_Z) t1 = t; else t0 = t;
        }
        if (m === Infinity) return [t0, t1];
        const [dx, dy, dz] = [bx - ax, by - ay, bz - az];
        const qa = dx * dx + dy * dy - m * dz * dz;
        const qb = 2.0 * (ax * dx + ay * dy - m * az * dz);
        const qc = ax * ax + ay * ay - m * az * az;
        let roots = [];
        if (qa !== 0.0) {
            const disc = qb * qb - 4.0 * qa * qc;
            if (disc > 0.0) {
                const sq = Math.sqrt(disc);
                const h = qb >= 0.0 ? -0.5 * (qb + sq) : -0.5 * (qb - sq);
                roots = [h / qa, qc / h];
            }
        } else if (qb !== 0.0) {
            roots = [-qc / qb];
        }
        const knots = [t0, t1, ...roots.filter((r) => r > t0 && r < t1)].sort((x, y) => x - y);
        const inside = [];
        for (let k = 0; k + 1 < knots.length; k += 1) {
            const mid = 0.5 * (knots[k] + knots[k + 1]);
            if ((qa * mid + qb) * mid + qc <= 0.0) inside.push([knots[k], knots[k + 1]]);
        }
        if (!inside.length || inside[inside.length - 1][1] <= inside[0][0]) return null;
        return [inside[0][0], inside[inside.length - 1][1]];
    }

    // Projectable part of edge a-b (camera frame) as EDGE_STEPS + 1 image points, or null.
    function edgePolyline(a, b, cam) {
        const span = clipEdgeView(a, b, maxR2(cam));
        if (!span) return null;
        const [t0, t1] = span;
        const pts = [];
        for (let k = 0; k <= EDGE_STEPS; k += 1) {
            const t = t0 + (t1 - t0) * k / EDGE_STEPS;
            const p = [0, 1, 2].map((i) => a[i] + t * (b[i] - a[i]));
            pts.push(distortToPixel(p[0] / p[2], p[1] / p[2], cam));
        }
        return pts;
    }

    function colorFor(id) {
        // reduced mod 360 before multiplying and kept non-negative, so it equals Python's
        // int(id) * 2654435761 % 360 for every safe integer id (|id| < 2^53; the plugin passes
        // clientID, a small per-page counter). The plain product would pass 2^53.
        const h = id === null || id === undefined ? 0
            : ((((Math.trunc(Number(id)) % 360) * (2654435761 % 360)) % 360) + 360) % 360;
        const c = 255;
        const x = Math.trunc(255 * (1 - Math.abs(((h / 60) % 2) - 1)));
        const table = [[c, x, 0], [x, c, 0], [0, c, x], [0, x, c], [x, 0, c], [c, 0, x]];
        return table[Math.trunc(h / 60) % 6];
    }

    // box: {id, label, position, rotation, scale}. Returns null when out of view.
    function projectBox(box, calib, camera) {
        const cam = calib.cameras[camera];
        const [W, H] = cam.image_size;
        const corners = boxCorners(box.position, box.rotation, box.scale).map((p) => lidarToCam(p, calib, camera));
        const uv = corners.map((p) => project(p, cam));
        const valid = uv.filter((q) => q.valid);
        const inImage = valid.filter((q) => q.u >= 0 && q.u < W && q.v >= 0 && q.v < H).length;
        if (!inImage) return null;

        const edges = [];
        const drawn = valid.map((q) => [q.u, q.v]);
        for (const [i, j] of EDGES) {
            const pts = edgePolyline(corners[i], corners[j], cam);
            if (pts) {
                edges.push({ pts, front: FRONT_EDGES.has(`${i}-${j}`) });
                drawn.push(...pts);
            }
        }
        // convex projectable region: four valid front corners mean the whole face is inside
        const ring = FRONT.map((k, n) => edgePolyline(corners[k], corners[FRONT[(n + 1) % 4]], cam));
        const front = FRONT.every((k) => uv[k].valid) && ring.every(Boolean)
            ? ring.flatMap((pts) => pts.slice(0, -1)) : null;
        const clamp = (v, hi) => Math.min(Math.max(v, 0), hi);
        let [u0, v0, u1, v1] = [Infinity, Infinity, -Infinity, -Infinity];
        for (const [u, v] of drawn) {
            u0 = Math.min(u0, u); v0 = Math.min(v0, v); u1 = Math.max(u1, u); v1 = Math.max(v1, v);
        }
        const depth = corners.reduce((s, p) => s + p[2], 0) / 8;
        return {
            id: box.id,
            label: box.label,
            edges,
            front,
            bbox: [clamp(u0, W - 1), clamp(v0, H - 1), clamp(u1, W - 1), clamp(v1, H - 1)],
            depth,
            fullyVisible: inImage === 8,
        };
    }

    function drawProjected(ctx, items, width) {
        const base = Math.max(2, width / 480);
        ctx.save();
        ctx.lineJoin = 'round';
        ctx.font = `bold ${Math.round(width / 60)}px sans-serif`;
        ctx.textBaseline = 'bottom';
        for (const it of items) {
            const [r, g, b] = colorFor(it.id);
            if (it.front) {
                ctx.fillStyle = `rgba(${r},${g},${b},0.24)`;
                ctx.beginPath();
                it.front.forEach(([u, v], k) => (k ? ctx.lineTo(u, v) : ctx.moveTo(u, v)));
                ctx.closePath();
                ctx.fill();
            }
            ctx.strokeStyle = `rgb(${r},${g},${b})`;
            for (const e of it.edges) {
                ctx.lineWidth = e.front ? base * 1.5 : base;
                ctx.beginPath();
                e.pts.forEach(([u, v], k) => (k ? ctx.lineTo(u, v) : ctx.moveTo(u, v)));
                ctx.stroke();
            }
            const text = `#${it.id} ${it.label} ${Math.round(it.depth)}m`;
            const [x, y] = [it.bbox[0] + 3, Math.max(width / 60, it.bbox[1] - 3)];
            ctx.lineWidth = base * 1.5;
            ctx.strokeStyle = 'rgba(0,0,0,0.8)';
            ctx.strokeText(text, x, y);
            ctx.fillStyle = `rgb(${r},${g},${b})`;
            ctx.fillText(text, x, y);
        }
        ctx.restore();
    }

    const geometry = {
        NEAR_Z, EDGE_STEPS, EDGES, eulerXYZ, boxCorners, lidarToCam, radialFoldR2, project,
        clipEdgeView, edgePolyline, colorFor, projectBox,
    };
    if (typeof module === 'object' && module.exports) {
        module.exports = geometry;
        return;
    }

    // ------------------------------------------------------------------ browser
    const CONFIG_URL = '/d14-overlay/private/config.json';
    const LOG = '[d14-overlay]';
    const cleanBitmaps = new Map(); // `${jobId}:${frame}:${key}` -> core's own (unmodified) bitmap
    const CLEAN_LIMIT = 16;
    const jobMetas = new Map(); // job id -> Promise of /api/jobs/{id}/data/meta (null on failure)
    const META_LIMIT = 32;
    let configPromise = null;
    let last = { job: null, frame: null, states: null };
    let redrawQueued = false;

    function loadConfig() {
        if (!configPromise) {
            configPromise = fetch(CONFIG_URL, { credentials: 'same-origin', cache: 'no-store' })
                .then((r) => {
                    if (!r.ok) throw new Error(`config HTTP ${r.status}`);
                    return r.json();
                })
                .then((cfg) => {
                    const v = cfg.calibration_version || {};
                    const version = v.sha256 ? `${v.file} ${v.sha256.slice(0, 8)}` : 'unversioned';
                    console.info(LOG, `calibration ${version}, ${(cfg.frames || []).length} frames`);
                    return { ...cfg, version, frameSet: new Set(cfg.frames || []) };
                })
                .catch((e) => {
                    console.warn(LOG, 'disabled:', e.message);
                    configPromise = null; // retry on next use (e.g. after login)
                    return null;
                });
        }
        return configPromise;
    }

    // Insert or refresh `key` as the newest entry and drop the oldest ones beyond `limit`.
    function remember(map, key, value, limit) {
        map.delete(key);
        map.set(key, value);
        while (map.size > limit) map.delete(map.keys().next().value);
    }

    // The frame list only changes when the task's data is replaced; deleting frames marks
    // them in deleted_frames and keeps every entry of meta.frames, so caching per job is safe.
    function jobMeta(job) {
        let meta = jobMetas.get(job.id);
        if (!meta) {
            meta = fetch(`/api/jobs/${job.id}/data/meta`, { credentials: 'same-origin' })
                .then((r) => {
                    if (!r.ok) throw new Error(`job meta HTTP ${r.status}`);
                    return r.json();
                })
                .catch((e) => {
                    console.warn(LOG, `job ${job.id} skipped:`, e.message);
                    if (jobMetas.get(job.id) === meta) jobMetas.delete(job.id); // retry next time
                    return null;
                });
        }
        remember(jobMetas, job.id, meta, META_LIMIT);
        return meta;
    }

    // Point-cloud file of a job frame; cvat-core's FramesMetaData indexes job meta by
    // frame - job start. The server sets included_frames only for specific-frame segments
    // (ground-truth jobs), which index differently: not supported, the image stays as is.
    // A frame step (frame_filter "step=N") is refused too: CVAT v2.74.1 serves the context
    // images of data frame `number` instead of job frame `number`, so the camera image
    // belongs to another frame than the point cloud and any box drawn on it would be wrong.
    function frameStem(meta, job, frame) {
        if (meta.included_frames) return null;
        if (meta.frame_filter) {
            if (!meta.frameFilterWarned) console.warn(LOG, `job ${job.id} skipped: frame_filter ${meta.frame_filter}`);
            meta.frameFilterWarned = true;
            return null;
        }
        const item = meta.frames[frame - job.startFrame];
        return item ? item.name.replace(/^.*\//, '').replace(/\.[^.]*$/, '') : null;
    }

    // The calibration belongs to one recording, so draw only on frames whose point cloud is one
    // of its files, whatever the task is called or numbered on this CVAT.
    async function configFor(job, frame) {
        if (!job || job.dimension !== '3d') return null;
        const cfg = await loadConfig();
        if (!cfg) return null;
        const meta = await jobMeta(job);
        return meta && cfg.frameSet.has(frameStem(meta, job, frame)) ? cfg : null;
    }

    function boxesFromStates(states) {
        return (states || [])
            .filter((s) => s.shapeType === 'cuboid' && !s.outside && !s.hidden && s.points && s.points.length >= 9)
            .map((s) => ({
                id: s.clientID,
                label: s.label ? s.label.name : '',
                position: s.points.slice(0, 3),
                rotation: s.points.slice(3, 6),
                scale: s.points.slice(6, 9),
            }));
    }

    // Which calibration drew the boxes, so a screenshot names the file it came from.
    function drawVersion(ctx, cfg, width, height) {
        const size = Math.max(10, Math.round(width / 90));
        ctx.save();
        ctx.font = `${size}px monospace`;
        ctx.textBaseline = 'bottom';
        ctx.lineWidth = Math.max(2, size / 4);
        ctx.strokeStyle = 'rgba(0,0,0,0.8)';
        ctx.fillStyle = 'rgba(255,255,255,0.9)';
        ctx.strokeText(`calib ${cfg.version}`, size / 2, height - size / 2);
        ctx.fillText(`calib ${cfg.version}`, size / 2, height - size / 2);
        ctx.restore();
    }

    function paint(ctx, clean, states, cfg, camera) {
        ctx.drawImage(clean, 0, 0);
        const [W, H] = cfg.calibration.cameras[camera].image_size;
        if (clean.width !== W || clean.height !== H) {
            // wrong-size media must not get a silently rescaled overlay
            ctx.fillStyle = 'rgba(200,0,0,0.85)';
            ctx.font = `bold ${Math.round(clean.width / 40)}px sans-serif`;
            ctx.fillText(`overlay off: image ${clean.width}x${clean.height} != calib ${W}x${H}`, 10, clean.height / 10);
            drawVersion(ctx, cfg, clean.width, clean.height);
            return;
        }
        // far to near, so nearer boxes stay on top; the frame fetch and the same-frame
        // repaint receive states in different orders and must still paint identical pixels
        const items = boxesFromStates(states).map((b) => projectBox(b, cfg.calibration, camera)).filter(Boolean)
            .sort((a, b) => b.depth - a.depth || a.id - b.id);
        drawProjected(ctx, items, W);
        drawVersion(ctx, cfg, W, H);
    }

    // The frame fetch and the same-frame repaint both rasterise here: drawing straight onto
    // the panel canvas antialiases differently, so the two paths would not match pixel for pixel.
    function overlayBitmap(clean, states, cfg, camera) {
        const canvas = new OffscreenCanvas(clean.width, clean.height);
        paint(canvas.getContext('2d'), clean, states, cfg, camera);
        return createImageBitmap(canvas);
    }

    // Wrap Job.frames.contextImage: return copies with the overlay, never touch core's cached bitmaps.
    async function contextImageLeave(plugin, result, frameId) {
        try {
            const job = this;
            const cfg = await configFor(job, frameId);
            if (!cfg || !result) return result;
            const states = await job.annotations.get(frameId, false, []);
            const out = { ...result };
            for (const [key, camera] of Object.entries(cfg.cameras)) {
                const clean = result[key];
                if (!clean) continue;
                remember(cleanBitmaps, `${job.id}:${frameId}:${key}`, clean, CLEAN_LIMIT);
                out[key] = await overlayBitmap(clean, states, cfg, camera);
            }
            return out;
        } catch (e) {
            console.error(LOG, 'contextImage overlay failed, showing original image', e);
            return result;
        }
    }

    // Same-frame edits do not refetch context images, so repaint the panel canvases directly.
    // Runs on the next animation frame, i.e. only while the tab is visible; a hidden tab
    // repaints with the latest state once it is shown again.
    async function redrawPanels() {
        redrawQueued = false;
        const snapshot = last;
        const { job, frame, states } = snapshot;
        const cfg = await configFor(job, frame);
        if (!cfg) return;
        for (const wrapper of document.querySelectorAll('.cvat-context-image-wrapper')) {
            const title = wrapper.querySelector('.cvat-context-image-title');
            const canvas = wrapper.querySelector('canvas');
            const key = title && title.textContent.trim();
            const camera = key && cfg.cameras[key];
            const clean = camera && cleanBitmaps.get(`${job.id}:${frame}:${key}`);
            if (!canvas || !clean || canvas.width !== clean.width || canvas.height !== clean.height) continue;
            try {
                const bitmap = await overlayBitmap(clean, states, cfg, camera);
                // newer state, frame or camera arrived meanwhile: that change queued its own redraw
                if (last === snapshot && title.textContent.trim() === key) canvas.getContext('2d').drawImage(bitmap, 0, 0);
                bitmap.close();
            } catch (e) {
                console.error(LOG, 'panel repaint failed', e);
            }
        }
    }

    function onStateUpdate(state) {
        const job = state.annotation && state.annotation.job.instance;
        if (!job || job.dimension !== '3d') return;
        const frame = state.annotation.player.frame.number;
        const { states } = state.annotation.annotations;
        if (job === last.job && frame === last.frame && states === last.states) return;
        last = { job, frame, states };
        if (!redrawQueued) {
            redrawQueued = true;
            requestAnimationFrame(redrawPanels);
        }
    }

    const plugin = {
        name: 'Day14 projection overlay',
        description: 'Projects 3D cuboids onto calibrated context images',
        cvat: { classes: { Job: { prototype: { frames: { contextImage: { leave: contextImageLeave } } } } } },
    };

    function builder({ core }) {
        core.plugins.register(plugin);
        console.info(LOG, 'registered');
        return {
            name: plugin.name,
            destructor() { cleanBitmaps.clear(); jobMetas.clear(); },
            globalStateDidUpdate: onStateUpdate,
        };
    }

    let registered = false;
    function register() {
        if (registered || !Object.prototype.hasOwnProperty.call(root, 'cvatUI')) return;
        registered = true;
        root.cvatUI.registerComponent(builder);
    }
    root.addEventListener('plugins.ready', register, { once: true });
    register(); // script may run after plugins.ready already fired
}(typeof window !== 'undefined' ? window : globalThis));
