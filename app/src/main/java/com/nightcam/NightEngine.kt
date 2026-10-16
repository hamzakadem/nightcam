package com.nightcam

import android.graphics.BitmapFactory
import android.graphics.BitmapRegionDecoder
import android.graphics.Rect
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

private const val TILE = 32

/**
 * NightEngine — خوارزمية "NSR" (Night Shift-and-add Reconstruction)
 *
 * 1) اختيار أوضح إطار كمرجع (تباين لابلاس).
 * 2) محاذاة دون-بكسلية: بحث SAD خشن على 1/4 ثم SSD دقيق + تقطيع قطعي مكافئ (parabolic).
 * 2b) ATWD — Adaptive Temporal-Wiener Denoise (خوارزمية إزالة الضوضاء متعددة الإطارات):
 *     أ) ملف ضوضاء يعتمد على السطوع σ(Y) يُقاس من الإطار المرجع (الضوضاء في الظلال ≠ الإضاءات).
 *     ب) تصحيح حركة محلي لكل مربع 32×32 (±2 بكسل) فوق المحاذاة العامة — يعالج حركة الأجسام والاهتزاز غير المنتظم.
 *     ج) تجميع تباين زمني لكل بكسل: إن زاد التباين عن المتوقع من الضوضاء فقط فهذا حركة، فنمزج نحو المرجع.
 *     د) تنعيم مكاني ثنائي (bilateral) شدّته تتبع N_eff (عدد الإطارات الفعلي المدموجة) —
 *        المناطق التي دُمج فيها إطارات أقل تُنعَّم أكثر، والنظيفة تُترك كما هي.
 * 3) دمج مقاوم للأشباح: وزن غاوسي حسب فرق السطوع عن المرجع، وعتبته تُشتق من تقدير ضوضاء الإطار.
 * 4) عند الزووم >= 2: Shift-and-Add — كل إطار يُرمى بكسله على شبكة 2x بموضعه الحقيقي دون-البكسلي
 *    (اهتزاز اليد يعطي عيّنات جديدة = تفاصيل حقيقية لا مجرد تكبير).
 * 5) Tone mapping ليلي محلي: رفع الظلال بكسب يتبع السطوع المحلي + ضغط الإضاءات (Reinhard الموسّع)
 *    + تنعيم الصبغة في المناطق المظلمة فقط.
 * 6) Unsharp مع coring: يشحذ الحواف الحقيقية ولا يضخّم الضوضاء المتبقية.
 */
class NightEngine {

    class Result(val pixels: IntArray, val w: Int, val h: Int)

    fun process(jpegs: List<ByteArray>, zoom: Float, log: (String) -> Unit): Result {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(jpegs[0], 0, jpegs[0].size, bounds)
        val fw = bounds.outWidth
        val fh = bounds.outHeight
        val z = max(1f, zoom)
        val cw = (fw / z).toInt() and 3.inv()
        val ch = (fh / z).toInt() and 3.inv()
        val left = (fw - cw) / 2
        val top = (fh - ch) / 2
        val rect = Rect(left, top, left + cw, top + ch)

        log("اختيار أوضح إطار…")
        var refIdx = 0
        var bestSharp = -1.0
        for (i in jpegs.indices) {
            val s = sharpness(jpegs[i])
            if (s > bestSharp) { bestSharp = s; refIdx = i }
        }

        val ref = decodeCrop(jpegs[refIdx], rect)
        val refG = gray(ref)
        val refS = down4(refG, cw, ch)
        val sigma = noiseSigma(refG, cw, ch)
        val lut = noiseProfile(refG, cw, ch, sigma)

        val s = if (z >= 2f) 2 else 1
        val ow = cw * s
        val oh = ch * s
        val sum = FloatArray(ow * oh * 3)
        val sumY2 = FloatArray(ow * oh)
        val wt = FloatArray(ow * oh)

        splat(ref, refG, cw, ch, 0f, 0f, null, refG, true, lut, s, sum, sumY2, wt, ow, oh)

        var done = 1
        for (i in jpegs.indices) {
            if (i == refIdx) continue
            done++
            log("محاذاة ودمج $done/${jpegs.size}…")
            val fr = decodeCrop(jpegs[i], rect)
            if (fr.size != ref.size) continue
            val fg = gray(fr)
            val fs = down4(fg, cw, ch)
            val sh = estimateShift(refG, fg, refS, fs, cw, ch)
            val ts = localShifts(refG, fg, cw, ch, sh[0], sh[1])
            splat(fr, fg, cw, ch, sh[0], sh[1], ts, refG, false, lut, s, sum, sumY2, wt, ow, oh)
        }

        log("إزالة الضوضاء الزمنية-المكانية…")
        val out = IntArray(ow * oh)
        for (y in 0 until oh) {
            for (x in 0 until ow) {
                val k = y * ow + x
                val wk = wt[k]
                if (wk > 1e-3f) {
                    val inv = 1f / wk
                    var r = sum[k * 3] * inv
                    var g = sum[k * 3 + 1] * inv
                    var b = sum[k * 3 + 2] * inv
                    if (s == 1) {
                        val my = (r * 77f + g * 150f + b * 29f) / 256f
                        val vr = sumY2[k] * inv - my * my
                        val sg = lut[my.toInt().coerceIn(0, 255)]
                        val rho = vr / (sg * sg + 1f)
                        val a = ((rho - 2f) / 4f).coerceIn(0f, 1f)
                        if (a > 0f) {
                            val c = ref[k]
                            r += (((c shr 16) and 255) - r) * a
                            g += (((c shr 8) and 255) - g) * a
                            b += ((c and 255) - b) * a
                        }
                    }
                    val ri = (r + 0.5f).toInt().coerceIn(0, 255)
                    val gi = (g + 0.5f).toInt().coerceIn(0, 255)
                    val bi = (b + 0.5f).toInt().coerceIn(0, 255)
                    out[k] = (255 shl 24) or (ri shl 16) or (gi shl 8) or bi
                } else {
                    out[k] = ref[min(y / s, ch - 1) * cw + min(x / s, cw - 1)]
                }
            }
        }
        spatialDenoise(out, wt, s, lut, ow, oh)

        log("معالجة الإضاءة الليلية…")
        val toned = tone(out, ow, oh)
        log("استرجاع التفاصيل…")
        sharpen(toned, ow, oh, if (s == 2) 1.3f else 0.9f, sigma)
        return Result(toned, ow, oh)
    }

    // ---------- decoding / helpers ----------

    private fun decodeCrop(b: ByteArray, r: Rect): IntArray {
        val dec = BitmapRegionDecoder.newInstance(b, 0, b.size)
        val opt = BitmapFactory.Options().apply { inPreferredConfig = android.graphics.Bitmap.Config.ARGB_8888 }
        val bmp = dec.decodeRegion(r, opt)
        dec.recycle()
        val px = IntArray(bmp.width * bmp.height)
        bmp.getPixels(px, 0, bmp.width, 0, 0, bmp.width, bmp.height)
        bmp.recycle()
        return px
    }

    private fun lum(c: Int): Int =
        ((((c shr 16) and 255) * 77) + (((c shr 8) and 255) * 150) + ((c and 255) * 29)) shr 8

    private fun gray(px: IntArray): ByteArray = ByteArray(px.size) { lum(px[it]).toByte() }

    private fun down4(g: ByteArray, w: Int, h: Int): ByteArray {
        val sw = w / 4
        val sh = h / 4
        val o = ByteArray(sw * sh)
        for (y in 0 until sh) for (x in 0 until sw) {
            var s = 0
            for (j in 0..3) {
                val r = (y * 4 + j) * w + x * 4
                for (i in 0..3) s += g[r + i].toInt() and 255
            }
            o[y * sw + x] = (s shr 4).toByte()
        }
        return o
    }

    private fun sharpness(b: ByteArray): Double {
        val o = BitmapFactory.Options().apply { inSampleSize = 4 }
        val bmp = BitmapFactory.decodeByteArray(b, 0, b.size, o) ?: return 0.0
        val w = bmp.width
        val h = bmp.height
        val px = IntArray(w * h)
        bmp.getPixels(px, 0, w, 0, 0, w, h)
        bmp.recycle()
        val g = gray(px)
        fun gv(i: Int) = g[i].toInt() and 255
        var s = 0.0
        var s2 = 0.0
        var n = 0
        for (y in 1 until h - 1 step 2) for (x in 1 until w - 1 step 2) {
            val i = y * w + x
            val l = (4 * gv(i) - gv(i - 1) - gv(i + 1) - gv(i - w) - gv(i + w)).toDouble()
            s += l; s2 += l * l; n++
        }
        if (n == 0) return 0.0
        val m = s / n
        return s2 / n - m * m
    }

    private fun noiseSigma(g: ByteArray, w: Int, h: Int): Float {
        var acc = 0L
        var n = 0L
        for (y in 1 until h - 1 step 4) for (x in 1 until w - 1 step 4) {
            val i = y * w + x
            val l = 4 * (g[i].toInt() and 255) - (g[i - 1].toInt() and 255) - (g[i + 1].toInt() and 255) -
                (g[i - w].toInt() and 255) - (g[i + w].toInt() and 255)
            acc += abs(l); n++
        }
        return if (n == 0L) 5f else (acc.toFloat() / n) / 3.57f
    }

    // ---------- alignment ----------

    /** يُرجع (dx,dy) بحيث frame(x+dx, y+dy) ≈ ref(x,y) */
    private fun estimateShift(rg: ByteArray, fg: ByteArray, rs: ByteArray, fs: ByteArray, w: Int, h: Int): FloatArray {
        val sw = w / 4
        val sh = h / 4
        val rad = 10
        var bx = 0
        var by = 0
        var best = Long.MAX_VALUE
        for (dy in -rad..rad) for (dx in -rad..rad) {
            var acc = 0L
            var y = rad
            while (y < sh - rad) {
                val rowR = y * sw
                val rowF = (y + dy) * sw + dx
                var x = rad
                while (x < sw - rad) {
                    acc += abs((rs[rowR + x].toInt() and 255) - (fs[rowF + x].toInt() and 255))
                    x += 2
                }
                y += 2
            }
            if (acc < best) { best = acc; bx = dx; by = dy }
        }
        val cx = bx * 4
        val cy = by * 4
        val m = max(abs(cx), abs(cy)) + 8
        val ss = DoubleArray(81)
        var minV = Double.MAX_VALUE
        var mi = 40
        for (j in -4..4) for (i in -4..4) {
            var acc = 0L
            var y = m
            while (y < h - m) {
                val rr = y * w
                val ff = (y + cy + j) * w + cx + i
                var x = m
                while (x < w - m) {
                    val d = (rg[rr + x].toInt() and 255) - (fg[ff + x].toInt() and 255)
                    acc += d * d
                    x += 4
                }
                y += 4
            }
            val idx = (j + 4) * 9 + (i + 4)
            ss[idx] = acc.toDouble()
            if (ss[idx] < minV) { minV = ss[idx]; mi = idx }
        }
        val mx = mi % 9
        val my = mi / 9
        fun sub(a: Double, b: Double, c: Double): Float {
            val den = a - 2 * b + c
            return if (den <= 1e-9) 0f else (0.5 * (a - c) / den).toFloat().coerceIn(-0.5f, 0.5f)
        }
        var sx = 0f
        var sy = 0f
        if (mx in 1..7) sx = sub(ss[my * 9 + mx - 1], ss[mi], ss[my * 9 + mx + 1])
        if (my in 1..7) sy = sub(ss[(my - 1) * 9 + mx], ss[mi], ss[(my + 1) * 9 + mx])
        return floatArrayOf(cx + (mx - 4) + sx, cy + (my - 4) + sy)
    }

    // ---------- merge / super-resolution ----------

    private fun splat(
        px: IntArray, g: ByteArray, w: Int, h: Int, dx: Float, dy: Float, ts: FloatArray?,
        refG: ByteArray, isRef: Boolean, lut: FloatArray, s: Int,
        sum: FloatArray, sumY2: FloatArray, wt: FloatArray, ow: Int, oh: Int
    ) {
        val tilesX = (w + TILE - 1) / TILE
        for (v in 0 until h) {
            val rowTile = (v / TILE) * tilesX
            for (u in 0 until w) {
                var sx = dx
                var sy = dy
                if (ts != null) {
                    val t = (rowTile + u / TILE) * 2
                    sx = ts[t]; sy = ts[t + 1]
                }
                val pxx = u - sx
                val py = v - sy
                if (pxx < 0f || pxx > w - 1f || py < 0f || py > h - 1f) continue
                var wgt = 1f
                if (!isRef) {
                    val x0 = pxx.toInt(); val x1 = min(x0 + 1, w - 1); val fx = pxx - x0
                    val y0 = py.toInt(); val y1 = min(y0 + 1, h - 1); val fy = py - y0
                    val rv = (refG[y0 * w + x0].toInt() and 255) * (1 - fx) * (1 - fy) +
                        (refG[y0 * w + x1].toInt() and 255) * fx * (1 - fy) +
                        (refG[y1 * w + x0].toInt() and 255) * (1 - fx) * fy +
                        (refG[y1 * w + x1].toInt() and 255) * fx * fy
                    val thr = max(6f, 3f * lut[rv.toInt().coerceIn(0, 255)])
                    val d = rv - (g[v * w + u].toInt() and 255)
                    wgt = exp(-d * d / (2f * thr * thr))
                    if (wgt < 0.02f) continue
                }
                val c = px[v * w + u]
                val r = ((c shr 16) and 255).toFloat()
                val gg = ((c shr 8) and 255).toFloat()
                val b = (c and 255).toFloat()
                val yy = (r * 77f + gg * 150f + b * 29f) / 256f
                val ox = (pxx + 0.5f) * s - 0.5f
                val oy = (py + 0.5f) * s - 0.5f
                val cx0 = floor(ox).toInt(); val gx = ox - cx0
                val cy0 = floor(oy).toInt(); val gy = oy - cy0
                for (j in 0..1) {
                    val cy = cy0 + j
                    if (cy < 0 || cy >= oh) continue
                    val wy = if (j == 0) 1 - gy else gy
                    for (i in 0..1) {
                        val cx = cx0 + i
                        if (cx < 0 || cx >= ow) continue
                        val ww = wgt * wy * (if (i == 0) 1 - gx else gx)
                        val k = cy * ow + cx
                        sum[k * 3] += r * ww
                        sum[k * 3 + 1] += gg * ww
                        sum[k * 3 + 2] += b * ww
                        sumY2[k] += yy * yy * ww
                        wt[k] += ww
                    }
                }
            }
        }
    }

    // ---------- ATWD: noise profile, local motion, adaptive spatial stage ----------

    /** σ(Y): انحراف الضوضاء المقدَّر لكل مستوى سطوع 0..255 من الإطار المرجع */
    private fun noiseProfile(g: ByteArray, w: Int, h: Int, global: Float): FloatArray {
        val sum = DoubleArray(16)
        val cnt = IntArray(16)
        for (y in 1 until h - 1 step 3) for (x in 1 until w - 1 step 3) {
            val i = y * w + x
            val c = g[i].toInt() and 255
            val l = abs(
                4 * c - (g[i - 1].toInt() and 255) - (g[i + 1].toInt() and 255) -
                    (g[i - w].toInt() and 255) - (g[i + w].toInt() and 255)
            )
            if (l > 60) continue // تجاهل الحواف القوية
            sum[c shr 4] += l.toDouble(); cnt[c shr 4]++
        }
        val bins = FloatArray(16) { if (cnt[it] > 200) (sum[it] / cnt[it] / 3.57).toFloat() else global }
        val sm = FloatArray(16) {
            (bins[max(it - 1, 0)] + 2f * bins[it] + bins[min(it + 1, 15)]) / 4f
        }
        return FloatArray(256) { sm[it shr 4].coerceIn(1.5f, 40f) }
    }

    /** تصحيح حركة محلي: لكل مربع TILE×TILE يُرجع (dx,dy) الكلي (عام + بقايا صحيحة ±2) */
    private fun localShifts(rg: ByteArray, fg: ByteArray, w: Int, h: Int, gdx: Float, gdy: Float): FloatArray {
        val tilesX = (w + TILE - 1) / TILE
        val tilesY = (h + TILE - 1) / TILE
        val out = FloatArray(tilesX * tilesY * 2)
        val bx = Math.round(gdx)
        val by = Math.round(gdy)
        val rad = 2
        for (ty in 0 until tilesY) for (tx in 0 until tilesX) {
            val x0 = tx * TILE; val y0 = ty * TILE
            val x1 = min(x0 + TILE, w); val y1 = min(y0 + TILE, h)
            var rx = 0
            var ry = 0
            val inside = x0 + bx - rad >= 0 && x1 - 1 + bx + rad <= w - 1 &&
                y0 + by - rad >= 0 && y1 - 1 + by + rad <= h - 1
            if (inside) {
                var best = Long.MAX_VALUE
                var sad0 = 0L
                for (j in -rad..rad) for (i in -rad..rad) {
                    var acc = 0L
                    var y = y0
                    while (y < y1) {
                        val rr = y * w
                        val ff = (y + by + j) * w + bx + i
                        var x = x0
                        while (x < x1) {
                            acc += abs((rg[rr + x].toInt() and 255) - (fg[ff + x].toInt() and 255))
                            x += 2
                        }
                        y += 2
                    }
                    if (i == 0 && j == 0) sad0 = acc
                    if (acc < best) { best = acc; rx = i; ry = j }
                }
                if (best.toFloat() > 0.93f * sad0) { rx = 0; ry = 0 } // لا تغيّر إلا عند تحسّن واضح
            }
            val t = (ty * tilesX + tx) * 2
            out[t] = gdx + rx
            out[t + 1] = gdy + ry
        }
        return out
    }

    /** تنعيم ثنائي 5×5 شدّته تتبع الضوضاء المتبقية σ(Y)/√N_eff — حافظ للحواف */
    private fun spatialDenoise(px: IntArray, wt: FloatArray, s: Int, lut: FloatArray, w: Int, h: Int) {
        val yv = ByteArray(px.size) { lum(px[it]).toByte() }
        val yb = boxBlur(yv, w, h, 1)
        val src = px.copyOf()
        val rad = 2
        val sp = FloatArray(25)
        for (j in -rad..rad) for (i in -rad..rad)
            sp[(j + rad) * 5 + i + rad] = exp(-(i * i + j * j) / (2f * 1.5f * 1.5f))
        val s2 = (s * s).toFloat()
        for (y in rad until h - rad) {
            for (x in rad until w - rad) {
                val i0 = y * w + x
                val yc = yb[i0].toInt() and 255
                val neff = max(1f, wt[i0] * s2)
                val sres = lut[yc] / sqrt(neff)
                if (sres < 0.6f) continue
                val sr = 2.2f * sres + 1f
                val inv2 = 1f / (2f * sr * sr)
                var rs = 0f; var gs = 0f; var bs = 0f; var ws = 0f
                for (j in -rad..rad) {
                    val row = (y + j) * w + x
                    for (i in -rad..rad) {
                        val q = row + i
                        val d = ((yb[q].toInt() and 255) - yc).toFloat()
                        val wq = sp[(j + rad) * 5 + i + rad] * exp(-d * d * inv2)
                        val c = src[q]
                        rs += ((c shr 16) and 255) * wq
                        gs += ((c shr 8) and 255) * wq
                        bs += (c and 255) * wq
                        ws += wq
                    }
                }
                val iv = 1f / ws
                val r = (rs * iv + 0.5f).toInt().coerceIn(0, 255)
                val g = (gs * iv + 0.5f).toInt().coerceIn(0, 255)
                val b = (bs * iv + 0.5f).toInt().coerceIn(0, 255)
                px[i0] = (255 shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
    }

    // ---------- night tone mapping ----------

    private fun tone(px: IntArray, w: Int, h: Int): IntArray {
        val hist = IntArray(256)
        var n = 0
        var i = 0
        while (i < px.size) { hist[lum(px[i])]++; n++; i += 7 }
        var acc = 0
        var bp = 0
        for (v in 0..255) { acc += hist[v]; if (acc >= n * 0.005f) { bp = v; break } }
        val bpf = min(bp / 255f, 0.06f)
        val inv = 1f / (1f - bpf)
        val lin = FloatArray(256) { v -> (((v / 255f) - bpf) * inv).coerceIn(0f, 1f).pow(2.2f) }
        val enc = IntArray(4097) { (((it / 4096f).pow(1f / 2.2f)) * 255f + 0.5f).toInt().coerceIn(0, 255) }

        val lw = w / 4
        val lh = h / 4
        val lr = FloatArray(lw * lh)
        val lg = FloatArray(lw * lh)
        val lb = FloatArray(lw * lh)
        val ly = FloatArray(lw * lh)
        for (y in 0 until lh) for (x in 0 until lw) {
            var r = 0f; var g = 0f; var b = 0f
            for (j in 0..3) for (k in 0..3) {
                val c = px[(y * 4 + j) * w + x * 4 + k]
                r += lin[(c shr 16) and 255]; g += lin[(c shr 8) and 255]; b += lin[c and 255]
            }
            val o = y * lw + x
            lr[o] = r / 16f; lg[o] = g / 16f; lb[o] = b / 16f
            ly[o] = 0.2126f * lr[o] + 0.7152f * lg[o] + 0.0722f * lb[o]
        }

        val out = IntArray(px.size)
        for (y in 0 until h) {
            val fy = ((y + 0.5f) / 4f - 0.5f).coerceIn(0f, lh - 1f)
            val y0 = fy.toInt(); val y1 = min(y0 + 1, lh - 1); val ty = fy - y0
            for (x in 0 until w) {
                val fx = ((x + 0.5f) / 4f - 0.5f).coerceIn(0f, lw - 1f)
                val x0 = fx.toInt(); val x1 = min(x0 + 1, lw - 1); val tx = fx - x0
                val i00 = y0 * lw + x0; val i10 = y0 * lw + x1
                val i01 = y1 * lw + x0; val i11 = y1 * lw + x1
                val w00 = (1 - tx) * (1 - ty); val w10 = tx * (1 - ty)
                val w01 = (1 - tx) * ty; val w11 = tx * ty
                val lbase = ly[i00] * w00 + ly[i10] * w10 + ly[i01] * w01 + ly[i11] * w11

                val c = px[y * w + x]
                var r = lin[(c shr 16) and 255]
                var g = lin[(c shr 8) and 255]
                var b = lin[c and 255]
                val yy = 0.2126f * r + 0.7152f * g + 0.0722f * b

                // تنعيم الصبغة في الظلال فقط
                val a = (1f - yy * 20f).coerceIn(0f, 1f) * 0.75f
                if (a > 0f) {
                    val cr = lr[i00] * w00 + lr[i10] * w10 + lr[i01] * w01 + lr[i11] * w11 - lbase
                    val cg = lg[i00] * w00 + lg[i10] * w10 + lg[i01] * w01 + lg[i11] * w11 - lbase
                    val cb = lb[i00] * w00 + lb[i10] * w10 + lb[i01] * w01 + lb[i11] * w11 - lbase
                    r = yy + (1 - a) * (r - yy) + a * cr
                    g = yy + (1 - a) * (g - yy) + a * cg
                    b = yy + (1 - a) * (b - yy) + a * cb
                }

                val gain = (0.18f / (lbase + 0.004f)).pow(0.5f).coerceIn(1f, 5f)
                val y2 = yy * gain
                val ratio = gain * (1f + y2 * (1f / 9f)) / (1f + y2) * 1.2f
                val yo = yy * ratio
                val sat = 1.1f
                val ro = yo + sat * (r * ratio - yo)
                val go = yo + sat * (g * ratio - yo)
                val bo = yo + sat * (b * ratio - yo)
                val er = enc[(ro.coerceIn(0f, 1f) * 4096f).toInt()]
                val eg = enc[(go.coerceIn(0f, 1f) * 4096f).toInt()]
                val eb = enc[(bo.coerceIn(0f, 1f) * 4096f).toInt()]
                out[y * w + x] = (255 shl 24) or (er shl 16) or (eg shl 8) or eb
            }
        }
        return out
    }

    // ---------- detail recovery ----------

    private fun sharpen(px: IntArray, w: Int, h: Int, amt: Float, sigma: Float) {
        val yv = ByteArray(px.size) { lum(px[it]).toByte() }
        val bl = boxBlur(yv, w, h, 2)
        val t = max(3f, sigma * 1.5f)
        for (i in px.indices) {
            val d = (yv[i].toInt() and 255) - (bl[i].toInt() and 255)
            if (d == 0) continue
            val df = d.toFloat()
            val f = df * df / (df * df + t * t)
            val delta = (amt * df * f).coerceIn(-30f, 30f).toInt()
            if (delta == 0) continue
            val c = px[i]
            val r = (((c shr 16) and 255) + delta).coerceIn(0, 255)
            val g = (((c shr 8) and 255) + delta).coerceIn(0, 255)
            val b = ((c and 255) + delta).coerceIn(0, 255)
            px[i] = (255 shl 24) or (r shl 16) or (g shl 8) or b
        }
    }

    private fun boxBlur(src: ByteArray, w: Int, h: Int, r: Int): ByteArray {
        val k = 2 * r + 1
        val tmp = ByteArray(src.size)
        for (y in 0 until h) {
            val row = y * w
            var s = 0
            for (i in -r..r) s += src[row + i.coerceIn(0, w - 1)].toInt() and 255
            for (x in 0 until w) {
                tmp[row + x] = (s / k).toByte()
                val add = min(x + r + 1, w - 1)
                val sb = max(x - r, 0)
                s += (src[row + add].toInt() and 255) - (src[row + sb].toInt() and 255)
            }
        }
        val out = ByteArray(src.size)
        val col = IntArray(w)
        for (i in -r..r) {
            val row = i.coerceIn(0, h - 1) * w
            for (x in 0 until w) col[x] += tmp[row + x].toInt() and 255
        }
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) out[row + x] = (col[x] / k).toByte()
            val ar = min(y + r + 1, h - 1) * w
            val sr = max(y - r, 0) * w
            for (x in 0 until w) col[x] += (tmp[ar + x].toInt() and 255) - (tmp[sr + x].toInt() and 255)
        }
        return out
    }
}
