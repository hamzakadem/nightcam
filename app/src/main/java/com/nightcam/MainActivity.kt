package com.nightcam

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.util.Size
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var status: TextView
    private lateinit var zoomLabel: TextView
    private lateinit var shutter: Button
    private var imageCapture: ImageCapture? = null
    private val exec = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())
    private var zoom = 1f
    private var busy = false

    private val permLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
            if (ok) startCamera() else status.text = "الكاميرا تحتاج إلى إذن"
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
            startCamera()
        else permLauncher.launch(Manifest.permission.CAMERA)
    }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private fun buildUi() {
        val mp = ViewGroup.LayoutParams.MATCH_PARENT
        val wc = ViewGroup.LayoutParams.WRAP_CONTENT
        val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
        previewView = PreviewView(this).apply { scaleType = PreviewView.ScaleType.FIT_CENTER }
        root.addView(previewView, FrameLayout.LayoutParams(mp, mp))

        val panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(dp(16), dp(12), dp(16), dp(24))
            setBackgroundColor(0x99000000.toInt())
        }
        status = TextView(this).apply { setTextColor(Color.WHITE); text = "جاهز — ثبّت الهاتف ثم اضغط التقاط"; gravity = Gravity.CENTER }
        zoomLabel = TextView(this).apply { setTextColor(Color.WHITE); text = "زووم ×1.0"; gravity = Gravity.CENTER }
        val seek = SeekBar(this).apply {
            max = 30
            progress = 0
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                    zoom = 1f + p / 10f
                    zoomLabel.text = "زووم ×%.1f".format(zoom)
                    previewView.scaleX = zoom
                    previewView.scaleY = zoom
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {}
            })
        }
        shutter = Button(this).apply { text = "التقاط ليلي"; setOnClickListener { shoot() } }
        panel.addView(status, LinearLayout.LayoutParams(mp, wc))
        panel.addView(zoomLabel, LinearLayout.LayoutParams(mp, wc))
        panel.addView(seek, LinearLayout.LayoutParams(mp, wc))
        panel.addView(shutter, LinearLayout.LayoutParams(wc, wc))
        root.addView(panel, FrameLayout.LayoutParams(mp, wc, Gravity.BOTTOM))
        setContentView(root)
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
            val rs = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(Size(3264, 2448), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                ).build()
            val ic = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setResolutionSelector(rs)
                .setJpegQuality(97)
                .build()
            provider.unbindAll()
            val cam = provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, ic)
            val range = cam.cameraInfo.exposureState.exposureCompensationRange
            cam.cameraControl.setExposureCompensationIndex((range.upper * 0.5f).toInt())
            imageCapture = ic
        }, ContextCompat.getMainExecutor(this))
    }

    private fun shoot() {
        val ic = imageCapture ?: return
        if (busy) return
        busy = true
        shutter.isEnabled = false
        val z = zoom
        val n = if (z >= 2f) 10 else 8
        val frames = ArrayList<ByteArray>()
        var rot = 0

        fun next() {
            if (frames.size >= n) {
                exec.execute { runPipeline(frames, rot, z) }
                return
            }
            ic.takePicture(exec, object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    val buf = image.planes[0].buffer
                    val bytes = ByteArray(buf.remaining())
                    buf.get(bytes)
                    rot = image.imageInfo.rotationDegrees
                    image.close()
                    frames.add(bytes)
                    runOnUiThread { status.text = "ثبّت الهاتف… ${frames.size}/$n" }
                    main.post { next() }
                }

                override fun onError(exception: ImageCaptureException) {
                    runOnUiThread {
                        status.text = "فشل الالتقاط: ${exception.message}"
                        busy = false
                        shutter.isEnabled = true
                    }
                }
            })
        }
        next()
    }

    private fun runPipeline(frames: List<ByteArray>, rot: Int, z: Float) {
        try {
            val res = NightEngine().process(frames, z) { msg -> runOnUiThread { status.text = msg } }
            val bmp = Bitmap.createBitmap(res.pixels, res.w, res.h, Bitmap.Config.ARGB_8888)
            val m = Matrix().apply { postRotate(rot.toFloat()) }
            val outBmp = Bitmap.createBitmap(bmp, 0, 0, res.w, res.h, m, true)

            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, "NIGHT_${System.currentTimeMillis()}.jpg")
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/NightCam")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
            val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)!!
            contentResolver.openOutputStream(uri)!!.use { outBmp.compress(Bitmap.CompressFormat.JPEG, 95, it) }
            values.clear()
            values.put(MediaStore.Images.Media.IS_PENDING, 0)
            contentResolver.update(uri, values, null, null)
            runOnUiThread { status.text = "تم الحفظ في Pictures/NightCam (${outBmp.width}×${outBmp.height})" }
        } catch (t: Throwable) {
            runOnUiThread { status.text = "خطأ: ${t.javaClass.simpleName} ${t.message}" }
        } finally {
            runOnUiThread { busy = false; shutter.isEnabled = true }
        }
    }
}
