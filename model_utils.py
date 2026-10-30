"""
Model + Grad-CAM++ utilities for ULCARE_V3.keras.

====================================================================================
 THINGS YOU MUST CHECK BEFORE TRUSTING ANY PREDICTION FROM THIS FILE
====================================================================================
I don't have access to your Colab training notebook, so three things below are
best-guess defaults, not facts I verified against your actual training code. Open
your notebook and confirm each one; if any differs, edit the constants at the top
of this file (search "CONFIG — VERIFY THIS").

1. PREPROCESS_MODE — how pixel values were scaled before going into the model
   during training (plain /255 rescale? tf.keras.applications.mobilenet_v2's
   preprocess_input which maps to [-1, 1]? something else?). Get this wrong and
   the model will still happily output a confident-looking prediction — just a
   wrong one. There's no error message for this, so you have to check by hand.

2. CONV_LAYER_NAME — which layer Grad-CAM++ reads activations from. Left as
   None, the code auto-picks the last Conv2D layer it can find (recursing into
   nested backbones like MobileNetV2/EfficientNet). That's usually right, but
   if your model has an unusual branching structure, set this explicitly to the
   layer name you want (check with `model.summary()`).

3. STAGE_ORDER — whether softmax output index 0..4 really corresponds to Wagner
   Stage 0..4 in that order. This depends entirely on how your label encoder /
   ImageDataGenerator class_indices were set up during training. If your classes
   came out in a different order, fix STAGE_ORDER below (e.g. if your training
   data folders were alphabetically "stage0","stage1",... this is usually fine;
   if you used a custom LabelEncoder on string labels, double check).
====================================================================================
"""

import base64
import io

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------------------------
# CONFIG — VERIFY THIS against your training notebook before trusting results
# ---------------------------------------------------------------------------
PREPROCESS_MODE = "rescale"  # one of: "rescale" (/255), "mobilenet_v2", "efficientnet", "none"
CONV_LAYER_NAME = None       # None = auto-detect last Conv2D layer (incl. nested backbones)
STAGE_ORDER = [0, 1, 2, 3, 4]  # softmax index -> Wagner stage; identity unless you trained differently
TTA_HFLIP = True             # average the prediction with a horizontally-flipped pass (matches
                              # the "Probabilitas (rata-rata TTA)" label from your Colab figure)
# ---------------------------------------------------------------------------


@tf.keras.utils.register_keras_serializable()
def _unused_placeholder(*args, **kwargs):  # pragma: no cover
    """Kept only so load_model() doesn't choke if the .keras file references a
    custom object by this name. Safe to delete if you don't have custom layers."""
    raise NotImplementedError


def load_ulcare_model(path):
    """Loads ULCARE_V3.keras. If your model used custom layers/losses that
    aren't plain Keras built-ins, you'll need to pass them here via
    `custom_objects={...}` — ping me the error message and I'll wire it in."""
    return tf.keras.models.load_model(path, compile=False)


def get_input_size(model):
    """(height, width) the model expects, read straight from the model itself."""
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    h, w = shape[1], shape[2]
    if h is None or w is None:
        h, w = 224, 224  # fallback if the model accepts variable input size
    return int(h), int(w)


def _find_last_conv_output(model):
    """Recursively finds the last Conv2D layer's output tensor, including inside
    a nested backbone (e.g. `base_model(inputs)` used as a single layer in a
    Functional model). Returns the tensor to use as the Grad-CAM target."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.output
    # No top-level Conv2D — look for a nested sub-model (a transfer-learning backbone)
    # and recurse into it; its `.output` is still wired into the outer graph because
    # it was called directly on the outer model's input tensor.
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            try:
                return _find_last_conv_output(layer)
            except Exception:
                continue
    raise ValueError(
        "Could not auto-find a Conv2D layer for Grad-CAM++. Set CONV_LAYER_NAME "
        "explicitly in model_utils.py (check model.summary() for the right name)."
    )


def _grad_target(model):
    if CONV_LAYER_NAME:
        return model.get_layer(CONV_LAYER_NAME).output
    return _find_last_conv_output(model)


def preprocess_image(pil_img, size):
    img = pil_img.convert("RGB").resize((size[1], size[0]))
    arr = np.asarray(img).astype("float32")
    if PREPROCESS_MODE == "rescale":
        arr = arr / 255.0
    elif PREPROCESS_MODE == "mobilenet_v2":
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    elif PREPROCESS_MODE == "efficientnet":
        arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    elif PREPROCESS_MODE == "none":
        pass
    else:
        raise ValueError(f"Unknown PREPROCESS_MODE: {PREPROCESS_MODE}")
    return np.expand_dims(arr, axis=0)


def predict_with_tta(model, pil_img):
    """Returns (probabilities[5], predicted_stage_index), averaged over TTA passes."""
    size = get_input_size(model)
    batch = preprocess_image(pil_img, size)
    passes = [batch]
    if TTA_HFLIP:
        passes.append(batch[:, :, ::-1, :])

    all_probs = []
    for p in passes:
        out = model(p, training=False).numpy()[0]
        all_probs.append(out)
    avg = np.mean(all_probs, axis=0)

    # Reorder if STAGE_ORDER says the model's class index != stage index directly
    probs_by_stage = np.zeros_like(avg)
    for class_idx, stage_idx in enumerate(STAGE_ORDER):
        probs_by_stage[stage_idx] = avg[class_idx]

    stage = int(np.argmax(probs_by_stage))
    return probs_by_stage * 100.0, stage


def make_gradcam_plusplus(model, pil_img, class_idx):
    """Returns a (H, W) heatmap in [0, 1], sized to the model's input resolution."""
    size = get_input_size(model)
    img_array = preprocess_image(pil_img, size)
    target = _grad_target(model)
    grad_model = tf.keras.Model(model.inputs, [target, model.output])

    img_tensor = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape3:
                conv_out, preds = grad_model(img_tensor, training=False)
                # STAGE_ORDER maps model-output-index -> stage; invert it to get back
                # the model's own class index for the stage we want to explain.
                model_class_idx = STAGE_ORDER.index(class_idx)
                score = preds[:, model_class_idx]
            grads1 = tape3.gradient(score, conv_out)
        grads2 = tape2.gradient(grads1, conv_out)
    grads3 = tape1.gradient(grads2, conv_out)

    conv_out = conv_out[0].numpy()
    grads1 = grads1[0].numpy()
    grads2 = grads2[0].numpy()
    grads3 = grads3[0].numpy()

    global_sum = conv_out.reshape((-1, conv_out.shape[-1])).sum(axis=0)

    alpha_num = grads2
    alpha_denom = grads2 * 2.0 + grads3 * global_sum.reshape((1, 1, -1))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones_like(alpha_denom))
    alphas = alpha_num / alpha_denom

    alpha_norm = np.sum(alphas, axis=(0, 1))
    alpha_norm = np.where(alpha_norm != 0.0, alpha_norm, np.ones_like(alpha_norm))
    alphas = alphas / alpha_norm.reshape((1, 1, -1))

    weights = np.maximum(grads1, 0.0)
    deep_weights = np.sum((weights * alphas).reshape((-1, conv_out.shape[-1])), axis=0)

    cam = np.sum(deep_weights * conv_out, axis=-1)
    cam = np.maximum(cam, 0.0)
    max_val = cam.max()
    cam = cam / max_val if max_val > 0 else cam
    cam = cv2.resize(cam, (size[1], size[0]))
    return cam


def heatmap_to_rgb(heatmap):
    """Jet colormap version of the heatmap (the 'Peta aktivasi' panel)."""
    heat_u8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)  # BGR
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored)


def build_overlay_with_contours(pil_img, heatmap, alpha=0.45):
    """Original photo + heatmap blend + 50%/75% activation contour lines
    ('Overlay + kontur 50% / 75%' panel)."""
    size = heatmap.shape[:2]  # (H, W)
    base = np.asarray(pil_img.convert("RGB").resize((size[1], size[0]))).astype("float32")

    heat_u8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET).astype("float32")
    colored = cv2.cvtColor(colored.astype("uint8"), cv2.COLOR_BGR2RGB).astype("float32")

    blended = (base * (1 - alpha) + colored * alpha).astype("uint8")

    for level, color, thickness, dashed in ((0.5, (255, 255, 255), 2, True), (0.75, (0, 0, 0), 2, False)):
        mask = (heatmap >= level).astype("uint8") * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if dashed:
            for cnt in contours:
                for i in range(0, len(cnt), 6):  # skip points to fake a dashed line
                    seg = cnt[i:i + 3]
                    if len(seg) >= 2:
                        cv2.polylines(blended, [seg], False, color, thickness, lineType=cv2.LINE_AA)
        else:
            cv2.drawContours(blended, contours, -1, color, thickness, lineType=cv2.LINE_AA)

    return Image.fromarray(blended)


def to_data_url(pil_img, fmt="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def build_summary_figure(pil_img, heatmap, overlay_img, probs, stage, stage_names=None):
    """Composes the 4-panel figure (Foto / Peta aktivasi / Overlay+kontur / bar chart)
    matching the layout of your Colab Grad-CAM++ output, as a single PNG for the
    separate 'Download Figure PNG' button."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stage_names = stage_names or [f"Stage {i}" for i in range(len(probs))]
    size = heatmap.shape[:2]
    photo_resized = pil_img.convert("RGB").resize((size[1], size[0]))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.1]})
    fig.suptitle(f"GRADCAM++ untuk kelas: {stage_names[stage]}", fontsize=13, fontweight="bold")

    axes[0].imshow(photo_resized); axes[0].set_title("Foto", fontsize=10); axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Peta aktivasi", fontsize=10); axes[1].axis("off")
    axes[2].imshow(overlay_img); axes[2].set_title("Overlay + kontur 50% / 75%", fontsize=10); axes[2].axis("off")

    colors = ["#cbd5e1"] * len(probs)
    colors[stage] = "#dc2626"
    axes[3].barh(range(len(probs)), probs, color=colors)
    axes[3].set_yticks(range(len(probs)))
    axes[3].set_yticklabels(stage_names, fontsize=9)
    axes[3].invert_yaxis()
    axes[3].set_xlim(0, 100)
    axes[3].set_title("Probabilitas (rata-rata TTA)", fontsize=10)
    for i, v in enumerate(probs):
        axes[3].text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=8)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
