
#!/usr/bin/env python3
"""
Team 2: MedSAM2 real-vs-imputed MRI segmentation experiment.

Outputs:
  - Real MRI segmentation
  - Imputed MRI segmentation
  - Real MRI uncertainty map
  - Imputed MRI uncertainty map
  - CSV with Dice, uncertainty and stability metrics

IMPORTANT:
This version uses GT-derived bounding boxes and the GT-derived
prompt slice. It is an ORACLE-PROMPT BASELINE, not a fully
automatic inference pipeline.

Uncertainty:
  U(v) = 4 * p(v) * (1 - p(v))
where p(v) is the fraction of perturbed prompts predicting
foreground at voxel v.

Array convention:
  SimpleITK NumPy volumes: (Z, Y, X)
  Predictor frames:       (3, 512, 512)
"""



#!/usr/bin/env python3

"""
TEAM 2
MedSAM2 real-vs-imputed MRI segmentation experiment

Automatically processes ALL subjects.

For each subject:
    Real MRI
        ↓
    MedSAM2 with multiple perturbed prompts
        ↓
    Real segmentation + uncertainty

    Imputed MRI
        ↓
    MedSAM2 with the SAME prompt perturbation protocol
        ↓
    Imputed segmentation + uncertainty

Ground truth:
    <subject>-seg.nii.gz

IMPORTANT:
    This is currently an ORACLE-PROMPT experiment.
    The prompt is derived from the GT tumor mask.

    This is useful for benchmarking prompt stability.
    It is NOT yet the final fully automatic pipeline.
"""

# ============================================================
# IMPORTS
# ============================================================

import random
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from PIL import Image
from scipy.stats import pearsonr, spearmanr

from sam2.build_sam import build_sam2_video_predictor_npz


# ============================================================
# CONFIGURATION
# ============================================================

MEDSAM2_DIR = Path.home() / "software" / "MedSAM2"



# Experiment images: original + imputed
EXPERIMENT_ROOT = Path(
    "/home/dell/software/TRUSTMRI/results/africa_pretrained_test"
)

# Ground-truth dataset
GT_ROOT = Path(
    "/home/dell/software/"
    "BraTs Africa/BraTS-Africa Dataset/"
    "BraTS-Africa_split_preprocessed/"
    "test"
)

MODALITY = "t2w"

# ------------------------------------------------------------
# MedSAM2 configuration
# ------------------------------------------------------------

CONFIG_RELATIVE = "configs/sam2.1_hiera_t512.yaml"

CONFIG_PATH = (
    MEDSAM2_DIR
    / "sam2"
    / "configs"
    / "sam2.1_hiera_t512.yaml"
)

CHECKPOINT = (
    MEDSAM2_DIR
    / "checkpoints"
    / "MedSAM2_latest.pt"
)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_DIR = (
    MEDSAM2_DIR
    / "team2_results_all_subjects"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# EXPERIMENT PARAMETERS
# ------------------------------------------------------------

IMAGE_SIZE = 512

# Start with 5.
# After confirming everything works, use 10 or 20.
N_PROMPTS = 5

# Small prompt perturbation
PROMPT_JITTER_FRACTION = 0.02
PROMPT_SCALE_JITTER = 0.05

RANDOM_SEED = 42

# Use bfloat16 on your GB10
USE_AMP = True

# Whole tumor = all non-zero BraTS labels
GT_LABELS = None


# ============================================================
# RANDOM SEEDS
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# CUDA CHECK
# ============================================================

def check_cuda():

    print("=" * 70)
    print("SYSTEM")
    print("=" * 70)

    print("PyTorch:", torch.__version__)
    print("CUDA build:", torch.version.cuda)
    print("Device:", DEVICE)

    if DEVICE == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "Capability:",
            torch.cuda.get_device_capability(0)
        )

        try:

            a = torch.randn(
                256,
                256,
                device="cuda"
            )

            b = torch.randn(
                256,
                256,
                device="cuda"
            )

            _ = a @ b

            torch.cuda.synchronize()

            print(
                "CUDA computation test passed."
            )

        except Exception as exc:

            raise RuntimeError(
                "CUDA computation failed."
            ) from exc

    print()


# ============================================================
# LOAD MEDSAM2
# ============================================================

def load_model():

    if not CONFIG_PATH.exists():

        raise FileNotFoundError(
            f"Missing config:\n{CONFIG_PATH}"
        )

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Missing checkpoint:\n{CHECKPOINT}"
        )

    print("=" * 70)
    print("LOADING MEDSAM2")
    print("=" * 70)

    predictor = build_sam2_video_predictor_npz(
        CONFIG_RELATIVE,
        str(CHECKPOINT),
        device=DEVICE,
    )

    predictor.eval()

    print("MedSAM2 loaded successfully.")
    print()

    return predictor


# ============================================================
# FIND ALL SUBJECTS
# ============================================================
def discover_cases():

    print("=" * 70)
    print("DISCOVERING SUBJECTS")
    print("=" * 70)

    if not EXPERIMENT_ROOT.exists():
        raise FileNotFoundError(
            f"Experiment root does not exist:\n"
            f"{EXPERIMENT_ROOT}"
        )

    if not GT_ROOT.exists():
        raise FileNotFoundError(
            f"GT root does not exist:\n"
            f"{GT_ROOT}"
        )

    print("Experiment root:")
    print(EXPERIMENT_ROOT)

    print("\nGT root:")
    print(GT_ROOT)

    print("\nModality:")
    print(MODALITY)

    cases = []

    # ========================================================
    # FIND ALL EXPERIMENT SUBJECT DIRECTORIES
    # ========================================================

    experiment_subjects = {}

    for category_dir in EXPERIMENT_ROOT.iterdir():

        if not category_dir.is_dir():
            continue

        # e.g.
        # 51_OtherNeoplasms
        # 95_Glioma

        for subject_dir in category_dir.iterdir():

            if not subject_dir.is_dir():
                continue

            subject_id = subject_dir.name

            if not subject_id.startswith("BraTS-"):
                continue

            experiment_subjects[subject_id] = {
                "category": category_dir.name,
                "subject_dir": subject_dir,
            }

    print(
        f"\nFound {len(experiment_subjects)} "
        "experiment subjects."
    )

    # ========================================================
    # FIND GT SUBJECT DIRECTORIES
    # ========================================================

    gt_subjects = {}

    for category_dir in GT_ROOT.iterdir():

        if not category_dir.is_dir():
            continue

        for subject_dir in category_dir.iterdir():

            if not subject_dir.is_dir():
                continue

            subject_id = subject_dir.name

            if not subject_id.startswith("BraTS-"):
                continue

            gt_subjects[subject_id] = {
                "category": category_dir.name,
                "subject_dir": subject_dir,
            }

    print(
        f"Found {len(gt_subjects)} GT subjects."
    )

    # ========================================================
    # MATCH SUBJECTS
    # ========================================================

    for subject_id, exp_info in sorted(
        experiment_subjects.items()
    ):

        category = exp_info["category"]
        subject_dir = exp_info["subject_dir"]

        print()
        print("-" * 70)
        print(f"Checking: {subject_id}")
        print(f"Category: {category}")

        # ----------------------------------------------------
        # Find corresponding GT subject
        # ----------------------------------------------------

        if subject_id not in gt_subjects:

            print(
                "[SKIP] No corresponding GT subject."
            )

            continue

        gt_subject_dir = gt_subjects[
            subject_id
        ]["subject_dir"]

        # ----------------------------------------------------
        # Find tumor segmentation
        # ----------------------------------------------------

        gt_candidates = [
            gt_subject_dir
            / f"{subject_id}-seg.nii.gz",

            gt_subject_dir
            / f"{subject_id}-seg.nii",

            gt_subject_dir
            / "seg.nii.gz",

            gt_subject_dir
            / "seg.nii",
        ]

        gt_path = None

        for candidate in gt_candidates:

            if candidate.exists():

                gt_path = candidate
                break

        if gt_path is None:

            print(
                "[WARNING] No seg.nii.gz found."
            )

            print(
                "Available mask-like files:"
            )

            for f in gt_subject_dir.glob(
                "*.nii.gz"
            ):

                if (
                    "mask" in f.name.lower()
                    or
                    "seg" in f.name.lower()
                ):
                    print(
                        "   ",
                        f
                    )

            print(
                "[SKIP] Subject skipped."
            )

            continue

        # ----------------------------------------------------
        # Modality
        # ----------------------------------------------------

        modality_dir = (
            subject_dir
            / MODALITY
        )

        if not modality_dir.exists():

            print(
                f"[SKIP] Missing modality directory:\n"
                f"{modality_dir}"
            )

            continue

        # ----------------------------------------------------
        # ORIGINAL
        # ----------------------------------------------------

        original_dir = (
            modality_dir
            / "original"
        )

        if not original_dir.exists():

            print(
                f"[SKIP] Missing original directory:\n"
                f"{original_dir}"
            )

            continue

        original_files = sorted(
            list(
                original_dir.glob("*.nii.gz")
            )
            +
            list(
                original_dir.glob("*.nii")
            )
        )

        original_files = [
            f
            for f in original_files
            if "seg" not in f.name.lower()
            and "mask" not in f.name.lower()
        ]

        modality_matches = [
            f
            for f in original_files
            if MODALITY.lower()
            in f.name.lower()
        ]

        if len(modality_matches) == 1:

            real_path = modality_matches[0]

        elif len(original_files) == 1:

            real_path = original_files[0]

        else:

            print(
                "[SKIP] Could not uniquely identify "
                "original MRI."
            )

            for f in original_files:
                print(
                    "   ",
                    f
                )

            continue

        # ----------------------------------------------------
        # IMPUTED
        # ----------------------------------------------------

        imputed_dir = (
            modality_dir
            / "imputed"
        )

        if not imputed_dir.exists():

            print(
                f"[SKIP] Missing imputed directory:\n"
                f"{imputed_dir}"
            )

            continue

        imputed_files = sorted(
            list(
                imputed_dir.glob("*.nii.gz")
            )
            +
            list(
                imputed_dir.glob("*.nii")
            )
        )

        imputed_files = [
            f
            for f in imputed_files
            if "seg" not in f.name.lower()
            and "mask" not in f.name.lower()
        ]

        modality_matches = [
            f
            for f in imputed_files
            if MODALITY.lower()
            in f.name.lower()
        ]

        if len(modality_matches) == 1:

            imputed_path = modality_matches[0]

        elif len(imputed_files) == 1:

            imputed_path = imputed_files[0]

        else:

            print(
                "[SKIP] Could not uniquely identify "
                "imputed MRI."
            )

            for f in imputed_files:
                print(
                    "   ",
                    f
                )

            continue

        # ====================================================
        # ADD CASE
        # ====================================================

        cases.append(
            {
                "case_id": subject_id,

                "category": category,

                "real": real_path,

                "imputed": imputed_path,

                "mask": gt_path,
            }
        )

        print()
        print("  ORIGINAL:")
        print(
            "   ",
            real_path
        )

        print("  IMPUTED:")
        print(
            "   ",
            imputed_path
        )

        print("  TUMOR GT:")
        print(
            "   ",
            gt_path
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        f"PAIRED SUBJECTS READY: {len(cases)}"
    )
    print("=" * 70)

    for case in cases:

        print(
            f"{case['case_id']} "
            f"[{case['category']}]"
        )

    print()

    return cases
# ============================================================
# LOAD NIFTI
# ============================================================

def load_nifti(path):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    nii = sitk.ReadImage(
        str(path)
    )

    array = sitk.GetArrayFromImage(
        nii
    )

    if array.ndim != 3:

        raise ValueError(
            f"Expected 3D volume:\n{path}\n"
            f"Shape={array.shape}"
        )

    return nii, array


# ============================================================
# CHECK GEOMETRY
# ============================================================

def same_geometry(
    reference,
    other,
    name
):

    if reference.GetSize() != other.GetSize():

        raise ValueError(
            f"{name}: dimensions differ.\n"
            f"Reference: {reference.GetSize()}\n"
            f"Other: {other.GetSize()}"
        )

    if not np.allclose(
        reference.GetSpacing(),
        other.GetSpacing(),
        atol=1e-5
    ):

        raise ValueError(
            f"{name}: spacing differs."
        )

    if not np.allclose(
        reference.GetOrigin(),
        other.GetOrigin(),
        atol=1e-4
    ):

        raise ValueError(
            f"{name}: origin differs."
        )

    if not np.allclose(
        reference.GetDirection(),
        other.GetDirection(),
        atol=1e-5
    ):

        raise ValueError(
            f"{name}: direction differs."
        )


# ============================================================
# GT MASK
# ============================================================

def make_gt_binary(
    gt_array
):

    if GT_LABELS is None:

        return (
            gt_array > 0
        ).astype(np.uint8)

    return np.isin(
        gt_array,
        GT_LABELS
    ).astype(np.uint8)


# ============================================================
# MRI PREPROCESSING
# ============================================================

def make_predictor_frames(
    volume
):

    volume = np.asarray(
        volume,
        dtype=np.float32
    )

    finite = np.isfinite(
        volume
    )

    if not finite.any():

        raise ValueError(
            "MRI contains no finite values."
        )

    values = volume[finite]

    low, high = np.percentile(
        values,
        [1, 99]
    )

    if high <= low:

        raise ValueError(
            f"Invalid MRI intensity range: "
            f"{low}, {high}"
        )

    volume_norm = np.zeros_like(
        volume,
        dtype=np.float32
    )

    volume_norm[finite] = np.clip(
        (
            volume[finite] - low
        )
        /
        (
            high - low
        ),
        0.0,
        1.0,
    )

    # ImageNet normalization
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )[:, None, None]

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )[:, None, None]

    frames = []

    for z in range(
        volume_norm.shape[0]
    ):

        slice_2d = (
            volume_norm[z] * 255.0
        ).astype(np.uint8)

        resized = Image.fromarray(
            slice_2d
        ).resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            resample=Image.Resampling.BILINEAR
        )

        gray = (
            np.asarray(
                resized,
                dtype=np.float32
            )
            / 255.0
        )

        rgb = np.stack(
            [
                gray,
                gray,
                gray
            ],
            axis=0
        )

        rgb = (
            rgb - mean
        ) / std

        frames.append(
            torch.from_numpy(
                rgb.astype(
                    np.float32
                )
            )
        )

    return frames


# ============================================================
# GT PROMPT
# ============================================================

def get_prompt_slice_and_box(
    gt_binary
):

    areas = gt_binary.sum(
        axis=(1, 2)
    )

    if areas.max() == 0:

        raise ValueError(
            "GT contains no tumor."
        )

    z = int(
        np.argmax(areas)
    )

    ys, xs = np.where(
        gt_binary[z] > 0
    )

    x1 = float(
        xs.min()
    )

    y1 = float(
        ys.min()
    )

    x2 = float(
        xs.max() + 1
    )

    y2 = float(
        ys.max() + 1
    )

    return z, np.array(
        [
            x1,
            y1,
            x2,
            y2
        ],
        dtype=np.float32
    )


# ============================================================
# PERTURB PROMPT
# ============================================================

def perturb_box(
    box,
    width,
    height,
    rng
):

    x1, y1, x2, y2 = box.copy()

    bw = max(
        x2 - x1,
        2.0
    )

    bh = max(
        y2 - y1,
        2.0
    )

    cx = (
        x1 + x2
    ) / 2.0

    cy = (
        y1 + y2
    ) / 2.0

    cx += (
        rng.uniform(-1, 1)
        * PROMPT_JITTER_FRACTION
        * bw
    )

    cy += (
        rng.uniform(-1, 1)
        * PROMPT_JITTER_FRACTION
        * bh
    )

    scale_x = (
        1.0
        +
        rng.uniform(
            -PROMPT_SCALE_JITTER,
            PROMPT_SCALE_JITTER
        )
    )

    scale_y = (
        1.0
        +
        rng.uniform(
            -PROMPT_SCALE_JITTER,
            PROMPT_SCALE_JITTER
        )
    )

    new_w = max(
        2.0,
        bw * scale_x
    )

    new_h = max(
        2.0,
        bh * scale_y
    )

    x1 = np.clip(
        cx - new_w / 2,
        0,
        width - 1
    )

    x2 = np.clip(
        cx + new_w / 2,
        x1 + 1,
        width
    )

    y1 = np.clip(
        cy - new_h / 2,
        0,
        height - 1
    )

    y2 = np.clip(
        cy + new_h / 2,
        y1 + 1,
        height
    )

    return np.array(
        [
            x1,
            y1,
            x2,
            y2
        ],
        dtype=np.float32
    )


# ============================================================
# MEDSAM2 PROPAGATION
# ============================================================

def propagate_direction(
    predictor,
    frames,
    start_z,
    box,
    native_height,
    native_width,
    reverse
):

    state = predictor.init_state(
        images=frames,
        video_height=native_height,
        video_width=native_width,
        offload_video_to_cpu=True,
    )

    try:

        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=int(start_z),
            obj_id=1,
            box=box,
        )

        predictions = {}

        with torch.inference_mode():

            if (
                DEVICE == "cuda"
                and USE_AMP
            ):

                autocast_context = (
                    torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16
                    )
                )

            else:

                from contextlib import nullcontext

                autocast_context = (
                    nullcontext()
                )

            with autocast_context:

                for (
                    frame_idx,
                    obj_ids,
                    mask_logits
                ) in predictor.propagate_in_video(
                    state,
                    start_frame_idx=int(start_z),
                    reverse=reverse,
                ):

                    logits = (
                        mask_logits[0]
                        .squeeze()
                    )

                    mask = (
                        logits > 0
                    ).to(
                        dtype=torch.uint8
                    ).cpu().numpy()

                    predictions[
                        int(frame_idx)
                    ] = mask

        return predictions

    finally:

        del state

        if DEVICE == "cuda":

            torch.cuda.empty_cache()


# ============================================================
# RUN ONE PROMPT
# ============================================================

def run_one_prompt(
    predictor,
    frames,
    start_z,
    box,
    depth,
    native_height,
    native_width
):

    forward = propagate_direction(
        predictor,
        frames,
        start_z,
        box,
        native_height,
        native_width,
        reverse=False,
    )

    reverse = propagate_direction(
        predictor,
        frames,
        start_z,
        box,
        native_height,
        native_width,
        reverse=True,
    )

    volume_mask = np.zeros(
        (
            depth,
            native_height,
            native_width
        ),
        dtype=np.uint8
    )

    for z, mask in reverse.items():

        volume_mask[z] = mask

    for z, mask in forward.items():

        volume_mask[z] = mask

    return volume_mask


# ============================================================
# SEGMENTATION EVALUATION METRICS
# ============================================================

from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt
)


# ============================================================
# BASIC CONFUSION MATRIX
# ============================================================

def confusion_counts(
    prediction,
    ground_truth
):
    """
    Calculate voxel-level TP, TN, FP, FN.

    Both inputs must be binary masks.
    """

    prediction = (
        prediction > 0
    )

    ground_truth = (
        ground_truth > 0
    )

    tp = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    tn = np.logical_and(
        ~prediction,
        ~ground_truth
    ).sum()

    fp = np.logical_and(
        prediction,
        ~ground_truth
    ).sum()

    fn = np.logical_and(
        ~prediction,
        ground_truth
    ).sum()

    return (
        int(tp),
        int(tn),
        int(fp),
        int(fn)
    )


# ============================================================
# DICE / DSC
# ============================================================

def dice_score(
    prediction,
    ground_truth
):

    prediction = (
        prediction > 0
    )

    ground_truth = (
        ground_truth > 0
    )

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    denominator = (
        prediction.sum()
        +
        ground_truth.sum()
    )

    if denominator == 0:

        return 1.0

    return float(
        2.0
        *
        intersection
        /
        denominator
    )


# ============================================================
# IOU / JACCARD
# ============================================================

def iou_score(
    prediction,
    ground_truth
):

    prediction = (
        prediction > 0
    )

    ground_truth = (
        ground_truth > 0
    )

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    union = np.logical_or(
        prediction,
        ground_truth
    ).sum()

    if union == 0:

        return 1.0

    return float(
        intersection / union
    )


# ============================================================
# SENSITIVITY / RECALL
# ============================================================

def sensitivity_score(
    prediction,
    ground_truth
):

    tp, tn, fp, fn = (
        confusion_counts(
            prediction,
            ground_truth
        )
    )

    denominator = (
        tp + fn
    )

    if denominator == 0:

        return 1.0

    return float(
        tp / denominator
    )


# ============================================================
# SPECIFICITY
# ============================================================

def specificity_score(
    prediction,
    ground_truth
):

    tp, tn, fp, fn = (
        confusion_counts(
            prediction,
            ground_truth
        )
    )

    denominator = (
        tn + fp
    )

    if denominator == 0:

        return 1.0

    return float(
        tn / denominator
    )


# ============================================================
# PRECISION
# ============================================================

def precision_score_binary(
    prediction,
    ground_truth
):

    tp, tn, fp, fn = (
        confusion_counts(
            prediction,
            ground_truth
        )
    )

    denominator = (
        tp + fp
    )

    if denominator == 0:

        return 1.0

    return float(
        tp / denominator
    )


# ============================================================
# F1 SCORE
# ============================================================

def f1_score_binary(
    prediction,
    ground_truth
):

    precision = (
        precision_score_binary(
            prediction,
            ground_truth
        )
    )

    recall = (
        sensitivity_score(
            prediction,
            ground_truth
        )
    )

    denominator = (
        precision + recall
    )

    if denominator == 0:

        return 0.0

    return float(
        2.0
        *
        precision
        *
        recall
        /
        denominator
    )


# ============================================================
# SURFACE EXTRACTION
# ============================================================

def get_surface(
    mask
):
    """
    Extract boundary/surface voxels from a binary 3D mask.
    """

    mask = (
        mask > 0
    )

    if not mask.any():

        return np.zeros_like(
            mask,
            dtype=bool
        )

    eroded = binary_erosion(
        mask,
        structure=np.ones(
            (3, 3, 3),
            dtype=bool
        ),
        border_value=0
    )

    surface = (
        mask
        &
        ~eroded
    )

    return surface


# ============================================================
# SURFACE DISTANCES
# ============================================================

def surface_distances(
    prediction,
    ground_truth,
    spacing
):
    """
    Calculate distances between prediction and GT surfaces.

    spacing:
        (z_spacing, y_spacing, x_spacing)
    """

    prediction = (
        prediction > 0
    )

    ground_truth = (
        ground_truth > 0
    )

    pred_surface = get_surface(
        prediction
    )

    gt_surface = get_surface(
        ground_truth
    )

    if not pred_surface.any():
        return None

    if not gt_surface.any():
        return None

    # Distance to prediction surface
    dt_pred = distance_transform_edt(
        ~pred_surface,
        sampling=spacing
    )

    # Distance to GT surface
    dt_gt = distance_transform_edt(
        ~gt_surface,
        sampling=spacing
    )

    # GT surface -> prediction
    gt_to_pred = dt_pred[
        gt_surface
    ]

    # Prediction surface -> GT
    pred_to_gt = dt_gt[
        pred_surface
    ]

    return (
        pred_to_gt,
        gt_to_pred
    )


# ============================================================
# HD95
# ============================================================

def hd95_score(
    prediction,
    ground_truth,
    spacing
):
    """
    95th percentile Hausdorff distance.

    Units: millimeters.
    """

    distances = surface_distances(
        prediction,
        ground_truth,
        spacing
    )

    if distances is None:

        pred_exists = (
            np.any(prediction > 0)
        )

        gt_exists = (
            np.any(ground_truth > 0)
        )

        if (
            not pred_exists
            and
            not gt_exists
        ):
            return 0.0

        return np.inf

    pred_to_gt, gt_to_pred = (
        distances
    )

    all_distances = np.concatenate(
        [
            pred_to_gt,
            gt_to_pred
        ]
    )

    if len(all_distances) == 0:

        return 0.0

    return float(
        np.percentile(
            all_distances,
            95
        )
    )


# ============================================================
# ASSD
# ============================================================

def assd_score(
    prediction,
    ground_truth,
    spacing
):
    """
    Average Symmetric Surface Distance.

    Units: millimeters.
    """

    distances = surface_distances(
        prediction,
        ground_truth,
        spacing
    )

    if distances is None:

        pred_exists = (
            np.any(prediction > 0)
        )

        gt_exists = (
            np.any(ground_truth > 0)
        )

        if (
            not pred_exists
            and
            not gt_exists
        ):
            return 0.0

        return np.inf

    pred_to_gt, gt_to_pred = (
        distances
    )

    all_distances = np.concatenate(
        [
            pred_to_gt,
            gt_to_pred
        ]
    )

    if len(all_distances) == 0:

        return 0.0

    return float(
        np.mean(
            all_distances
        )
    )


# ============================================================
# NORMALIZED SURFACE DICE
# ============================================================

def normalized_surface_dice(
    prediction,
    ground_truth,
    spacing,
    tolerance_mm=1.0
):
    """
    Normalized Surface Dice (NSD).

    A surface voxel is considered correctly
    matched if its distance to the opposite
    surface is <= tolerance_mm.
    """

    distances = surface_distances(
        prediction,
        ground_truth,
        spacing
    )

    if distances is None:

        pred_exists = (
            np.any(prediction > 0)
        )

        gt_exists = (
            np.any(ground_truth > 0)
        )

        if (
            not pred_exists
            and
            not gt_exists
        ):
            return 1.0

        return 0.0

    pred_to_gt, gt_to_pred = (
        distances
    )

    pred_surface_hits = (
        pred_to_gt
        <= tolerance_mm
    ).sum()

    gt_surface_hits = (
        gt_to_pred
        <= tolerance_mm
    ).sum()

    numerator = (
        pred_surface_hits
        +
        gt_surface_hits
    )

    denominator = (
        len(pred_to_gt)
        +
        len(gt_to_pred)
    )

    if denominator == 0:

        return 1.0

    return float(
        numerator / denominator
    )


# ============================================================
# VOLUME SIMILARITY
# ============================================================

def volume_similarity(
    prediction,
    ground_truth
):
    """
    Volume Similarity:

        1 - |Vpred - Vgt| / (Vpred + Vgt)

    Range:
        0 to 1
    """

    v_pred = float(
        np.sum(
            prediction > 0
        )
    )

    v_gt = float(
        np.sum(
            ground_truth > 0
        )
    )

    denominator = (
        v_pred + v_gt
    )

    if denominator == 0:

        return 1.0

    return float(
        1.0
        -
        abs(
            v_pred - v_gt
        )
        /
        denominator
    )


# ============================================================
# ALL METRICS
# ============================================================

def calculate_segmentation_metrics(
    prediction,
    ground_truth,
    spacing,
    nsd_tolerance_mm=1.0
):
    """
    Calculate ALL segmentation metrics used
    in the Team 2 experiment.

    Returns a dictionary.
    """

    prediction = (
        prediction > 0
    ).astype(
        np.uint8
    )

    ground_truth = (
        ground_truth > 0
    ).astype(
        np.uint8
    )

    tp, tn, fp, fn = (
        confusion_counts(
            prediction,
            ground_truth
        )
    )

    metrics = {

        # ------------------------------
        # Overlap
        # ------------------------------

        "dice":
            dice_score(
                prediction,
                ground_truth
            ),

        "iou":
            iou_score(
                prediction,
                ground_truth
            ),

        # ------------------------------
        # Classification
        # ------------------------------

        "sensitivity":
            sensitivity_score(
                prediction,
                ground_truth
            ),

        "specificity":
            specificity_score(
                prediction,
                ground_truth
            ),

        "precision":
            precision_score_binary(
                prediction,
                ground_truth
            ),

        "f1":
            f1_score_binary(
                prediction,
                ground_truth
            ),

        # ------------------------------
        # Surface metrics
        # ------------------------------

        "hd95_mm":
            hd95_score(
                prediction,
                ground_truth,
                spacing
            ),

        "assd_mm":
            assd_score(
                prediction,
                ground_truth,
                spacing
            ),

        "nsd":
            normalized_surface_dice(
                prediction,
                ground_truth,
                spacing,
                tolerance_mm=nsd_tolerance_mm
            ),

        # ------------------------------
        # Volume
        # ------------------------------

        "volume_similarity":
            volume_similarity(
                prediction,
                ground_truth
            ),

        # ------------------------------
        # Raw counts
        # ------------------------------

        "tp":
            tp,

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "prediction_voxels":
            int(
                prediction.sum()
            ),

        "ground_truth_voxels":
            int(
                ground_truth.sum()
            ),
    }

    return metrics

# ============================================================
# UNCERTAINTY
# ============================================================

def calculate_uncertainty(
    prediction_masks
):

    stack = np.stack(
        prediction_masks,
        axis=0
    ).astype(np.float32)

    p = stack.mean(
        axis=0
    )

    # 0 = complete agreement
    # 1 = maximum disagreement
    uncertainty = (
        4.0
        *
        p
        *
        (1.0 - p)
    )

    consensus = (
        p >= 0.5
    ).astype(np.uint8)

    return (
        p,
        uncertainty.astype(
            np.float32
        ),
        consensus
    )


# ============================================================
# SAVE NIFTI
# ============================================================

def save_nifti(
    array,
    reference_nii,
    output_path
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    image = sitk.GetImageFromArray(
        array
    )

    image.CopyInformation(
        reference_nii
    )

    sitk.WriteImage(
        image,
        str(output_path)
    )


# ============================================================
# PROCESS ONE MODALITY
# ============================================================

# ============================================================
# PROCESS ONE MODALITY
# ============================================================

def process_modality(
    predictor,
    case,
    modality,
    mri_path,
    gt_nii,
    gt_binary,
    prompt_z,
    original_box
):

    case_id = case["case_id"]

    print()
    print("-" * 70)
    print(f"{case_id} | {modality}")
    print(f"MRI: {mri_path}")

    # ========================================================
    # LOAD MRI
    # ========================================================

    mri_nii, mri_volume = load_nifti(
        mri_path
    )

    same_geometry(
        gt_nii,
        mri_nii,
        f"{modality} MRI"
    )

    depth, height, width = mri_volume.shape

    print(
        f"Volume shape: {mri_volume.shape}"
    )

    # ========================================================
    # PREPROCESS MRI FOR MEDSAM2
    # ========================================================

    frames = make_predictor_frames(
        mri_volume
    )

    # ========================================================
    # REPRODUCIBLE PROMPT PERTURBATIONS
    # ========================================================

    rng = np.random.default_rng(
        RANDOM_SEED
        +
        sum(
            ord(c)
            for c in (
                case_id + modality
            )
        )
    )

    prompt_predictions = []

    # ========================================================
    # RUN MULTIPLE PERTURBED PROMPTS
    # ========================================================

    for i in range(N_PROMPTS):

        box = perturb_box(
            original_box,
            width,
            height,
            rng
        )

        print(
            f"  Prompt {i + 1}/{N_PROMPTS}; "
            f"box={np.round(box, 1).tolist()}"
        )

        prediction = run_one_prompt(
            predictor,
            frames,
            prompt_z,
            box,
            depth,
            height,
            width
        )

        print(
            f"    predicted voxels: "
            f"{int(prediction.sum())}"
        )

        prompt_predictions.append(
            prediction
        )

    # ========================================================
    # UNCERTAINTY
    # ========================================================
    #
    # IMPORTANT:
    # Uncertainty is ALWAYS calculated.
    #
    # p(v):
    # fraction of perturbed prompts predicting foreground
    #
    # uncertainty:
    # U(v) = 4 * p(v) * (1 - p(v))
    #
    # consensus:
    # majority-vote final segmentation
    # ========================================================

    probability_map, uncertainty, consensus = (
        calculate_uncertainty(
            prompt_predictions
        )
    )

    # ========================================================
    # UNCERTAINTY STATISTICS
    # ========================================================

    # Whole volume uncertainty
    mean_uncertainty_volume = float(
        np.mean(uncertainty)
    )

    median_uncertainty_volume = float(
        np.median(uncertainty)
    )

    std_uncertainty_volume = float(
        np.std(uncertainty)
    )

    p95_uncertainty_volume = float(
        np.percentile(
            uncertainty,
            95
        )
    )

    max_uncertainty_volume = float(
        np.max(uncertainty)
    )

    fraction_high_uncertainty = float(
        np.mean(
            uncertainty > 0.5
        )
    )

    # --------------------------------------------------------
    # UNCERTAINTY INSIDE GT TUMOR
    # --------------------------------------------------------

    gt_tumor = (
        gt_binary > 0
    )

    if gt_tumor.any():

        tumor_uncertainty_values = (
            uncertainty[
                gt_tumor
            ]
        )

        mean_uncertainty_tumor = float(
            np.mean(
                tumor_uncertainty_values
            )
        )

        median_uncertainty_tumor = float(
            np.median(
                tumor_uncertainty_values
            )
        )

        std_uncertainty_tumor = float(
            np.std(
                tumor_uncertainty_values
            )
        )

        p95_uncertainty_tumor = float(
            np.percentile(
                tumor_uncertainty_values,
                95
            )
        )

    else:

        mean_uncertainty_tumor = np.nan
        median_uncertainty_tumor = np.nan
        std_uncertainty_tumor = np.nan
        p95_uncertainty_tumor = np.nan

    # --------------------------------------------------------
    # UNCERTAINTY INSIDE PREDICTED TUMOR
    # --------------------------------------------------------
    #
    # This is useful because it does NOT require GT to define
    # the region whose uncertainty is being measured.
    # --------------------------------------------------------

    predicted_tumor = (
        consensus > 0
    )

    if predicted_tumor.any():

        predicted_tumor_uncertainty_values = (
            uncertainty[
                predicted_tumor
            ]
        )

        mean_uncertainty_predicted_tumor = float(
            np.mean(
                predicted_tumor_uncertainty_values
            )
        )

        median_uncertainty_predicted_tumor = float(
            np.median(
                predicted_tumor_uncertainty_values
            )
        )

        std_uncertainty_predicted_tumor = float(
            np.std(
                predicted_tumor_uncertainty_values
            )
        )

        p95_uncertainty_predicted_tumor = float(
            np.percentile(
                predicted_tumor_uncertainty_values,
                95
            )
        )

    else:

        mean_uncertainty_predicted_tumor = np.nan
        median_uncertainty_predicted_tumor = np.nan
        std_uncertainty_predicted_tumor = np.nan
        p95_uncertainty_predicted_tumor = np.nan

    # --------------------------------------------------------
    # BACKGROUND UNCERTAINTY
    # --------------------------------------------------------

    background = ~gt_tumor

    if background.any():

        background_uncertainty_values = (
            uncertainty[
                background
            ]
        )

        mean_uncertainty_background = float(
            np.mean(
                background_uncertainty_values
            )
        )

    else:

        mean_uncertainty_background = np.nan

    # ========================================================
    # SEGMENTATION EVALUATION METRICS
    # ========================================================

    spacing = mri_nii.GetSpacing()

    # SimpleITK:
    #     (X, Y, Z)
    #
    # NumPy:
    #     (Z, Y, X)

    spacing_zyx = (
        float(spacing[2]),
        float(spacing[1]),
        float(spacing[0])
    )

    metrics = calculate_segmentation_metrics(
        prediction=consensus,
        ground_truth=gt_binary,
        spacing=spacing_zyx,
        nsd_tolerance_mm=1.0
    )

    # ========================================================
    # PRINT SEGMENTATION METRICS
    # ========================================================

    print()
    print("  SEGMENTATION METRICS")
    print("  --------------------")

    print(
        f"  Dice / DSC:        "
        f"{metrics['dice']:.4f}"
    )

    print(
        f"  IoU:               "
        f"{metrics['iou']:.4f}"
    )

    print(
        f"  Sensitivity:       "
        f"{metrics['sensitivity']:.4f}"
    )

    print(
        f"  Specificity:       "
        f"{metrics['specificity']:.4f}"
    )

    print(
        f"  Precision:         "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"  F1:                "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"  HD95:              "
        f"{metrics['hd95_mm']:.4f} mm"
    )

    print(
        f"  ASSD:              "
        f"{metrics['assd_mm']:.4f} mm"
    )

    print(
        f"  NSD:               "
        f"{metrics['nsd']:.4f}"
    )

    print(
        f"  Volume similarity: "
        f"{metrics['volume_similarity']:.4f}"
    )

    # ========================================================
    # PRINT UNCERTAINTY
    # ========================================================

    print()
    print("  UNCERTAINTY")
    print("  -----------")

    print(
        f"  Mean volume uncertainty:       "
        f"{mean_uncertainty_volume:.4f}"
    )

    print(
        f"  Median volume uncertainty:     "
        f"{median_uncertainty_volume:.4f}"
    )

    print(
        f"  Std volume uncertainty:        "
        f"{std_uncertainty_volume:.4f}"
    )

    print(
        f"  P95 volume uncertainty:        "
        f"{p95_uncertainty_volume:.4f}"
    )

    print(
        f"  Mean GT tumor uncertainty:     "
        f"{mean_uncertainty_tumor:.4f}"
    )

    print(
        f"  Mean predicted tumor uncertainty: "
        f"{mean_uncertainty_predicted_tumor:.4f}"
    )

    print(
        f"  Mean background uncertainty:   "
        f"{mean_uncertainty_background:.4f}"
    )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    prefix = (
        OUTPUT_DIR
        /
        f"{case_id}_{modality}"
    )

    prediction_path = (
        str(prefix)
        +
        "_prediction.nii.gz"
    )

    uncertainty_path = (
        str(prefix)
        +
        "_uncertainty.nii.gz"
    )

    probability_path = (
        str(prefix)
        +
        "_probability.nii.gz"
    )

    # ========================================================
    # SAVE FINAL PREDICTION
    # ========================================================

    save_nifti(
        consensus.astype(
            np.uint8
        ),
        mri_nii,
        prediction_path
    )

    # ========================================================
    # SAVE UNCERTAINTY MAP
    # ========================================================

    save_nifti(
        uncertainty.astype(
            np.float32
        ),
        mri_nii,
        uncertainty_path
    )

    # ========================================================
    # SAVE PROMPT AGREEMENT / PROBABILITY MAP
    # ========================================================

    save_nifti(
        probability_map.astype(
            np.float32
        ),
        mri_nii,
        probability_path
    )

    # ========================================================
    # SAVE INDIVIDUAL PROMPT MASKS
    # ========================================================

    prompt_dir = (
        OUTPUT_DIR
        /
        f"{case_id}_{modality}_prompt_masks"
    )

    prompt_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for i, mask in enumerate(
        prompt_predictions
    ):

        save_nifti(
            mask.astype(
                np.uint8
            ),
            mri_nii,
            prompt_dir
            /
            f"prompt_{i + 1:02d}.nii.gz"
        )

    # ========================================================
    # OUTPUT SUMMARY
    # ========================================================

    print()
    print(
        f"  Saved prediction:   "
        f"{prediction_path}"
    )

    print(
        f"  Saved uncertainty:  "
        f"{uncertainty_path}"
    )

    print(
        f"  Saved probability:  "
        f"{probability_path}"
    )

    # ========================================================
    # RETURN ALL RESULTS
    # ========================================================

    result = {

        # ----------------------------------------------------
        # IDENTIFICATION
        # ----------------------------------------------------

        "case_id":
            case_id,

        "category":
            case["category"],

        "modality":
            modality,

        "prompt_slice_z":
            prompt_z,

        "n_prompts":
            N_PROMPTS,

        # ----------------------------------------------------
        # SEGMENTATION METRICS
        # ----------------------------------------------------

        **metrics,

        # ----------------------------------------------------
        # UNCERTAINTY - WHOLE VOLUME
        # ----------------------------------------------------

        "mean_uncertainty_volume":
            mean_uncertainty_volume,

        "median_uncertainty_volume":
            median_uncertainty_volume,

        "std_uncertainty_volume":
            std_uncertainty_volume,

        "p95_uncertainty_volume":
            p95_uncertainty_volume,

        "max_uncertainty_volume":
            max_uncertainty_volume,

        "fraction_uncertainty_gt_0_5":
            fraction_high_uncertainty,

        # ----------------------------------------------------
        # UNCERTAINTY - GT TUMOR
        # ----------------------------------------------------

        "mean_uncertainty_tumor":
            mean_uncertainty_tumor,

        "median_uncertainty_tumor":
            median_uncertainty_tumor,

        "std_uncertainty_tumor":
            std_uncertainty_tumor,

        "p95_uncertainty_tumor":
            p95_uncertainty_tumor,

        # ----------------------------------------------------
        # UNCERTAINTY - PREDICTED TUMOR
        # ----------------------------------------------------

        "mean_uncertainty_predicted_tumor":
            mean_uncertainty_predicted_tumor,

        "median_uncertainty_predicted_tumor":
            median_uncertainty_predicted_tumor,

        "std_uncertainty_predicted_tumor":
            std_uncertainty_predicted_tumor,

        "p95_uncertainty_predicted_tumor":
            p95_uncertainty_predicted_tumor,

        # ----------------------------------------------------
        # UNCERTAINTY - BACKGROUND
        # ----------------------------------------------------

        "mean_uncertainty_background":
            mean_uncertainty_background,

        # ----------------------------------------------------
        # OUTPUT PATHS
        # ----------------------------------------------------

        "prediction_path":
            prediction_path,

        "uncertainty_path":
            uncertainty_path,

        "probability_path":
            probability_path,
    }

    return result

# ============================================================
# CORRELATIONS
# ============================================================

def safe_correlation(
    x,
    y,
    method
):

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 3:

        return np.nan, np.nan

    if (
        np.std(x) == 0
        or
        np.std(y) == 0
    ):

        return np.nan, np.nan

    if method == "pearson":

        result = pearsonr(
            x,
            y
        )

    else:

        result = spearmanr(
            x,
            y
        )

    return (
        float(result.statistic),
        float(result.pvalue)
    )


# ============================================================
# FINAL ANALYSIS
# ============================================================

def analyze_results(
    results
):

    df = pd.DataFrame(
        results
    )

    csv_path = (
        OUTPUT_DIR
        /
        "segmentation_results.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # REAL / IMPUTED SUMMARY
    # --------------------------------------------------------

    for modality in [
        "real",
        "imputed"
    ]:

        subset = df[
            df["modality"]
            ==
            modality
        ]

        if subset.empty:

            continue

        print()
        print(
            modality.upper()
        )

        print(
            "Subjects:",
            len(subset)
        )

        print(
            "Mean Dice:",
            f"{subset['dice'].mean():.4f}"
        )

        print(
            "Median Dice:",
            f"{subset['dice'].median():.4f}"
        )

        print(
            "Std Dice:",
            f"{subset['dice'].std():.4f}"
        )

        print(
            "Mean tumor uncertainty:",
            f"{subset['mean_uncertainty_tumor'].mean():.4f}"
        )

        r, rp = safe_correlation(
            subset[
                "mean_uncertainty_tumor"
            ],
            subset["dice"],
            "pearson"
        )

        rho, sp = safe_correlation(
            subset[
                "mean_uncertainty_tumor"
            ],
            subset["dice"],
            "spearman"
        )

        print(
            f"Pearson r: {r:.4f}, "
            f"p={rp:.4g}"
        )

        print(
            f"Spearman rho: {rho:.4f}, "
            f"p={sp:.4g}"
        )

    # --------------------------------------------------------
    # PAIRED REAL VS IMPUTED
    # --------------------------------------------------------

    dice_pivot = df.pivot(
        index="case_id",
        columns="modality",
        values="dice"
    )

    uncertainty_pivot = df.pivot(
        index="case_id",
        columns="modality",
        values="mean_uncertainty_tumor"
    )

    if {
        "real",
        "imputed"
    }.issubset(
        dice_pivot.columns
    ):

        paired = dice_pivot.dropna(
            subset=[
                "real",
                "imputed"
            ]
        )

        if not paired.empty:

            dice_difference = (
                paired["real"]
                -
                paired["imputed"]
            )

            print()
            print("=" * 70)
            print("PAIRED REAL VS IMPUTED")
            print("=" * 70)

            print(
                "Paired subjects:",
                len(paired)
            )

            print(
                "Mean real Dice:",
                f"{paired['real'].mean():.4f}"
            )

            print(
                "Mean imputed Dice:",
                f"{paired['imputed'].mean():.4f}"
            )

            print(
                "Mean Dice difference:",
                f"{dice_difference.mean():.4f}"
            )

            print(
                "Median Dice difference:",
                f"{dice_difference.median():.4f}"
            )

    if {
        "real",
        "imputed"
    }.issubset(
        uncertainty_pivot.columns
    ):

        paired_u = uncertainty_pivot.dropna(
            subset=[
                "real",
                "imputed"
            ]
        )

        if not paired_u.empty:

            uncertainty_difference = (
                paired_u["real"]
                -
                paired_u["imputed"]
            )

            print(
                "Mean tumor uncertainty difference:",
                f"{uncertainty_difference.mean():.4f}"
            )

            print(
                "Median tumor uncertainty difference:",
                f"{uncertainty_difference.median():.4f}"
            )

    print()
    print(
        "CSV saved to:"
    )

    print(
        csv_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    check_cuda()

    cases = discover_cases()

    if len(cases) == 0:

        raise RuntimeError(
            "No paired subjects found.\n"
            "Check REAL_ROOT, IMPUTED_ROOT "
            "and MODALITY."
        )

    predictor = load_model()

    results = []

    total = len(cases)

    for idx, case in enumerate(
        cases,
        start=1
    ):

        case_id = case[
            "case_id"
        ]

        print()
        print()
        print(
            "#" * 70
        )

        print(
            f"SUBJECT {idx}/{total}: "
            f"{case_id}"
        )

        print(
            "#" * 70
        )

        try:

            # ------------------------------------------------
            # LOAD GT
            # ------------------------------------------------

            gt_nii, gt_array = (
                load_nifti(
                    case["mask"]
                )
            )

            gt_binary = (
                make_gt_binary(
                    gt_array
                )
            )

            if gt_binary.sum() == 0:

                print(
                    f"[SKIP] {case_id}: "
                    "empty GT."
                )

                continue

            # ------------------------------------------------
            # CHECK REAL / IMPUTED GEOMETRY
            # ------------------------------------------------

            real_nii, _ = (
                load_nifti(
                    case["real"]
                )
            )

            imputed_nii, _ = (
                load_nifti(
                    case["imputed"]
                )
            )

            same_geometry(
                gt_nii,
                real_nii,
                "Real MRI"
            )

            same_geometry(
                gt_nii,
                imputed_nii,
                "Imputed MRI"
            )

            # ------------------------------------------------
            # PROMPT
            # ------------------------------------------------

            prompt_z, original_box = (
                get_prompt_slice_and_box(
                    gt_binary
                )
            )

            print()
            print(
                "GT shape:",
                gt_binary.shape
            )

            print(
                "GT voxels:",
                int(gt_binary.sum())
            )

            print(
                "Prompt slice:",
                prompt_z
            )

            print(
                "Oracle box:",
                original_box.tolist()
            )

            print()

            # ------------------------------------------------
            # REAL
            # ------------------------------------------------

            real_result = (
                process_modality(
                    predictor=predictor,
                    case=case,
                    modality="real",
                    mri_path=case["real"],
                    gt_nii=gt_nii,
                    gt_binary=gt_binary,
                    prompt_z=prompt_z,
                    original_box=original_box
                )
            )

            results.append(
                real_result
            )

            # ------------------------------------------------
            # IMPUTED
            # ------------------------------------------------

            imputed_result = (
                process_modality(
                    predictor=predictor,
                    case=case,
                    modality="imputed",
                    mri_path=case["imputed"],
                    gt_nii=gt_nii,
                    gt_binary=gt_binary,
                    prompt_z=prompt_z,
                    original_box=original_box
                )
            )

            results.append(
                imputed_result
            )

            # ------------------------------------------------
            # SAVE PROGRESS IMMEDIATELY
            # ------------------------------------------------

            pd.DataFrame(
                results
            ).to_csv(
                OUTPUT_DIR
                /
                "segmentation_results.csv",
                index=False
            )

            print()
            print(
                f"FINISHED {case_id}"
            )

            print(
                f"Real Dice: "
                f"{real_result['dice']:.4f}"
            )

            print(
                f"Imputed Dice: "
                f"{imputed_result['dice']:.4f}"
            )

            print(
                f"Dice difference: "
                f"{real_result['dice'] - imputed_result['dice']:.4f}"
            )

        except Exception as exc:

            print()
            print(
                f"[FAILED] {case_id}"
            )

            print(
                str(exc)
            )

            traceback.print_exc()

            # Continue with next subject
            continue

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    if results:

        analyze_results(
            results
        )

    else:

        print(
            "No subjects completed."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()