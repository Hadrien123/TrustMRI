"""Segment every imputed run with team 2's MedSAM2 module (segmentation_module(1).py), N_PROMPTS prompts per run.

Masks, as trust_pipeline_t1n.py reads them:
    <IMPUTED_DIR>/<class>/<subject>/<MOD>/segmentation/runNN/<subject>_imputed_prompt_masks/prompt_KK.nii.gz
Metrics of each run's consensus mask: <IMPUTED_DIR>/segmentation_<MOD>.csv
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
IMPUTED_DIR = HERE / "results/trust_t1n"  # output_dir of config_trust_t1n.yaml
GT_DIR = HERE / "test"  # its data_dir
MEDSAM2 = HERE / "work/MedSAM2"  # cloned by the notebook, with its checkpoint
MOD, N_PROMPTS = "t1n", 5

sys.path.append(str(MEDSAM2))
# The module's file name is not importable: load it from its path
spec = importlib.util.spec_from_file_location("segmentation_module", HERE / "segmentation_module(1).py")
seg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seg)
seg.CONFIG_PATH = MEDSAM2 / "sam2/configs/sam2.1_hiera_t512.yaml"
seg.CHECKPOINT = MEDSAM2 / "checkpoints/MedSAM2_latest.pt"
seg.MODALITY, seg.N_PROMPTS = MOD, N_PROMPTS
predictor = seg.load_model()

rows = []
for mod_dir in sorted(IMPUTED_DIR.glob(f"*/*/{MOD}")):  # <class>/<subject>/<MOD>
    sid = mod_dir.parent.name
    gt_nii, gt = seg.load_nifti(next(GT_DIR.glob(f"*/{sid}/{sid}-seg.nii*")))
    gt_binary = seg.make_gt_binary(gt)
    prompt_z, box = seg.get_prompt_slice_and_box(gt_binary)
    case = {"case_id": sid, "category": mod_dir.parent.parent.name}
    for run_path in sorted((mod_dir / "imputed").glob("*.nii*")):
        run = run_path.name.split("-")[-1].split(".")[0]  # runNN
        seg.OUTPUT_DIR = mod_dir / "segmentation" / run
        # Same modality label for every run: the module seeds the prompt boxes from it, so all runs get the same boxes
        rows.append({"run": run, **seg.process_modality(predictor, case, "imputed", str(run_path),
                                                        gt_nii, gt_binary, prompt_z, box)})
pd.DataFrame(rows).to_csv(IMPUTED_DIR / f"segmentation_{MOD}.csv", index=False)  # written once all runs are done
