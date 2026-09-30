"""
Master Orchestrator Engine for ASTRA Synthetic Dataset Generation (Engine 8).
Coordinates Engines 1 through 7 into a single, scalable, reproducible, and verifiable pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Generator, Sequence
import numpy as np
import yaml
from tqdm import tqdm

from astra_synthetic.payload.generator import PayloadGenerator
from astra_synthetic.framing.generator import FrameGenerator
from astra_synthetic.fec.generator import FECGenerator
from astra_synthetic.interleaving.generator import InterleaverGenerator
from astra_synthetic.modulation.generator import ModulationGenerator
from astra_synthetic.channel.generator import ChannelGenerator
from astra_synthetic.capture.generator import CaptureGenerator

from .models import DatasetRecord, DatasetMetadata
from .seed_manager import SeedManager
from .chain_ids import (
    make_chain_id,
    make_dataset_record_id,
    make_payload_id,
    make_frame_id,
    make_fec_id,
    make_interleaver_id,
    make_modulation_id,
    make_channel_id,
    make_capture_id,
)
from .balancing import ProfileSampler
from .splits import assign_split_by_chain_id, verify_no_split_leakage
from .truth_builder import build_full_truth_document
from .visible_metadata import build_visible_metadata_document
from .validation import validate_full_synthetic_chain
from .manifests import build_master_manifest, build_split_manifests, build_task_specific_manifests
from .statistics import compute_dataset_statistics, generate_dataset_reports
from .serializers import export_blind_test_package, export_evaluation_truth_package

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "dataset_config.yaml"


class ASTRASyntheticDatasetGenerator:
    """Master orchestrator connecting Engines 1-7 into complete reproducible datasets."""

    GENERATOR_VERSIONS = {
        "payload": "1.0.0",
        "framing": "1.0.0",
        "fec": "1.0.0",
        "interleaving": "1.0.0",
        "modulation": "1.0.0",
        "channel": "1.0.0",
        "capture": "1.0.0",
        "orchestrator": "1.0.0",
    }

    def __init__(self, config: dict[str, Any] | Path | str | None = None):
        if config is None:
            if DEFAULT_CONFIG_PATH.exists():
                with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f) or {}
            else:
                self.config = {}
        elif isinstance(config, (str, Path)):
            with open(config, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
        else:
            self.config = config

        ds_cfg = self.config.get("dataset", {})
        self.dataset_name = ds_cfg.get("name", "astra_synthetic_v1")
        self.version = ds_cfg.get("version", "1.0.0")
        self.master_seed = int(ds_cfg.get("master_seed", 42))
        self.total_records = int(ds_cfg.get("total_records", 100))
        self.output_root = Path(ds_cfg.get("output_root", "./output/astra_synthetic_v1"))

        split_cfg = ds_cfg.get("split", {})
        self.train_ratio = float(split_cfg.get("train", 0.70))
        self.val_ratio = float(split_cfg.get("validation", 0.15))
        self.test_ratio = float(split_cfg.get("test", 0.15))

        self.seed_mgr = SeedManager(master_seed=self.master_seed)
        self.sampler = ProfileSampler(self.config)

        # Initialize upstream engines
        self.payload_gen = PayloadGenerator()
        self.frame_gen = FrameGenerator()
        self.fec_gen = FECGenerator()
        self.int_gen = InterleaverGenerator()
        self.mod_gen = ModulationGenerator()
        self.chan_gen = ChannelGenerator()
        self.cap_gen = CaptureGenerator()

        self._records: list[DatasetRecord] = []

    def generate_one(
        self,
        index: int,
        output_dir: Path | str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> DatasetRecord:
        """Execute the full 7-stage chain for a single dataset record."""
        target_root = Path(output_dir) if output_dir else self.output_root
        cap_dir = target_root / "captures"
        vis_dir = target_root / "visible_metadata"
        truth_dir = target_root / "truth"

        cap_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)
        truth_dir.mkdir(parents=True, exist_ok=True)

        # 1. IDs & Seeds
        chain_id = make_chain_id(index)
        ds_rec_id = make_dataset_record_id(index)
        seeds = self.seed_mgr.get_chain_seeds(index)
        rng = np.random.default_rng(seeds.master_chain_seed)

        # 2. Sample or override parameters
        params = self.sampler.sample_chain_parameters(rng, balanced_index=index)
        if overrides:
            for k, v in overrides.items():
                if k in params and isinstance(params[k], dict) and isinstance(v, dict):
                    params[k].update(v)
                else:
                    params[k] = v

        # 3. Determine Split
        split_name = assign_split_by_chain_id(
            source_chain_id=chain_id,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
        )

        # STEP 1: Payload (Engine 1)
        p_cfg = params["payload"]
        payload_rec = self.payload_gen.generate(
            payload_type=p_cfg.get("payload_type", "counter"),
            bit_length=p_cfg.get("bit_length", 512),
            seed=seeds.payload_seed,
        )

        # STEP 2: Frame (Engine 2)
        frame_rec = self.frame_gen.generate(
            payload_rec,
            seed=seeds.frame_seed,
        )

        # STEP 3: FEC Encoding (Engine 3)
        f_cfg = params["fec"]
        fec_rec = self.fec_gen.encode(
            frame_rec,
            profile_name=f_cfg.get("profile", "none"),
        )

        # STEP 4: Interleaver (Engine 4)
        i_cfg = params["interleaving"]
        int_rec = self.int_gen.interleave(
            fec_rec,
            profile_name=i_cfg.get("profile", "none"),
            seed=seeds.interleaver_seed,
        )

        # STEP 5: Modulation / Clean IQ (Engine 5)
        m_cfg = params["modulation"]
        mod_rec = self.mod_gen.modulate(
            int_rec,
            profile_name=m_cfg.get("profile", "qpsk_rrc"),
            symbol_rate=m_cfg.get("symbol_rate", 24000.0),
            sample_rate=m_cfg.get("sample_rate", 192000.0),
        )

        # STEP 6: Channel Impairments (Engine 6)
        c_cfg = params["channel"]
        is_clean_control = (c_cfg.get("profile") == "clean" or c_cfg.get("difficulty") == "clean")
        chan_rec = self.chan_gen.apply(
            mod_rec,
            profile_name=c_cfg.get("profile", "combined_medium"),
            seed=seeds.channel_seed,
        )

        # STEP 7: Sampling / Capture Representation (Engine 7)
        cap_cfg = params["capture"]
        cap_rec = self.cap_gen.write(
            chan_rec,
            profile_name=cap_cfg.get("profile", "raw_f32_le_iq"),
            output_dir=cap_dir,
            seed=seeds.capture_seed,
        )

        # STEP 8: Validation
        is_valid, val_report = validate_full_synthetic_chain(
            payload_rec=payload_rec,
            frame_rec=frame_rec,
            fec_rec=fec_rec,
            int_rec=int_rec,
            mod_rec=mod_rec,
            chan_rec=chan_rec,
            cap_rec=cap_rec,
            is_clean_control=is_clean_control,
        )

        # STEP 9: Build & Save Metadata Documents
        truth_doc = build_full_truth_document(
            dataset_record_id=ds_rec_id,
            source_chain_id=chain_id,
            split=split_name,
            payload_rec=payload_rec,
            frame_rec=frame_rec,
            fec_rec=fec_rec,
            int_rec=int_rec,
            mod_rec=mod_rec,
            chan_rec=chan_rec,
            cap_rec=cap_rec,
            generator_versions=self.GENERATOR_VERSIONS,
        )
        truth_path = truth_dir / f"{cap_rec.capture_record_id}.truth.json"
        with open(truth_path, "w", encoding="utf-8") as f:
            json.dump(truth_doc, f, indent=2)

        vis_doc = build_visible_metadata_document(
            cap_rec=cap_rec,
            expose_sample_rate=cap_rec.sample_rate_visible,
        )
        vis_path = vis_dir / f"{cap_rec.capture_record_id}.capture.json"
        with open(vis_path, "w", encoding="utf-8") as f:
            json.dump(vis_doc, f, indent=2)

        # Construct concise dataset record
        labels = {
            "modulation_type": mod_rec.modulation_type,
            "modulation_family": mod_rec.modulation_family,
            "symbol_rate": mod_rec.symbol_rate,
            "sample_rate": mod_rec.sample_rate,
            "snr_db": chan_rec.snr_db_measured,
            "cfo_hz": chan_rec.cfo_hz,
            "fec_family": fec_rec.fec_type,
            "fec_profile": fec_rec.fec_profile or fec_rec.fec_type,
            "code_rate": fec_rec.effective_code_rate,
            "interleaver_family": int_rec.interleaver_type,
            "interleaver_profile": int_rec.profile_name or int_rec.interleaver_type,
            "difficulty": c_cfg.get("difficulty", "medium"),
            "file_format": cap_rec.file_format,
            "storage_dtype": cap_rec.storage_dtype,
            "endianness": cap_rec.endianness,
            "iq_order": cap_rec.iq_order,
            "sync_start": frame_rec.sync_start,
            "sync_length": frame_rec.sync_length,
            "header_start": frame_rec.header_start,
            "header_length": frame_rec.header_length,
            "payload_start": frame_rec.payload_start,
            "payload_length": frame_rec.payload_length,
            "crc_start": frame_rec.crc_start,
            "crc_length": frame_rec.crc_length,
        }

        record = DatasetRecord(
            dataset_record_id=ds_rec_id,
            source_chain_id=chain_id,
            payload_id=payload_rec.payload_id,
            frame_id=frame_rec.frame_id,
            fec_record_id=fec_rec.fec_record_id,
            interleaver_record_id=int_rec.interleaver_record_id,
            modulation_record_id=mod_rec.modulation_record_id,
            channel_record_id=chan_rec.channel_record_id,
            capture_record_id=cap_rec.capture_record_id,
            capture_path=str(cap_rec.file_path),
            visible_metadata_path=str(vis_path),
            truth_metadata_path=str(truth_path),
            split=split_name,
            labels=labels,
            generation_parameters=params,
            source_seed=seeds.master_chain_seed,
            generator_versions=self.GENERATOR_VERSIONS,
            valid=is_valid,
            validation_results=val_report,
        )

        return record

    def generate_dataset(
        self,
        count: int | None = None,
        output_root: Path | str | None = None,
        show_progress: bool = True,
    ) -> list[DatasetRecord]:
        """Stream and generate a complete synthetic dataset release."""
        n_records = count if count is not None else self.total_records
        root_dir = Path(output_root) if output_root else self.output_root
        root_dir.mkdir(parents=True, exist_ok=True)

        manifests_dir = root_dir / "manifests"
        reports_dir = root_dir / "reports"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        records: list[DatasetRecord] = []

        iterator = range(1, n_records + 1)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Generating {self.dataset_name}", unit="signal")

        for idx in iterator:
            rec = self.generate_one(index=idx, output_dir=root_dir)
            records.append(rec)

        self._records = records

        # 1. Build Manifests
        master_manifest_path = manifests_dir / "all.csv"
        build_master_manifest(records, master_manifest_path)
        split_paths = build_split_manifests(records, manifests_dir)
        task_paths = build_task_specific_manifests(records, manifests_dir)

        # 2. Build Reports
        report_paths = generate_dataset_reports(records, reports_dir)

        # 3. Build Global Dataset Metadata
        stats = compute_dataset_statistics(records)
        cfg_str = json.dumps(self.config, sort_keys=True)
        cfg_sha = hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()

        all_manifest_paths = {
            "all": str(master_manifest_path),
            **{k: str(v) for k, v in split_paths.items()},
            **{k: str(v) for k, v in task_paths.items()},
        }

        meta = DatasetMetadata(
            dataset_name=self.dataset_name,
            version=self.version,
            creation_time_utc=datetime.now(timezone.utc).isoformat(),
            master_seed=self.master_seed,
            total_requested=n_records,
            total_valid=stats["valid_records"],
            total_failed=stats["failed_records"],
            split_ratios={"train": self.train_ratio, "validation": self.val_ratio, "test": self.test_ratio},
            split_counts=stats["splits"],
            class_distributions=stats["distributions"],
            generator_versions=self.GENERATOR_VERSIONS,
            configuration_sha256=cfg_sha,
            storage_root=str(root_dir),
            manifest_paths=all_manifest_paths,
            summary_metrics={
                "snr_metrics": stats["snr_metrics"],
                "cfo_metrics": stats["cfo_metrics"],
            },
        )

        meta_path = root_dir / "dataset_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta.to_dict(), f, indent=2)

        return records

    def export_blind_test_set(
        self,
        output_dir: Path | str,
        split_name: str = "test",
    ) -> Path:
        """Export sanitized test partition package without labels for blind receiver evaluation."""
        if not self._records:
            raise RuntimeError("No records generated yet. Call generate_dataset() first.")
        test_records = [r for r in self._records if (r.split or "test") == split_name]
        return export_blind_test_package(test_records, output_dir=output_dir)

    def export_evaluation_truth(
        self,
        output_dir: Path | str,
        split_name: str = "test",
    ) -> Path:
        """Export secret truth package for scoring blind receiver outputs."""
        if not self._records:
            raise RuntimeError("No records generated yet. Call generate_dataset() first.")
        test_records = [r for r in self._records if (r.split or "test") == split_name]
        return export_evaluation_truth_package(test_records, output_dir=output_dir)
