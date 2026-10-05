"""Run MTEB retrieval tasks on local (unpushed) checkpoints.

``mteb.get_model`` only resolves Hub ids or registered models. This module
builds the model wrapper directly so a local ColBERT (PyLate) or dense
(SentenceTransformer) checkpoint can be scored with ``mteb.evaluate``.
"""

from __future__ import annotations

from pathlib import Path

# English retrieval tasks from MTEB(Medical, v1) plus R2MED.
MEDICAL_RETRIEVAL_TASKS: list[str] = [
    "NFCorpus",
    "SciFact",
    "TRECCOVID",
    "MedicalQARetrieval",
    "CUREv1",
    "PublicHealthQA",
]
R2MED_TASKS: list[str] = [
    "R2MEDBiologyRetrieval",
    "R2MEDBioinformaticsRetrieval",
    "R2MEDMedicalSciencesRetrieval",
    "R2MEDMedXpertQAExamRetrieval",
    "R2MEDMedQADiagRetrieval",
    "R2MEDPMCTreatmentRetrieval",
    "R2MEDPMCClinicalRetrieval",
    "R2MEDIIYiClinicalRetrieval",
]


def load_local_model(model_path: str | Path, arch: str, name: str):
    """Wrap a local checkpoint so ``mteb.evaluate`` accepts it."""
    from mteb.models.model_meta import ModelMeta, ScoringFunction

    path = str(model_path)
    if arch == "colbert":
        from mteb.models.model_implementations import pylate_models
        from mteb.models.model_implementations.pylate_models import MultiVectorModel

        # mteb 2.12.30 calls create_dataloader(ds, task_metadata, ...) but
        # task_metadata is keyword-only there; accept the positional form.
        _orig = pylate_models.create_dataloader

        def _create_dataloader(dataset, *args, **kwargs):
            if args:
                kwargs.setdefault("task_metadata", args[0])
            return _orig(dataset, **kwargs)

        pylate_models.create_dataloader = _create_dataloader

        meta = ModelMeta(
            loader=MultiVectorModel,
            name=name,
            model_type=["late-interaction"],
            languages=["eng-Latn"],
            open_weights=True,
            revision="local",
            release_date=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            license=None,
            similarity_fn_name=ScoringFunction.MAX_SIM,
            framework=["PyLate"],
            reference=None,
            use_instructions=False,
            public_training_code=None,
            public_training_data=None,
            training_datasets=None,
        )
        model = MultiVectorModel(path)
        model.mteb_model_meta = meta
        return model
    if arch == "dense":
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(path)
    raise ValueError(f"unknown arch: {arch}")


def main_scores(results) -> dict[str, float]:
    """Flatten an mteb ModelResult into ``{task: main_score}``."""
    out: dict[str, float] = {}
    for tr in results.task_results:
        out[tr.task_name] = float(tr.get_score())
    return out
