"""Runtime settings. Every field is overridable from the environment as CERTUS_<NAME>."""
import glob
import os

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the api/ directory
REPO = os.path.dirname(ROOT)                                         # the checkout root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CERTUS_", extra="ignore")

    db_url: str = f"sqlite:///{ROOT}/certus.db"
    storage_dir: str = f"{ROOT}/storage"

    # Resolved from this file's location so a fresh clone runs anywhere. Every one is still
    # overridable as CERTUS_TORCH_ROOT / CERTUS_SCRIPTS_ROOT / CERTUS_RUNS_DIR.
    torch_root: str = f"{REPO}/torch"            # CertusNet + config live here
    scripts_root: str = f"{REPO}/scripts"        # retina.py (FOV normalisation)
    runs_dir: str = f"{REPO}/runs_torch"
    checkpoint: str = ""                         # blank -> newest best.pt under runs_dir
    trust_json: str = ""                         # blank -> trust.json beside the checkpoint

    device: str = "auto"                         # auto | cuda | cpu
    batch_tiles: int = 10                        # one eye = 3x3 tiles + global view

    # demo auth: key:role pairs. Roles are technician | ophthalmologist | admin.
    api_keys: str = "demo-tech:technician,demo-doc:ophthalmologist,demo-admin:admin"
    sla_hours: int = 48
    urgent_grades: str = "3,4"                   # sight-threatening -> jump the queue

    def resolve_checkpoint(self) -> str:
        if self.checkpoint:
            return self.checkpoint
        found = sorted(glob.glob(f"{self.runs_dir}/train_*/best.pt"), key=os.path.getmtime)
        if not found:
            raise RuntimeError(f"no best.pt under {self.runs_dir}; train first or set CERTUS_CHECKPOINT")
        return found[-1]

    def resolve_trust(self) -> str:
        if self.trust_json:
            return self.trust_json
        beside = os.path.join(os.path.dirname(self.resolve_checkpoint()), "trust.json")
        return beside if os.path.exists(beside) else ""

    def keys(self) -> dict:
        out = {}
        for pair in self.api_keys.split(","):
            if ":" in pair:
                k, r = pair.split(":", 1)
                out[k.strip()] = r.strip()
        return out


settings = Settings()
