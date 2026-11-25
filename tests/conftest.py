"""Configure isolated test-only accounts, independent of the developer environment."""
import os

for _u in ("ADMIN", "SCIENTIST", "PARTNER", "EXPO"):
    os.environ[f"CRYSTALSCOUT_PW_{_u}"] = f"test-only-{_u.lower()}-password"
os.environ["CRYSTALSCOUT_NO_AUDIT"] = "1"
