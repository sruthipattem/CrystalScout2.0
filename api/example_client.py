"""Call the CrystalScout API from Python, using only the standard library.

Start the API first (run_api.bat), then:   python -m api.example_client
"""
import json
import os
import sys
import urllib.request
import uuid
from pathlib import Path

BASE = "http://127.0.0.1:8000"
USERNAME = "scientist"
PASSWORD = os.environ.get("CRYSTALSCOUT_PW_SCIENTIST")


def call(method, path, token=None, body=None, files=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    data = None
    if files:  # one file, multipart/form-data
        boundary = uuid.uuid4().hex
        name, content = files
        data = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
                f"Content-Type: chemical/x-cif\r\n\r\n").encode() + content + f"\r\n--{boundary}--\r\n".encode()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    elif body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def main():
    if not PASSWORD:
        sys.exit("Set CRYSTALSCOUT_PW_SCIENTIST before running this example.")
    print("1. Health:", call("GET", "/health")["status"])
    token = call("POST", "/auth/login", body={"username": USERNAME, "password": PASSWORD})["access_token"]
    print("2. Signed in, got a token")

    cif = Path(__file__).resolve().parent.parent / "data" / "obelix" / "cifs" / "jqc.cif"
    r = call("POST", "/predict/cif", token, files=(cif.name, cif.read_bytes()))
    print(f"3. {r['formula']}: score {r['probability']:.2f} ({r['verdict']}) - {r['why']}")

    s = call("POST", "/screen", token, body={"band_gap_min": 2.0, "top": 5})
    print(f"4. Screen: {s['summary']['n_stage_a']} materials passed Stage A. Top 5:")
    for row in s["results"]:
        print(f"   #{row['rank']}  {row['formula']:<14} score {row['p_logreg']:.2f}")


if __name__ == "__main__":
    try:
        main()
    except OSError as exc:
        sys.exit(f"Could not reach the API at {BASE}. Start it with run_api.bat first. ({exc})")
