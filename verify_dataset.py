
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parent
checksum_file = root / "SHA256SUMS.txt"

checked = 0
passed = 0
failed = []
missing = []

with checksum_file.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        expected, relative_path = line.split(maxsplit=1)
        relative_path = relative_path.lstrip("*")
        file_path = root / relative_path

        if not file_path.is_file():
            missing.append(relative_path)
            continue

        digest = hashlib.sha256()
        with file_path.open("rb") as data:
            for chunk in iter(lambda: data.read(1024 * 1024), b""):
                digest.update(chunk)

        checked += 1
        if digest.hexdigest().lower() == expected.lower():
            passed += 1
        else:
            failed.append(relative_path)

print(f"\nFiles checked: {checked}")
print(f"Checksum matches: {passed}")
print(f"Checksum mismatches: {len(failed)}")
print(f"Missing files: {len(missing)}")

if failed:
    print("\nMISMATCHED FILES:")
    print("\n".join(failed[:30]))

if missing:
    print("\nMISSING FILES:")
    print("\n".join(missing[:30]))

if checked > 0 and passed == checked and not missing and not failed:
    print("\nSUCCESS: All files listed in SHA256SUMS.txt are verified!")
else:
    print("\nVerification incomplete or errors found. Review the results above.")
