#!/usr/bin/env python3
"""Configure and build the SmartGC C++ simulator via CMake.

`simulator/CMakeLists.txt` is authoritative; this script only locates a usable
CMake and generator so the same command works on a fresh machine:

    python scripts/build_simulator.py            # configure + build
    python scripts/build_simulator.py --test     # ... then run the unit tests
    python scripts/build_simulator.py --clean    # discard the build tree first

On Windows, CMake installed through `winget install Kitware.CMake` is not on
PATH until the shell is restarted, so the usual install locations are searched
as well.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "simulator"
BUILD_DIR = SOURCE_DIR / "build"


def find_cmake() -> str:
    """Return a path to a cmake executable, or exit with an actionable message."""
    found = shutil.which("cmake")
    if found:
        return found

    candidates: list[Path] = []
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        base = Path(local_appdata)
        candidates.append(base / "Microsoft" / "WinGet" / "Links" / "cmake.exe")
        packages = base / "Microsoft" / "WinGet" / "Packages"
        if packages.is_dir():
            candidates.extend(sorted(packages.glob("Kitware.CMake_*/*/bin/cmake.exe")))
    candidates.append(Path("C:/Program Files/CMake/bin/cmake.exe"))

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    sys.exit(
        "CMake was not found.\n"
        "  Windows: winget install Kitware.CMake  (then restart the shell)\n"
        "  Debian/Ubuntu: sudo apt install cmake build-essential\n"
        "  macOS: brew install cmake"
    )


def choose_generator() -> list[str]:
    """Pick an explicit generator only where CMake's default would not work.

    On Windows the default generator is Visual Studio. When the available
    compiler is the MSYS2/MinGW GCC (as on the reference machine) that default
    fails, so select MinGW Makefiles whenever mingw32-make is present and cl.exe
    is not. Everywhere else CMake's own default is correct.
    """
    if os.name != "nt":
        return []
    has_mingw_make = shutil.which("mingw32-make") or shutil.which("make")
    has_msvc = shutil.which("cl")
    if has_mingw_make and not has_msvc:
        generator = "MinGW Makefiles" if shutil.which("mingw32-make") else "Unix Makefiles"
        return ["-G", generator]
    return []


def run(command: list[str]) -> None:
    print("+ " + " ".join(str(part) for part in command), flush=True)
    result = subprocess.run(command)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean", action="store_true", help="delete the build tree first")
    parser.add_argument("--test", action="store_true", help="run the unit test suite after building")
    parser.add_argument("--build-type", default="Release",
                        choices=["Debug", "Release", "RelWithDebInfo", "MinSizeRel"])
    args = parser.parse_args()

    cmake = find_cmake()
    print(f"Using CMake: {cmake}")

    if args.clean and BUILD_DIR.exists():
        print(f"Removing {BUILD_DIR}")
        shutil.rmtree(BUILD_DIR)

    configure = [cmake, "-S", str(SOURCE_DIR), "-B", str(BUILD_DIR),
                 f"-DCMAKE_BUILD_TYPE={args.build_type}"]
    configure.extend(choose_generator())
    run(configure)
    run([cmake, "--build", str(BUILD_DIR), "--config", args.build_type])

    if args.test:
        suffix = ".exe" if os.name == "nt" else ""
        for candidate in (BUILD_DIR / f"simulator_tests{suffix}",
                          BUILD_DIR / args.build_type / f"simulator_tests{suffix}"):
            if candidate.is_file():
                # Run from the repository root so the suite can reach config/config.yaml.
                print(f"+ {candidate}")
                result = subprocess.run([str(candidate)], cwd=str(REPO_ROOT))
                return result.returncode
        sys.exit(f"Built successfully, but simulator_tests was not found under {BUILD_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
