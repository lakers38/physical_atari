# Copyright 2025 Keen Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run to diagnose and potentially resolve performance issues with the host system.
# This script is intended to be run on the host system, not within a docker container.
# sudo check_performance.py

import glob
import os
import subprocess
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Check:
    name: str
    ok: bool
    message: str
    fix_script: Optional[str] = None
    validate: Optional[Callable[[], bool]] = None
    info_only: bool = False  # informational only, no pass/fail


def run_cmd(cmd, timeout=3):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"


def check_cpu_frequencies():
    cpu_dir = "/sys/devices/system/cpu"
    min_freqs = []
    max_freqs = []
    cur_freqs = []

    cpu_dirs = [d for d in os.listdir(cpu_dir) if d.startswith("cpu") and d[3:].isdigit()]

    def read_int_file(path):
        try:
            with open(path) as f:
                return int(f.read().strip())
        except Exception:
            return None

    for cpu in cpu_dirs:
        base = os.path.join(cpu_dir, cpu, "cpufreq")
        min_freqs.append(read_int_file(os.path.join(base, "scaling_min_freq")))
        max_freqs.append(read_int_file(os.path.join(base, "scaling_max_freq")))
        cur_freqs.append(read_int_file(os.path.join(base, "scaling_cur_freq")))

    def to_mhz(freq_list):
        return [f // 1000 if f else None for f in freq_list]

    min_mhz = to_mhz(min_freqs)
    max_mhz = to_mhz(max_freqs)
    cur_mhz = to_mhz(cur_freqs)

    summary = f"Current: {cur_mhz}\n  Min: {min_mhz}\n  Max: {max_mhz}"
    return True, summary


def check_cpu_governor() -> tuple[bool, str]:
    governors = []
    cpu_dir = "/sys/devices/system/cpu"
    cpu_dirs = [d for d in os.listdir(cpu_dir) if d.startswith("cpu") and d[3:].isdigit()]
    for cpu in cpu_dirs:
        gov_path = os.path.join(cpu_dir, cpu, "cpufreq", "scaling_governor")
        if os.path.isfile(gov_path):
            with open(gov_path) as f:
                governors.append(f.read().strip())
        else:
            governors.append("N/A")
    summary = Counter(governors)
    all_perf = all(g == "performance" for g in governors if g != "N/A")
    return all_perf, f"{dict(summary)}"


def check_cpu_online() -> tuple[bool, str]:
    try:
        with open("/sys/devices/system/cpu/online") as f:
            online_cpus = f.read().strip()
        online = sum(int(r.split("-")[1]) - int(r.split("-")[0]) + 1 if "-" in r else 1 for r in online_cpus.split(","))
        total = len([d for d in os.listdir("/sys/devices/system/cpu") if d.startswith("cpu") and d[3:].isdigit()])
        return online == total, f"{online} / {total} CPUs online"
    except Exception as e:
        return False, f"ERROR: {e}"


def check_nvidia_persistence():
    out = run_cmd(["nvidia-smi", "-q"])
    for line in out.splitlines():
        if "Persistence Mode" in line:
            status = line.split(":")[1].strip()
            return status.lower() == "enabled", f"Persistence Mode: {status}"
    return False, "Persistence Mode: Unknown"


def check_nvidia_powerd():
    try:
        active = run_cmd(["systemctl", "is-active", "nvidia-powerd.service"])
        enabled = run_cmd(["systemctl", "is-enabled", "nvidia-powerd.service"])
        return active == "active" and enabled == "enabled", f"Active: {active}, Enabled: {enabled}"
    except Exception as e:
        return False, f"ERROR: {e}"


def check_nvidia_power_limit():
    out = run_cmd(["nvidia-smi", "-q", "-d", "POWER"])
    fields = {"Current Power Limit": None, "Requested Power Limit": None, "Max Power Limit": None}
    for line in out.splitlines():
        if ":" in line:
            key, val = map(str.strip, line.split(":", 1))
            if key in fields and val != "N/A":
                fields[key] = float(val.split()[0])
    ok = (
        fields["Current Power Limit"] == fields["Requested Power Limit"]
        and fields["Current Power Limit"]
        and fields["Current Power Limit"] >= 100.0
    )
    msg = ", ".join(f"{k}: {v} W" for k, v in fields.items())
    return ok, msg


def check_nvidia_pstate():
    out = run_cmd(["nvidia-smi", "-q"])
    for line in out.splitlines():
        if "Performance State" in line:
            return True, line.strip()
    return False, "Unknown P-State"


def check_nvidia_gpu_clocks_and_throttling():
    out = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=clocks.sm,clocks.max.sm,clocks_throttle_reasons.active",
            "--format=csv,noheader,nounits",
        ]
    )
    try:
        sm_clock, sm_max, throttle_hex = map(str.strip, out.split(","))
        sm_clock, sm_max, throttle = int(sm_clock), int(sm_max), int(throttle_hex, 16)
        idle_only = throttle == 0x1
        sm_ok = True if idle_only else sm_clock >= 0.95 * sm_max
        ok = sm_ok and idle_only
        return ok, f"SM Clock: {sm_clock} MHz, Max: {sm_max} MHz, Throttle: 0x{throttle:04x}"
    except Exception as e:
        return False, f"ERROR parsing GPU clock: {e}"


def check_thermal_limits():
    try:
        gpu_out = run_cmd(["nvidia-smi", "-q", "-d", "TEMPERATURE"])
        gpu_temp = next(
            (int(line.split(":")[1].split()[0]) for line in gpu_out.splitlines() if "Target Temperature" in line), None
        )
    except Exception:
        gpu_temp = None

    cpu_paths = glob.glob("/sys/devices/platform/coretemp.*/hwmon/hwmon*/temp1_max")
    try:
        cpu_temp = int(open(cpu_paths[0]).read().strip()) / 1000 if cpu_paths else None
    except Exception:
        cpu_temp = None

    warn = any(t is not None and t < 80 for t in (cpu_temp, gpu_temp))
    return not warn, f"CPU limit: {cpu_temp}°C, GPU target: {gpu_temp}°C"


def get_effective_max_power(constraints):
    # keys like 'constraint_0_max_power_w', sorted by constraint index
    for key in sorted(constraints.keys()):
        val = constraints[key]
        if val is not None and val > 0:
            return val
    return None


def read_int_file(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None


def check_powercap_limits():
    import glob

    rapl_domains = glob.glob("/sys/class/powercap/intel-rapl*")
    results = {}

    for domain in rapl_domains:
        energy_path = os.path.join(domain, "energy_uj")
        if not os.path.isfile(energy_path):
            # Try nested subdirs (e.g. intel-rapl:0:0)
            subdomains = glob.glob(os.path.join(domain, "intel-rapl:*"))
            if subdomains:
                for sub in subdomains:
                    sub_energy_path = os.path.join(sub, "energy_uj")
                    if os.path.isfile(sub_energy_path):
                        energy = read_int_file(sub_energy_path)
                        constraints = {}
                        for i in range(10):  # max 10 constraints
                            c_path = os.path.join(sub, f"constraint_{i}_max_power_uw")
                            if os.path.isfile(c_path):
                                value = read_int_file(c_path)
                                if value is not None:
                                    constraints[f"constraint_{i}_max_power_w"] = value / 1e6
                        results[sub] = {"energy_j": energy / 1e6 if energy else None, **constraints}
            continue

        energy = read_int_file(energy_path)
        constraints = {}
        for i in range(10):
            c_path = os.path.join(domain, f"constraint_{i}_max_power_uw")
            if os.path.isfile(c_path):
                value = read_int_file(c_path)
                if value is not None:
                    constraints[f"constraint_{i}_max_power_w"] = value / 1e6
        results[domain] = {"energy_j": energy / 1e6 if energy else None, **constraints}

    if results:
        return True, results
    else:
        return False, {"error": "No intel-rapl domains with energy_uj found"}


def check_power_profile():
    profile_path = "/sys/firmware/acpi/platform_profile"
    try:
        if os.path.isfile(profile_path):
            with open(profile_path) as f:
                profile = f.read().strip().lower()
                return profile == "performance", f"ACPI platform profile: {profile}"
    except Exception:
        # fallback
        pass

    try:
        # try powerprofilesctl
        result = subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            profile = result.stdout.strip().lower()
            return profile == "performance", f"powerprofilesctl: {profile}"
        else:
            return False, f"powerprofilesctl returned non-zero: {result.stdout.strip()}"
    except FileNotFoundError:
        return False, "No power profile method available (no ACPI or powerprofilesctl)"
    except Exception as e:
        return False, f"Error reading power profile: {e}"


def main():
    checks: list[Check] = []

    ok, msg = check_cpu_frequencies()
    checks.append(Check(name="CPU Frequencies", ok=ok, message=msg, info_only=True))

    ok, msg = check_cpu_governor()
    checks.append(
        Check(
            name="CPU Governor",
            ok=ok,
            message=msg,
            fix_script="./performance/cpu-governor.sh",
            validate=lambda: check_cpu_governor()[0],
        )
    )

    ok, msg = check_cpu_online()
    checks.append(Check(name="CPU Online", ok=ok, message=msg, fix_script=None, validate=lambda: check_cpu_online()[0]))

    ok, msg = check_nvidia_persistence()
    checks.append(
        Check(
            name="NVIDIA Persistence Mode",
            ok=ok,
            message=msg,
            fix_script="./performance/nvidia-persistence.sh",
            validate=lambda: check_nvidia_persistence()[0],
        )
    )

    ok, msg = check_nvidia_powerd()
    checks.append(
        Check(
            name="NVIDIA PowerD",
            ok=ok,
            message=msg,
            fix_script="./performance/nvidia-powerd.sh",
            validate=lambda: check_nvidia_powerd()[0],
        )
    )

    ok, msg = check_nvidia_power_limit()
    checks.append(
        Check(
            name="NVIDIA Power Limit",
            ok=ok,
            message=msg,
            fix_script=None,
            validate=lambda: check_nvidia_power_limit()[0],
        )
    )

    ok, msg = check_nvidia_pstate()
    checks.append(Check(name="NVIDIA P-State", ok=ok, message=msg, fix_script=None, validate=None))

    ok, msg = check_nvidia_gpu_clocks_and_throttling()
    checks.append(
        Check(
            name="GPU Clocks & Throttling",
            ok=ok,
            message=msg,
            fix_script=None,
            validate=lambda: check_nvidia_gpu_clocks_and_throttling()[0],
        )
    )

    ok, msg = check_thermal_limits()
    checks.append(
        Check(name="Thermal Limits", ok=ok, message=msg, fix_script=None, validate=lambda: check_thermal_limits()[0])
    )

    ok, msg = check_power_profile()
    checks.append(
        Check(
            name="Power Profile",
            ok=ok,
            message=msg,
            fix_script="./performance/power-profile.sh",
            validate=lambda: check_power_profile()[0],
        )
    )

    num_tests = sum(1 for c in checks if not c.info_only)
    num_failures = sum(1 for c in checks if not c.ok and not c.info_only)

    print("\n---- Performance Check Summary ----")
    for c in checks:
        status = "OK" if c.ok else "FAIL"
        print(f"[{status}] {c.name}: {c.message}")

    if num_failures == 0:
        print(f"\nAll {num_tests}/{num_tests} checks passed.")
    else:
        print(f"\n{num_tests - num_failures}/{num_tests} checks passed. {num_failures} failed.")

    remaining_issues = []
    for c in checks:
        if c.ok or not c.fix_script:
            continue
        resp = input(f"\nFix '{c.name}'? (y/n): ").strip().lower()
        if resp == "y":
            print(f"Running {c.fix_script}...")
            os.system(f"bash {c.fix_script}")
            if c.validate and c.validate():
                print(f"Fix verified for {c.name}")
            else:
                remaining_issues.append(c)
                print(f"Fix did not resolve issue with {c.name}")

    if remaining_issues:
        print(f"\n{len(remaining_issues)} issue(s) remain unresolved:")
        for c in remaining_issues:
            print(f" - {c.name}")
    elif num_failures:
        print("\nCheck complete: All issues resolved after fixes.")
    else:
        print("\nCheck complete: No issues found.")


if __name__ == "__main__":
    main()
