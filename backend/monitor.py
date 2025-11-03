import subprocess

def gpu_stats():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,power.draw",
             "--format=csv,noheader,nounits"]
        ).decode().strip()
        mem, power = out.split(',')
        return dict(vram_mb=int(mem), power_w=float(power))
    except Exception:
        return dict(vram_mb=None, power_w=None)
