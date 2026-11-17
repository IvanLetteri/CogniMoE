import subprocess
import os
import json
import numpy as np
from datetime import datetime

MODEL_PATH = "/home/user/models/Wrench-35B-A3B-Q4_K_M.gguf"
BIN_PATH = "./build/bin/llama-cli"
WIKI_DIR = "cognimoe_wiki"

os.makedirs(WIKI_DIR, exist_ok=True)

DOMAINS = {
    "coding": "Write a lock-free ring buffer in C++20 using std::atomic with memory ordering explanations.",
    "math": "Solve the differential equation dy/dx + 2xy = x^3 step by step and find the particular solution for y(0)=1.",
    "prose": "Write a vivid, atmospheric opening paragraph for a science fiction novel set on an orbital shipyard.",
    "reasoning": "Three switches outside a closed room control three light bulbs inside. You can enter the room only once. How do you determine which switch controls which bulb?"
}

def run_profiling():
    for domain, prompt in DOMAINS.items():
        json_out = os.path.join(WIKI_DIR, f"profile_{domain}.json")
        md_out = os.path.join(WIKI_DIR, f"skill_{domain}.md")
        
        print(f"\n🔍 [CogniMoE Profiler] Estrazione mappa termica esperti per dominio: '{domain.upper()}'...")
        env = os.environ.copy()
        env["COGNIMOE_TELEMETRY_OUT"] = json_out
        
        cmd = [
            BIN_PATH, "-m", MODEL_PATH,
            "--cpu-moe", "--moe-cache-mib", "4800",
            "--single-turn", "-n", "64", "-p", prompt
        ]
        
        subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if os.path.exists(json_out):
            with open(json_out) as f:
                data = json.load(f)
            mat = np.array(data["activations"])
            expert_totals = mat.sum(axis=0)
            top_20 = np.argsort(expert_totals)[::-1][:20].tolist()
            coverage_66 = sum(np.sort(row)[::-1][:66].sum() for row in mat) / mat.sum() * 100
            
            with open(md_out, "w") as md:
                md.write(f"# Skill Profile: {domain.upper()}\n\n")
                md.write(f"- **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                md.write(f"- **Target Hardware:** RTX 2070 Super (CUDA0 - 4800 MiB Cache)\n")
                md.write(f"- **Theoretical VRAM Hit Rate (Top-66/layer):** {coverage_66:.2f}%\n")
                md.write(f"- **Telemetry File:** `profile_{domain}.json`\n\n")
                md.write("## Top-20 Global Hot Experts\n")
                md.write(f"`{top_20}`\n\n")
                md.write("## Routing Notes\n")
                md.write(f"Pre-warming VRAM with `profile_{domain}.json` eliminates cold-start PCIe transfers for {domain} workloads.\n")
            
            print(f"✅ Salvato {json_out} e nota Wiki {md_out} (Copertura teorica Top-66: {coverage_66:.1f}%)")

if __name__ == "__main__":
    run_profiling()
