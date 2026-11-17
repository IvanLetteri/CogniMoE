import streamlit as st
import subprocess
import os
import re
import json
import time
import numpy as np
import pandas as pd
import pynvml
from sentence_transformers import SentenceTransformer
from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="CogniMoE Supervisor", page_icon="🧠", layout="wide")
st.title("🧠 CogniMoE: Cognitive Expert Routing & VRAM Orchestrator")

MODEL_PATH = "/home/user/models/Wrench-35B-A3B-Q4_K_M.gguf"
BIN_PATH = "./build/bin/llama-completion"
WIKI_DIR = "cognimoe_wiki"
HISTORY_FILE = os.path.join(WIKI_DIR, "bandit_history.json")

# --- TELEMETRIA GPU (PYNVML) ---
@st.cache_resource
def init_nvml():
    try:
        pynvml.nvmlInit()
        return True
    except Exception:
        return False

has_nvml = init_nvml()

def get_gpu_telemetry():
    if not has_nvml:
        return {"CUDA0": {"free_mib": 7677, "total_mib": 7786, "temp": 0}}
    stats = {}
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        stats[f"CUDA{i}"] = {
            "free_mib": int(mem.free / (1024**2)),
            "total_mib": int(mem.total / (1024**2)),
            "temp": temp
        }
    return stats

# --- MOTORE EMBEDDING SU CPU (ZERO VRAM) ---
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

embedder = load_embedder()

def build_context_vector(text: str, cache_mib: float) -> np.ndarray:
    """Costruisce il vettore di stato normalizzato (384 dim embedding + cache_gb)."""
    emb = embedder.encode([text], normalize_embeddings=True).flatten()
    cache_gb = np.array([cache_mib / 1024.0], dtype=np.float32)
    return np.concatenate([emb, cache_gb])

# --- DEFINIZIONE AZIONI / PROFILI DELLA WIKI ---
PROFILES = {
    "AUTO_COLD (No Pre-Warming)": None,
    "SKILL_CODING (Top-66 Code Experts)": os.path.join(WIKI_DIR, "profile_coding.json"),
    "SKILL_MATH (Top-66 Math Experts)": os.path.join(WIKI_DIR, "profile_math.json"),
    "SKILL_PROSE (Top-66 Prose Experts)": os.path.join(WIKI_DIR, "profile_prose.json"),
    "SKILL_REASONING (Top-66 Logic Experts)": os.path.join(WIKI_DIR, "profile_reasoning.json"),
}

# --- INIZIALIZZAZIONE AGENTE RL (CONTEXTUAL BANDIT) ---
def init_bandit():
    bandit = MAB(
        arms=list(PROFILES.keys()),
        learning_policy=LearningPolicy.LinUCB(alpha=0.85),
        neighborhood_policy=NeighborhoodPolicy.KNearest(k=3)
    )
    seed_prompts = {
        "SKILL_CODING (Top-66 Code Experts)": [
            "Write a Python script using asyncio to fetch multiple URLs concurrently.",
            "Implement a lock-free ring buffer in C++20 with std::atomic.",
            "Debug a SQL query and optimize database indexing performance."
        ],
        "SKILL_MATH (Top-66 Math Experts)": [
            "Solve the differential equation dy/dx + 2xy = x^3 step by step.",
            "Calculate the eigenvalues and eigenvectors of a 3x3 symmetric matrix.",
            "Prove by induction that the sum of the first n odd integers is n squared."
        ],
        "SKILL_PROSE (Top-66 Prose Experts)": [
            "Write a vivid opening paragraph for a science fiction novel.",
            "Compose an evocative description of an ancient library at sunset.",
            "Draft a persuasive executive summary for a sustainable energy project."
        ],
        "SKILL_REASONING (Top-66 Logic Experts)": [
            "Three switches outside a closed room control three light bulbs inside.",
            "Analyze the logical fallacies in a complex philosophical argument.",
            "Deduce the order of five people sitting around a table from clues."
        ]
    }
    vecs, arms, rewards = [], [], []
    for arm_name, examples in seed_prompts.items():
        if os.path.exists(PROFILES[arm_name]):
            for ex in examples:
                vecs.append(build_context_vector(ex, 4800.0))
                arms.append(arm_name)
                rewards.append(35.0)
    vecs.append(build_context_vector("Generic short greeting hello", 4800.0))
    arms.append("AUTO_COLD (No Pre-Warming)")
    rewards.append(25.0)

    bandit.fit(decisions=arms, rewards=rewards, contexts=np.array(vecs))
    return bandit

if "bandit" not in st.session_state:
    st.session_state.bandit = init_bandit()
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            st.session_state.history = json.load(f)
    else:
        st.session_state.history = []

# --- SIDEBAR: TOPOLOGIA HARDWARE & CONTROLLO MANUALE ---
st.sidebar.header("🖥️ Hardware & Offloading Nodes")
gpu_stats = get_gpu_telemetry()
for dev, info in gpu_stats.items():
    used_pct = 1.0 - (info["free_mib"] / info["total_mib"])
    st.sidebar.progress(used_pct, text=f"{dev}: {info['free_mib']} / {info['total_mib']} MiB Free ({info['temp']}°C)")

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Controllo Manuale Risorse")
control_mode = st.sidebar.radio("Modalità Operativa:", ["🤖 Auto-Apprendimento (RL Bandit)", "🖐️ Controllo Manuale GUI"])

manual_profile = st.sidebar.selectbox(
    "Profilo Esperti da Caricare su CUDA0:",
    options=list(PROFILES.keys()),
    disabled=(control_mode == "🤖 Auto-Apprendimento (RL Bandit)")
)

cache_mib = st.sidebar.slider("Budget Cache VRAM (MiB) su CUDA0:", min_value=1024, max_value=5200, value=4800, step=256)
cpu_miss_share = st.sidebar.slider(
    "Quota Miss su CPU (--moe-cache-cpu-share):",
    min_value=0.0, max_value=1.0, value=0.8, step=0.1,
    help="0.0 = carica tutto su GPU via PCIe; 0.8 = preserva gli esperti pre-caricati calcolando i miss rari su CPU."
)

rpc_nodes = st.sidebar.text_input("Nodi Android RPC (es. 192.168.1.50:50052):", value="", placeholder="Opzionale (futuro)")

if st.sidebar.button("🗑️ Resetta Memoria Bandit"):
    st.session_state.bandit = init_bandit()
    st.session_state.history = []
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    st.rerun()

# --- TABS PRINCIPALI ---
tab_infer, tab_heatmap, tab_wiki = st.tabs(["⚡ Inferenza & Routing RL", "🔥 Mappa Termica Esperti (40x256)", "📚 LLM Wiki (Second Brain)"])

with tab_infer:
    prompt = st.text_area("Prompt di Inferenza:", "Write a Python script using asyncio to fetch multiple URLs concurrently.", height=100)
    max_tokens = st.slider("Token da generare:", 32, 256, 96, step=32)

    if st.button("🚀 Esegui Inferenza su CogniMoE", type="primary"):
        context_vec = build_context_vector(prompt, float(cache_mib)).reshape(1, -1)

        if control_mode == "🤖 Auto-Apprendimento (RL Bandit)":
            pred = st.session_state.bandit.predict(context_vec)
            chosen_arm = pred[0] if isinstance(pred, (list, np.ndarray)) else pred
        else:
            chosen_arm = manual_profile

        prewarm_file = PROFILES[chosen_arm]
        st.info(f"**Strategia Attiva:** `{chosen_arm}` | **Cache VRAM:** `{cache_mib} MiB` | **CPU Miss Share:** `{cpu_miss_share}`")

        env = os.environ.copy()
        if prewarm_file and os.path.exists(prewarm_file):
            env["COGNIMOE_PREWARM_JSON"] = prewarm_file
        else:
            env.pop("COGNIMOE_PREWARM_JSON", None)

        live_telemetry_out = os.path.join(WIKI_DIR, "last_run_telemetry.json")
        env["COGNIMOE_TELEMETRY_OUT"] = live_telemetry_out

        cmd = [
            BIN_PATH, "-m", MODEL_PATH,
            "--cpu-moe",
            "--moe-cache-mib", str(cache_mib),
            "--moe-cache-cpu-share", str(cpu_miss_share),
            "-no-cnv",
            "--verbose",
            "-n", str(max_tokens),
            "-p", prompt
        ]
        if rpc_nodes.strip():
            cmd.extend(["--rpc", rpc_nodes.strip()])

        with st.spinner("Caricamento esperti in VRAM ed esecuzione grafo C++..."):
            t0 = time.time()
            proc = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True)
            elapsed = time.time() - t0

        full_logs = proc.stdout + "\n" + proc.stderr

        # Estrazione robusta della velocità di generazione
        gen_tps = 0.0
        eval_match = re.search(r"eval time =\s+[\d.,]+\s+ms\s+/\s+\d+\s+runs\s+\(\s+[\d.,]+\s+ms per token,\s+([\d.,]+)\s+tokens per second\)", full_logs)
        if eval_match:
            gen_tps = float(eval_match.group(1).replace(",", "."))
        else:
            bar_match = re.search(r"Generation:\s+([\d.,]+)\s+t/s", full_logs)
            if bar_match:
                gen_tps = float(bar_match.group(1).replace(",", "."))

        hit_match = re.search(r"ubatch <= 8:.*hit rate = ([\d.,]+)%", full_logs)
        hit_rate = float(hit_match.group(1).replace(",", ".")) if hit_match else 0.0

        # Calcolo Ricompensa (Reward) e Aggiornamento Online del Bandit
        reward = round(gen_tps + (hit_rate * 0.1), 2) if proc.returncode == 0 else -10.0
        st.session_state.bandit.partial_fit(decisions=[chosen_arm], rewards=[reward], contexts=context_vec)

        record = {
            "Prompt": prompt[:48] + ("..." if len(prompt) > 48 else ""),
            "Azione Scelta": chosen_arm,
            "Gen Speed (t/s)": gen_tps,
            "VRAM Hit Rate (%)": hit_rate,
            "Reward RL": reward
        }
        st.session_state.history.append(record)
        with open(HISTORY_FILE, "w") as f:
            json.dump(st.session_state.history, f, indent=2)

        col1, col2, col3 = st.columns(3)
        col1.metric("Velocità Generazione", f"{gen_tps:.2f} t/s")
        col2.metric("Cache Hit Rate (VRAM)", f"{hit_rate:.2f}%")
        col3.metric("Reward Agente RL", f"{reward:.2f}")

        st.subheader("Output Modello")
        if proc.returncode != 0 and not proc.stdout.strip():
            st.error("Errore durante l'esecuzione del motore C++:")
            st.code(proc.stderr[-1500:], language="text")
        else:
            st.code(proc.stdout.strip(), language="markdown")

    if st.session_state.history:
        st.subheader("📈 Registro Auto-Apprendimento (Contextual Bandit)")
        st.dataframe(pd.DataFrame(st.session_state.history), width="stretch")

with tab_heatmap:
    st.subheader("Distribuzione Attivazione Esperti (40 Layer x 256 Esperti)")
    json_files = [f for f in os.listdir(WIKI_DIR) if f.endswith(".json") and "bandit" not in f]
    selected_json = st.selectbox("Seleziona profilo telemetrico da ispezionare:", json_files)
    if selected_json:
        with open(os.path.join(WIKI_DIR, selected_json)) as f:
            jdata = json.load(f)
        mat = np.array(jdata["activations"])
        layer_sel = st.slider("Ispeziona Layer specifico:", 0, mat.shape[0] - 1, 5)
        df_layer = pd.DataFrame({"Expert ID": np.arange(mat.shape[1]), "Attivazioni": mat[layer_sel]})
        st.bar_chart(df_layer.set_index("Expert ID"))

with tab_wiki:
    st.subheader("Note di Conoscenza Persistente (Markdown)")
    md_files = [f for f in os.listdir(WIKI_DIR) if f.endswith(".md")]
    if md_files:
        chosen_md = st.selectbox("Apri nota Skill:", md_files)
        md_path = os.path.join(WIKI_DIR, chosen_md)
        with open(md_path) as f:
            md_text = f.read()
        edited_md = st.text_area("Contenuto Wiki (modificabile):", md_text, height=250)
        if st.button("💾 Salva Nota nella Wiki"):
            with open(md_path, "w") as f:
                f.write(edited_md)
            st.success("Nota aggiornata!")
