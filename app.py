"""Malicious URL Classifier - Streamlit web app (one page, four tabs).

Text-only analysis: the app never opens, resolves or fetches any URL.
Model: Stacking (LinearSVC + ComplementNB on char/word TF-IDF + 29 lexical features -> HistGradientBoosting)
with a benign threshold of 0.6, trained in notebooks/MaliciousURL_Full_CL.ipynb (models/url_model_original.joblib).
"""
import html
import random
import re
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import pandas as pd
import streamlit as st

import urlfeat  # noqa: F401  (the saved bundle refers to urlfeat.url_tokens, so it must be importable)
from urlfeat import LEXICAL_NAMES, lexical_features, normalize, predict_urls

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "url_model_original.joblib"
LABELS = ["benign", "phishing", "malware", "defacement"]
GAME_ROUNDS = 10
CONTROL = re.compile(r"[\x00-\x1F\x7F-\x9F]")
COLOR = {"benign": "#1E7A4F", "phishing": "#B91C1C", "malware": "#7F1D1D", "defacement": "#C2410C"}
INFO = {
    "benign": ("ปลอดภัย", "URL มีลักษณะเหมือนเว็บทั่วไป ไม่พบรูปแบบของภัยที่โมเดลรู้จัก",
               "ยังควรตรวจชื่อโดเมนให้ตรงกับเว็บที่ตั้งใจเข้า เพราะโมเดลดูจากข้อความ URL เท่านั้น"),
    "phishing": ("เว็บหลอกขโมยข้อมูล", "มีลักษณะของเว็บปลอมที่หลอกให้กรอกรหัสผ่าน ข้อมูลบัตร หรือข้อมูลส่วนตัว",
                 "อย่ากรอกข้อมูลใด ๆ ให้พิมพ์ชื่อเว็บจริงเองในเบราว์เซอร์แทนการกดลิงก์"),
    "malware": ("ลิงก์แจกไฟล์อันตราย", "มีลักษณะของลิงก์ที่แจกจ่ายไฟล์หรือโค้ดอันตราย",
                "อย่าดาวน์โหลดหรือเปิดไฟล์จากลิงก์นี้ และแจ้งผู้ดูแลระบบถ้าได้รับทางอีเมล"),
    "defacement": ("เว็บที่ถูกเจาะแก้ไข", "มีลักษณะของหน้าเว็บในเว็บไซต์ที่เคยถูกเจาะแก้ไข (มักเป็นเว็บ CMS เก่า)",
                   "ข้อมูลในหน้านี้อาจไม่น่าเชื่อถือ และเว็บอาจยังมีช่องโหว่อยู่"),
}
EXAMPLES = {
    "เว็บปกติ": "https://th.wikipedia.org/wiki/ประเทศไทย",
    "Phishing": "http://paypal-account-verify.secure-login.xyz/update/index.php",
    "Malware": "http://185.12.33.4:8080/bins/x86",
    "Defacement": "http://www.example-school.ac.th/index.php?option=com_content&view=article&id=70",
}
# (feature, rule, Thai text): lexical signals worth pointing out to the user
SIGNALS = [
    ("is_ip", lambda v: v > 0, "ใช้ IP แทนชื่อโดเมน"),
    ("has_port", lambda v: v > 0, "ระบุ port"),
    ("is_shortener", lambda v: v > 0, "เป็นลิงก์ย่อ (ซ่อนปลายทาง)"),
    ("sus_words", lambda v: v > 0, "มีคำล่อลวง {v:.0f} คำ (login, verify, bank ...)"),
    ("risky_ext", lambda v: v > 0, "ลงท้ายด้วยไฟล์เสี่ยง (.php .exe .apk ...)"),
    ("n_at", lambda v: v > 0, "มีเครื่องหมาย @ (ข้อความก่อน @ ถูกมองข้าม)"),
    ("n_host_hyphen", lambda v: v >= 2, "ชื่อโฮสต์มีขีด {v:.0f} ตัว"),
    ("n_host_dot", lambda v: v >= 3, "subdomain ซ้อนกันหลายชั้น ({v:.0f} จุด)"),
    ("url_len", lambda v: v > 75, "URL ยาวผิดปกติ ({v:.0f} ตัวอักษร)"),
    ("n_double_slash", lambda v: v > 0, "มี // ซ้อนอยู่ใน URL"),
    ("digit_ratio", lambda v: v > 0.2, "มีตัวเลขมาก ({v:.0%} ของ URL)"),
    ("host_entropy", lambda v: v > 3.8, "ชื่อโฮสต์ดูสุ่ม (entropy {v:.2f})"),
]

st.set_page_config(page_title="Malicious URL Classifier", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}
.hero {background: linear-gradient(120deg, #7F1D1D 0%, #B91C1C 55%, #DC2626 100%); color: #fff;
       border-radius: 14px; padding: 26px 32px; margin-bottom: 18px; position: relative; overflow: hidden;}
.hero:after {content: ""; position: absolute; right: -60px; top: -60px; width: 240px; height: 240px;
             border-radius: 50%; background: rgba(255,255,255,0.08);}
.hero h1 {color: #fff; font-size: 2.0rem; margin: 0 0 4px 0; padding: 0;}
.hero p {margin: 0; opacity: 0.92;}
.hero .tags span {display: inline-block; margin: 12px 6px 0 0; padding: 3px 12px; border-radius: 999px;
                  border: 1px solid rgba(255,255,255,0.55); font-size: 0.82rem; letter-spacing: 0.3px;}
.stTabs [data-baseweb="tab-list"] {gap: 4px; border-bottom: 2px solid #F3D4D4;}
.stTabs [data-baseweb="tab"] {padding: 8px 18px; border-radius: 8px 8px 0 0; font-weight: 600;}
.stTabs [aria-selected="true"] {background: #FCEDED; color: #B91C1C;}
.ex-label {font-size: 0.85rem; color: #6B7280; margin-bottom: 2px;}
.st-key-examples button {padding: 1px 14px; min-height: 0; font-size: 0.82rem; white-space: nowrap;
       border-radius: 999px; border: 1px solid #E8C4C4; color: #9B1C1C; background: #fff;}
.st-key-examples button p {font-size: 0.82rem;}
.st-key-examples button:hover {background: #FCEDED; border-color: #B91C1C; color: #7F1D1D;}
.result {border-radius: 12px; padding: 18px 22px; border-left: 6px solid; background: #FBF7F7; margin-bottom: 10px;}
.result .lbl {font-size: 1.35rem; font-weight: 700; line-height: 1.35;}
.result .sub {color: #4B5563; font-size: 0.95rem;}
.bar {margin: 6px 0;}
.bar .row {display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 2px;}
.bar .track {background: #F1E6E6; border-radius: 999px; height: 9px; overflow: hidden; position: relative;}
.bar .fill {height: 100%; border-radius: 999px;}
.card {border: 1px solid #EFDCDC; border-radius: 12px; padding: 16px 18px; background: #fff;
       box-shadow: 0 1px 3px rgba(127,29,29,0.06); margin-bottom: 10px;}
.card h4 {margin: 0 0 8px 0; color: #7F1D1D; font-size: 1rem;}
.chip {display: inline-block; padding: 3px 10px; margin: 3px; border-radius: 999px; font-size: 0.85rem;}
.chip-up {background: #FDE2E2; color: #9B1C1C;} .chip-ok {background: #E6F4EA; color: #1E6B47;}
.defang {font-family: Consolas, monospace; font-size: 1.15rem; background: #FBF2F2; border: 1px dashed #E3B5B5;
         padding: 14px 18px; border-radius: 10px; word-break: break-all;}
.score {text-align: center; border-radius: 12px; padding: 10px 6px; background: #FBF2F2;}
.score .n {font-size: 1.8rem; font-weight: 700; color: #B91C1C; line-height: 1.1;}
.score .t {font-size: 0.8rem; color: #6B7280;}
.dots {margin: 6px 0 14px 0;}
.dots span {display: inline-block; width: 22px; height: 6px; border-radius: 3px; margin-right: 4px; background: #F1E6E6;}
.dots .ok {background: #1E7A4F;} .dots .bad {background: #B91C1C;} .dots .now {background: #F59E9E;}
.footer {color: #9CA3AF; font-size: 0.8rem; text-align: center; margin-top: 28px;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="กำลังโหลดโมเดล (ครั้งแรกใช้เวลาประมาณ 10–20 วินาที)")
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_game():
    return pd.read_csv(ROOT / "data" / "game_urls.csv")


def defang(u: str) -> str:
    """Display-only: make a URL non-clickable."""
    return u.replace("http", "hxxp", 1).replace(".", "[.]")


def refang(u: str) -> str:
    """Undo defanging (hxxp, [.], [:]) so a URL copied from a report is predicted correctly."""
    u = re.sub(r"^hxxp", "http", u.strip(), flags=re.IGNORECASE)
    return u.replace("[.]", ".").replace("[:]", ":")


def validate(text: str):
    text = refang(text or "")
    if not text:
        return None, "กรุณากรอก URL ก่อน"
    if CONTROL.search(text):
        return None, "URL มีอักขระควบคุม (control character) จึงทำนายไม่ได้"
    if len(text) > 2000:
        return None, "URL ยาวเกิน 2,000 ตัวอักษร"
    return text, None


def predict_many(urls):
    """Stacking + threshold: returns predicted labels and a DataFrame of class probabilities."""
    out = predict_urls(load_model(), list(urls))
    proba = out[[f"P({c})" for c in urlfeat.LABELS]].copy()
    proba.columns = urlfeat.LABELS
    return out["prediction"].tolist(), proba.reset_index(drop=True)


def lexical_dict(url: str) -> dict:
    return dict(zip(LEXICAL_NAMES, lexical_features([url])[0]))


def signals(url: str):
    f = lexical_dict(url)
    return [msg.format(v=f[name]) for name, rule, msg in SIGNALS if rule(f[name])]


def anatomy(url: str) -> pd.DataFrame:
    raw = url.strip()
    parts = urlsplit(raw if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", raw) else "http://" + raw)
    f = lexical_dict(raw)
    yes = lambda v: "ใช่ (น่าสงสัย)" if v else "ไม่"
    rows = [("scheme", parts.scheme if "://" in raw else "(ไม่ได้ระบุ)"), ("โฮสต์", parts.hostname or "-"),
            ("path", parts.path or "-"), ("query", parts.query or "-"),
            ("ความยาว URL (หลัง normalize)", f"{int(f['url_len'])} ตัวอักษร"), ("จำนวนจุดในโฮสต์", str(int(f["n_host_dot"]))),
            ("ใช้ IP แทนชื่อโดเมน", yes(f["is_ip"])), ("ลิงก์ย่อ", yes(f["is_shortener"])),
            ("คำล่อลวง", f"{int(f['sus_words'])} คำ"), ("ความสุ่มของตัวอักษร (entropy)", f"{f['entropy']:.2f}")]
    return pd.DataFrame(rows, columns=["ส่วนประกอบ", "ค่า"])


def prob_bars(p, threshold) -> str:
    rows = []
    for c in LABELS:
        mark = (f'<div style="position:absolute;left:{threshold * 100:.0f}%;top:-2px;bottom:-2px;width:2px;background:#1F2933"></div>'
                if c == "benign" else "")
        rows.append(f'<div class="bar"><div class="row"><span>{c} · {INFO[c][0]}</span><b>{p[c]:.1%}</b></div>'
                    f'<div class="track"><div class="fill" style="width:{p[c] * 100:.1f}%;background:{COLOR[c]}"></div>{mark}</div></div>')
    return "".join(rows)


def new_game():
    s = st.session_state
    s.order = random.sample(range(len(load_game())), min(GAME_ROUNDS, len(load_game())))
    s.pos, s.you, s.ai, s.answered, s.marks = 0, 0, 0, False, []


# ------------------------------------------------------------------ header
st.markdown("""
<div class="hero">
  <h1>Malicious URL Classifier</h1>
  <p>จำแนก URL จากข้อความอย่างเดียว ระบบไม่เปิดหรือเรียกเว็บปลายทาง</p>
  <div class="tags"><span>benign</span><span>phishing</span><span>malware</span><span>defacement</span></div>
</div>""", unsafe_allow_html=True)
tab_check, tab_batch, tab_game, tab_about = st.tabs(["ตรวจ URL", "ตรวจหลาย URL", "เกมทาย URL", "เกี่ยวกับโมเดล"])

# ------------------------------------------------------------------ tab 1: single URL
with tab_check:
    url = st.text_input("กรอก URL ที่ต้องการตรวจ", key="url_input", placeholder="เช่น http://paypa1-secure-login.tk/verify")
    st.markdown('<div class="ex-label">ลองตัวอย่าง</div>', unsafe_allow_html=True)
    with st.container(horizontal=True, key="examples", gap="small"):
        for name, ex in EXAMPLES.items():
            st.button(name, key=f"ex_{name}", on_click=st.session_state.__setitem__, args=("url_input", ex))
    if st.button("Predict", type="primary"):
        text, err = validate(url)
        if err:
            st.warning(err)
        else:
            labels, proba = predict_many([text])
            label, p, t = labels[0], proba.iloc[0], load_model()["threshold"]
            name_th, meaning, advice = INFO[label]
            st.session_state.setdefault("history", []).insert(0, {"URL": defang(text), "ผล": label, "ความน่าจะเป็น": f"{p[label]:.1%}"})
            left, right = st.columns([3, 2], gap="large")
            with left:
                st.markdown(
                    f'<div class="result" style="border-color:{COLOR[label]}">'
                    f'<div class="lbl" style="color:{COLOR[label]}">{label.capitalize()} · {name_th}</div>'
                    f'<div class="sub">ความน่าจะเป็น {p[label]:.1%}</div></div>', unsafe_allow_html=True)
                st.markdown(f"**ความหมาย:** {meaning}")
                st.markdown(f"**คำแนะนำ:** {advice}")
                if label == "benign":
                    why = f"P(benign) = {p['benign']:.1%} ถึงเกณฑ์ {t:.0%} จึงทายว่าปลอดภัย"
                else:
                    why = (f"P(benign) = {p['benign']:.1%} ต่ำกว่าเกณฑ์ {t:.0%} จึงเลือกคลาสอันตรายที่น่าจะเป็นที่สุด"
                           f" คือ {label} ({p[label]:.1%})")
                st.caption(f"กฎการตัดสิน: {why}")
                if p[label] < 0.6:
                    st.info("ความน่าจะเป็นต่ำกว่า 60% ผลนี้ไม่ชัดเจน ควรตรวจซ้ำด้วยวิธีอื่น")
                st.markdown('<div class="card"><h4>ความน่าจะเป็นของแต่ละคลาส</h4>' + prob_bars(p, t)
                            + f'<div style="font-size:0.78rem;color:#6B7280;margin-top:6px">เส้นดำบนแถบ benign = เกณฑ์ {t:.0%}</div></div>',
                            unsafe_allow_html=True)
            with right:
                st.markdown("**ส่วนประกอบของ URL**")
                st.dataframe(anatomy(text), hide_index=True, width="stretch")
                st.caption(f"URL หลังเตรียมข้อมูล (ตัด http(s):// และ www.): `{normalize([text])[0]}`")
            sig = signals(text)
            chips = "".join(f'<span class="chip chip-up">{html.escape(m)}</span>' for m in sig) or \
                '<span class="chip chip-ok">ไม่พบจุดน่าสงสัยเชิงโครงสร้าง</span>'
            st.markdown(f'<div class="card"><h4>จุดสังเกตจากโครงสร้าง URL</h4>{chips}</div>', unsafe_allow_html=True)
            st.caption("จุดสังเกตมาจาก lexical feature 29 ตัวที่โมเดลใช้ ส่วนการตัดสินจริงรวมชิ้นตัวอักษรและคำใน URL (char + word TF-IDF) ด้วย")
    if st.session_state.get("history"):
        with st.expander(f"ประวัติการตรวจ ({len(st.session_state['history'])} รายการ)"):
            st.dataframe(pd.DataFrame(st.session_state["history"]), hide_index=True, width="stretch")

# ------------------------------------------------------------------ tab 2: batch
with tab_batch:
    st.write("วาง URL บรรทัดละ 1 รายการ หรืออัปโหลดไฟล์ CSV ที่มีคอลัมน์ `url` (สูงสุด 5,000 รายการ)")
    pasted = st.text_area("URL หลายรายการ", height=150, placeholder="google.com\nhttp://free-bonus-gift.tk/download/setup.exe")
    up_file = st.file_uploader("หรืออัปโหลด CSV", type=["csv"])
    if st.button("ตรวจทั้งหมด", type="primary"):
        urls = [u.strip() for u in pasted.splitlines() if u.strip()]
        if up_file is not None:
            df_in = pd.read_csv(up_file)
            col = next((c for c in df_in.columns if c.lower() == "url"), df_in.columns[0])
            urls += df_in[col].astype(str).tolist()
        urls = [refang(u) for u in urls if u and not CONTROL.search(u)][:5000]
        if not urls:
            st.warning("ยังไม่มี URL ให้ตรวจ")
        else:
            labels, proba = predict_many(urls)
            out = pd.DataFrame({"url": urls, "prediction": labels,
                                "probability": [proba.loc[i, l] for i, l in enumerate(labels)]}).round({"probability": 4})
            counts = out["prediction"].value_counts().reindex(LABELS, fill_value=0)
            for c, box in zip(LABELS, st.columns(4)):
                box.markdown(f'<div class="score"><div class="n" style="color:{COLOR[c]}">{int(counts[c])}</div>'
                             f'<div class="t">{c} · {INFO[c][0]}</div></div>', unsafe_allow_html=True)
            st.write("")
            st.bar_chart(counts, color="#B91C1C")
            st.dataframe(out.assign(url=out["url"].map(defang)), hide_index=True, width="stretch")
            st.download_button("ดาวน์โหลดผลเป็น CSV", out.to_csv(index=False).encode("utf-8-sig"), "url_predictions.csv", "text/csv")

# ------------------------------------------------------------------ tab 3: game
with tab_game:
    game = load_game()
    s = st.session_state
    if "order" not in s:
        new_game()
    n = len(s.order)
    st.write(f"สุ่ม {n} URL จาก {len(game)} ตัวอย่างของ dataset แล้วทายว่าเป็นประเภทไหน แข่งกับโมเดล AI "
             "(URL แสดงแบบ defang เพื่อไม่ให้กดเปิดได้)")
    c1, c2, c3 = st.columns(3)
    for box, val, cap in [(c1, f"{min(s.pos + 1, n)} / {n}", "ข้อที่"), (c2, s.you, "คะแนนคุณ"), (c3, s.ai, "คะแนน AI")]:
        box.markdown(f'<div class="score"><div class="n">{val}</div><div class="t">{cap}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="dots">' + "".join(
        f'<span class="{s.marks[i] if i < len(s.marks) else ("now" if i == s.pos else "")}"></span>' for i in range(n))
        + "</div>", unsafe_allow_html=True)
    if s.pos >= n:
        verdict = "คุณชนะ AI" if s.you > s.ai else "เสมอกับ AI" if s.you == s.ai else "AI ชนะ"
        st.success(f"จบเกม {verdict} · คุณได้ {s.you} คะแนน · AI ได้ {s.ai} คะแนน จาก {n} ข้อ")
    else:
        row = game.iloc[s.order[s.pos]]
        st.markdown(f'<div class="defang">{html.escape(defang(row.url))}</div>', unsafe_allow_html=True)
        st.write("")
        for c, b in zip(LABELS, st.columns(4)):
            if b.button(c, key=f"g_{c}", help=INFO[c][0], width="stretch", disabled=s.answered):
                ai = predict_many([row.url])[0][0]
                s.you += int(c == row.type)
                s.ai += int(ai == row.type)
                s.marks.append("ok" if c == row.type else "bad")
                s.last = (c, ai)
                s.answered = True
                st.rerun()
        if s.answered:
            you, ai = s.last
            mark = lambda x: "ถูก" if x == row.type else "ผิด"
            (st.success if you == row.type else st.error)(
                f"เฉลย: **{row.type}** · คุณตอบ {you} ({mark(you)}) · AI ตอบ {ai} ({mark(ai)})")
            st.caption(f"{INFO[row.type][1]} · ที่มา: {row.source}")
            if st.button("ข้อต่อไป" if s.pos + 1 < n else "ดูผลรวม", type="primary"):
                s.pos += 1
                s.answered = False
                st.rerun()
    if st.button("เริ่มเกมใหม่ (สุ่มชุดใหม่)"):
        new_game()
        st.rerun()

# ------------------------------------------------------------------ tab 4: about
with tab_about:
    st.subheader("โมเดลที่ใช้: Stacking + threshold 0.6")
    st.markdown("""
- **ชั้นที่ 1:** LinearSVC (C = 0.1) และ ComplementNB (α = 0.2) บน character n-gram 2–6 และ word token TF-IDF (hashing 1.5 ล้านมิติ)
- **ชั้นที่ 2:** HistGradientBoosting รับคะแนนจากชั้นที่ 1 รวมกับ lexical feature 29 ตัว
- **กฎการตัดสิน:** ทาย benign เมื่อ P(benign) ≥ 0.6 ไม่อย่างนั้นทายคลาสอันตรายที่น่าจะเป็นที่สุด (เลือก 0.6 จาก F2-score บน Validation เพื่อเน้น Recall)
- เทรนจาก Malicious URLs Dataset (Kaggle) แบ่ง Train / Validation / Test ตาม hostname
""")
    a, b, c, d = st.columns(4)
    for box, val, cap in [(a, "0.858", "Macro-F1 (Validation)"), (b, "0.862", "Macro-F1 (Test)"),
                          (c, "94.3%", "จับ URL อันตรายได้ (Test)"), (d, "14.1%", "เว็บปกติถูกเตือนผิด (Test)")]:
        box.markdown(f'<div class="score"><div class="n">{val}</div><div class="t">{cap}</div></div>', unsafe_allow_html=True)
    st.markdown("**ผลกับข้อมูลภายนอกที่ไม่เคยใช้เทรน**")
    st.dataframe(pd.DataFrame({
        "ชุดข้อมูล": ["PhishTank (phishing จริง)", "URLhaus (malware 30 วัน)", "DeepURLBench (เว็บปกติ)"],
        "จำนวน": ["77,755", "15,326", "164,782"],
        "ผล": ["จับ phishing ได้ 93.6%", "จับ malware ได้ 94.7%", "เตือนผิด 75.7%"]}), hide_index=True, width="stretch")
    st.subheader("ข้อจำกัด")
    st.markdown("""
- ดูจาก **ข้อความ URL อย่างเดียว** phishing ที่ฝากบนบริการใหญ่ (Google Docs, Weebly) หรือใช้โดเมนดูปกติอาจหลุด
- **เตือนผิดกับเว็บปกติจากแหล่งอื่นสูง** เพราะ label ใน Kaggle มีข้อมูล PhishStorm ที่ถูกสลับ label (14.7% ของไฟล์) ระบบนี้จึงเหมาะกับการคัดกรองเบื้องต้น
- ป้าย defacement ของ dataset ติดทั้งเว็บที่เคยถูกเจาะ ไม่ใช่เฉพาะหน้าที่ถูกเจาะ
- ผลเป็นการประเมินโดยโมเดล ใช้ประกอบการตัดสินใจ ไม่ใช่คำตัดสินสุดท้าย
""")

st.markdown('<div class="footer">Mini Project: Machine Learning Application · Malicious URL Classification</div>',
            unsafe_allow_html=True)
