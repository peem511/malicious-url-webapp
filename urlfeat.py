"""Feature functions shared by training (Notebook) and the Streamlit web app."""
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp

SCHEME = r"^[A-Za-z][A-Za-z0-9+.-]*://"
SHORTENERS = r"bit\.ly|goo\.gl|tinyurl\.com|ow\.ly|t\.co|is\.gd|buff\.ly|adf\.ly|bitly\.com|cutt\.ly|rebrand\.ly|shorte\.st|tiny\.cc|v\.gd|j\.mp|x\.co|qr\.net|lnkd\.in|db\.tt|bit\.do"
SUS_WORDS = r"paypal|login|signin|bank|account|update|free|ebayisapi|webscr|lucky|bonus|verify|secure|confirm|password|wallet"
LABELS = ["benign", "defacement", "malware", "phishing"]


def normalize(urls):
    """ตัด scheme, www. นำหน้า และ / ท้าย ซึ่งใน dataset นี้บอกคลาสได้โดยไม่เกี่ยวกับความอันตราย"""
    s = pd.Series(urls, dtype="string").fillna("").str.strip()
    s = s.str.replace(SCHEME, "", regex=True)
    s = s.str.replace(r"^www\d?\.", "", regex=True, case=False)
    return s.str.rstrip("/")


def split_parts(s):
    host = s.str.extract(r"^([^/?#]*)", expand=False).fillna("").str.lower()
    rest = s.str.replace(r"^[^/?#]*", "", regex=True)
    path = rest.str.replace(r"[?#].*$", "", regex=True)
    query = rest.str.extract(r"\?([^#]*)", expand=False).fillna("")
    return host, path, query


def _entropy(text):
    if not text:
        return 0.0
    _, counts = np.unique(np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


LEXICAL_NAMES = [
    "url_len", "host_len", "path_len", "query_len", "n_dot", "n_host_dot", "n_hyphen", "n_host_hyphen",
    "n_at", "n_qmark", "n_amp", "n_eq", "n_underscore", "n_percent", "n_slash", "n_double_slash",
    "n_digit", "n_upper", "digit_ratio", "host_digits", "tld_len", "is_ip", "has_port", "is_shortener",
    "sus_words", "first_dir_len", "risky_ext", "entropy", "host_entropy",
]


def lexical_features(urls):
    """29 ตัวเลขเชิงโครงสร้างของ URL (คำนวณหลัง normalize)"""
    s = normalize(urls)
    host, path, query = split_parts(s)
    host_noport = host.str.replace(r":\d+$", "", regex=True)
    tld = host_noport.str.extract(r"\.([a-z0-9-]+)$", expand=False).fillna("")
    length = s.str.len().replace(0, 1)
    f = pd.DataFrame({
        "url_len": s.str.len(), "host_len": host.str.len(), "path_len": path.str.len(), "query_len": query.str.len(),
        "n_dot": s.str.count(r"\."), "n_host_dot": host.str.count(r"\."), "n_hyphen": s.str.count("-"),
        "n_host_hyphen": host.str.count("-"), "n_at": s.str.count("@"), "n_qmark": s.str.count(r"\?"),
        "n_amp": s.str.count("&"), "n_eq": s.str.count("="), "n_underscore": s.str.count("_"),
        "n_percent": s.str.count("%"), "n_slash": s.str.count("/"), "n_double_slash": s.str.count("//"),
        "n_digit": s.str.count(r"\d"), "n_upper": s.str.count(r"[A-Z]"), "digit_ratio": s.str.count(r"\d") / length,
        "host_digits": host.str.count(r"\d"), "tld_len": tld.str.len(),
        "is_ip": host_noport.str.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}").fillna(False),
        "has_port": host.str.contains(r":\d+$", regex=True),
        "is_shortener": host_noport.str.fullmatch(SHORTENERS).fillna(False),
        "sus_words": s.str.count(f"(?i)(?:{SUS_WORDS})"),
        "first_dir_len": path.str.split("/").str[1].fillna("").str.len(),
        "risky_ext": path.str.contains(r"\.(?:php|html?|exe|js|asp|aspx|cgi|zip|rar|apk|bin|scr)$", regex=True, case=False),
        "entropy": [_entropy(u) for u in s], "host_entropy": [_entropy(h) for h in host],
    })
    return f[LEXICAL_NAMES].astype(np.float32).to_numpy()


def url_tokens(url):
    """แยกคำพร้อมติดป้ายว่าอยู่ส่วนไหนของ URL (h: host, tld:, sld:, nsub:, p: path, p2: คู่คำ, q: query, ext:)"""
    m = re.match(r"^([^/?#]*)([^?#]*)\??([^#]*)", url)
    host, path, query = m.group(1).lower(), m.group(2), m.group(3)
    hp = [t for t in re.split(r"[.\-:]", host) if t]
    toks = ["h:" + t for t in hp]
    if len(hp) > 1:
        toks += ["tld:" + hp[-1], "sld:" + hp[-2], f"nsub:{min(len(hp), 6)}"]
    pt = [t for t in re.split(r"[^A-Za-z0-9]+", path) if t]
    toks += ["p:" + t.lower() for t in pt]
    toks += ["p2:" + a.lower() + "_" + b.lower() for a, b in zip(pt, pt[1:])]
    toks += ["q:" + t.lower() for t in re.split(r"[^A-Za-z0-9]+", query) if t]
    ext = re.search(r"\.([A-Za-z0-9]{1,5})$", path)
    if ext:
        toks.append("ext:" + ext.group(1).lower())
    return toks


def text_matrix(bundle, norm):
    Xc = bundle["tfidf_char"].transform(bundle["char_hash"].transform(norm))
    Xw = bundle["tfidf_word"].transform(bundle["word_hash"].transform(norm))
    return sp.hstack([Xc, Xw]).tocsr().astype(np.float32)


def predict_urls(bundle, urls):
    """ใช้ใน Web App: รับ URL ดิบ -> คลาสที่ทำนาย + ความน่าจะเป็นของแต่ละคลาส"""
    norm = normalize(urls).to_numpy(dtype=object)
    X = text_matrix(bundle, norm)
    Z = np.hstack([bundle["svc"].decision_function(X), bundle["nb"].predict_log_proba(X), lexical_features(norm)])
    P = bundle["meta"].predict_proba(Z)
    pred = np.where(P[:, 0] >= bundle["threshold"], 0, 1 + P[:, 1:].argmax(1))
    out = pd.DataFrame(P, columns=[f"P({l})" for l in LABELS])
    out.insert(0, "prediction", [LABELS[i] for i in pred])
    out.insert(0, "url", list(urls))
    return out
