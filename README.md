# Malicious URL Classifier

Web Application สำหรับจำแนก URL เป็น 4 ประเภท ได้แก่ **benign** (ปลอดภัย), **phishing** (เว็บหลอกขโมยข้อมูล),
**malware** (ลิงก์แจกไฟล์อันตราย) และ **defacement** (เว็บที่ถูกเจาะแก้ไข) จาก **ข้อความ URL อย่างเดียว**
ระบบไม่เปิด ไม่ดาวน์โหลด และไม่เรียกเว็บปลายทางเลย จึงปลอดภัยต่อผู้ใช้

> Mini Project: Machine Learning Application (344-362 การเรียนรู้ของเครื่อง)

---

## 1. โครงสร้างไฟล์

```
MaliciousURL_WebApp/
├── app.py                          # Web App (Streamlit) หน้าเดียว 4 แท็บ
├── urlfeat.py                      # normalize URL, สร้าง feature และฟังก์ชัน predict_urls (ใช้ร่วมกับ Notebook)
├── models/
│   └── url_model_original.joblib   # โมเดล Stacking + threshold 0.6 (119 MB, เก็บด้วย Git LFS)
├── data/
│   └── game_urls.csv               # URL 100 ตัวอย่าง (คลาสละ 25) สำหรับเกมทาย URL
├── notebooks/
│   ├── MaliciousURL_Full_CL.ipynb  # Notebook หลัก: เตรียมข้อมูล → เทรน → ประเมิน → ทดสอบภายนอก → ตรวจ label
│   ├── External_Test_3Models.ipynb # ทดสอบโมเดลที่บันทึกไว้กับข้อมูลภายนอก
│   └── MiniProject_MaliciousURL.ipynb # Logistic Regression ที่เทรนจาก label ที่แก้การสลับของ PhishStorm แล้ว (ใช้เทียบ)
├── .streamlit/config.toml          # ธีมสีแดง-ขาว
├── requirements.txt                # ไลบรารีที่ต้องใช้ (ระบุเวอร์ชัน)
└── README.md
```

## 2. วิธีติดตั้งและรันบนเครื่อง

ต้องใช้ **Python 3.11** (โมเดลบันทึกด้วย scikit-learn 1.9.0 ต้องใช้เวอร์ชันเดียวกันจึงจะโหลดได้)

```bash
# 1) ดาวน์โหลดโค้ด (ถ้าโคลนจาก GitHub ต้องมี Git LFS เพื่อดึงไฟล์โมเดล)
git lfs install
git clone <ลิงก์ repository>
cd MaliciousURL_WebApp

# 2) สร้าง virtual environment และติดตั้งไลบรารี
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS / Linux: source .venv/bin/activate)
pip install -r requirements.txt

# 3) รัน Web App
streamlit run app.py
```

เบราว์เซอร์จะเปิดที่ `http://localhost:8501` เอง การโหลดโมเดลครั้งแรกใช้เวลาประมาณ 10–20 วินาที หลังจากนั้นจะเร็วเพราะถูก cache ไว้

> ถ้าเจอ error ว่าไฟล์โมเดลเสียหรือมีขนาดเล็กมาก (ประมาณ 1 KB) แปลว่ายังไม่ได้ดึงไฟล์จริงจาก Git LFS ให้รัน `git lfs pull`

## 3. วิธีใช้งาน

### แท็บ "ตรวจ URL"
1. กรอก URL ในช่อง หรือกดปุ่มตัวอย่างเล็ก ๆ (เว็บปกติ / Phishing / Malware / Defacement)
2. กดปุ่ม **Predict**
3. ระบบแสดง:
   - **ผลการทาย** พร้อมความหมายและคำแนะนำภาษาไทย
   - **กฎการตัดสิน** เช่น "P(benign) = 0.9% ต่ำกว่าเกณฑ์ 60% จึงเลือกคลาสอันตรายที่น่าจะเป็นที่สุด"
   - **ความน่าจะเป็นของทั้ง 4 คลาส** เป็นแถบสี มีเส้นดำแสดงเกณฑ์ 0.6 บนแถบ benign
   - **ส่วนประกอบของ URL** (scheme, โฮสต์, path, query, ใช้ IP หรือไม่ ฯลฯ)
   - **จุดสังเกตจากโครงสร้าง URL** เช่น ใช้ IP แทนชื่อโดเมน, มีคำล่อลวง, ระบุ port
   - **ประวัติการตรวจ** ของรอบการใช้งานนี้

วาง URL แบบ defang (เช่น `hxxp://evil[.]com`) ได้ ระบบจะแปลงกลับเป็น URL ปกติก่อนทำนาย

### แท็บ "ตรวจหลาย URL"
- วาง URL บรรทัดละ 1 รายการ หรืออัปโหลดไฟล์ CSV ที่มีคอลัมน์ `url` (สูงสุด 5,000 รายการ)
- กด **ตรวจทั้งหมด** จะได้จำนวนแต่ละคลาส กราฟสรุป ตารางผล และปุ่ม **ดาวน์โหลดผลเป็น CSV**

### แท็บ "เกมทาย URL"
- ระบบสุ่ม 10 ข้อจาก URL 100 ตัวอย่าง ให้ผู้เล่นทายว่าเป็นประเภทไหน แล้วเทียบกับคำตอบของโมเดล
- แถบด้านบนแสดงความคืบหน้า (เขียว = ถูก, แดง = ผิด) จบเกมแล้วบอกว่าใครชนะ
- กด **เริ่มเกมใหม่** เพื่อสุ่มชุดใหม่

### แท็บ "เกี่ยวกับโมเดล"
สรุปโมเดล ผลการวัด ผลกับข้อมูลภายนอก และข้อจำกัด

> URL อันตรายในตารางและในเกมแสดงแบบ **defang** (`http` → `hxxp`, `.` → `[.]`) เพื่อไม่ให้กดเปิดโดยไม่ตั้งใจ

## 4. โมเดล

**Stacking + threshold 0.6** (`models/url_model_original.joblib`)

| ส่วน | รายละเอียด |
|---|---|
| เตรียม URL | normalize: ตัด `http(s)://`, `www.` และ `/` ท้าย |
| Feature | Character n-gram 2–6 (hashing 1,048,576 มิติ) + Word token ติดป้ายตำแหน่ง (524,288 มิติ) ถ่วงด้วย TF-IDF และ Lexical 29 ตัว |
| ชั้นที่ 1 | LinearSVC (C = 0.1) และ ComplementNB (α = 0.2) |
| ชั้นที่ 2 | HistGradientBoosting (learning rate 0.05, max leaf nodes 63) รับคะแนนชั้นที่ 1 + lexical 29 ตัว |
| กฎการตัดสิน | ทาย benign เมื่อ P(benign) ≥ 0.6 ไม่อย่างนั้นทายคลาสอันตรายที่น่าจะเป็นที่สุด |

### ผลการประเมิน

| ชุดข้อมูล | ผล |
|---|---|
| Validation (62,817 URL) | Macro-F1 0.858 |
| Test (125,781 URL) | Macro-F1 0.862 · Accuracy 87.25% · Detection 94.3% · False alarm 14.1% |
| PhishTank (77,755 phishing จริง) | จับได้ 93.6% |
| URLhaus (15,326 malware) | จับได้ 94.7% |
| DeepURLBench (เว็บปกติ) | เตือนผิด 75.7% |

### เรียกใช้โมเดลจาก Python

```python
import joblib, urlfeat
bundle = joblib.load("models/url_model_original.joblib")
urlfeat.predict_urls(bundle, ["google.com", "http://185.12.33.4:8080/bins/x86"])
# คืน DataFrame: url, prediction, P(benign), P(defacement), P(malware), P(phishing)
```

## 5. เทรนโมเดลใหม่

เปิด `notebooks/MaliciousURL_Full_CL.ipynb` แล้วรันทุก cell ตามลำดับ (ใช้เวลาประมาณ 30–60 นาที และ RAM 8–10 GB)
Notebook จะดาวน์โหลด dataset จาก Kaggle ด้วย `kagglehub` และบันทึก `url_model_original.joblib` ให้นำไฟล์นั้นมาแทนในโฟลเดอร์ `models/`
ต้องติดตั้ง `kagglehub`, `tldextract`, `matplotlib`, `seaborn` เพิ่มจาก requirements.txt
ข้อมูลภายนอกสำหรับทดสอบอยู่ในโฟลเดอร์ `data/external/` ของโปรเจกต์หลัก (ไม่ได้รวมไว้ใน repository นี้)

## 6. Deploy บน Streamlit Community Cloud

1. push โฟลเดอร์นี้ขึ้น GitHub (ไฟล์ `.joblib` ถูกเก็บด้วย Git LFS ตาม `.gitattributes`)
2. เข้า <https://share.streamlit.io> → **Create app** → เลือก repository, branch `main` และไฟล์ `app.py`
3. **Advanced settings → Python version: 3.11**
4. กด **Deploy** รอติดตั้งไลบรารีประมาณ 3–5 นาที

## 7. ข้อจำกัด

- ดูจากข้อความ URL อย่างเดียว phishing ที่ฝากบนบริการใหญ่ (Google Docs, Weebly) หรือใช้โดเมนดูปกติอาจหลุด
- เตือนผิดกับเว็บปกติจากแหล่งอื่นค่อนข้างสูง เพราะ label ใน Kaggle มีข้อมูล PhishStorm ที่ถูกสลับ label (14.7% ของไฟล์) เช่น `github.com` ถูกทายว่า phishing
- ป้าย defacement ของ dataset ติดทั้งเว็บที่เคยถูกเจาะ ไม่ใช่เฉพาะหน้าที่ถูกเจาะ
- ผลเป็นการประเมินโดยโมเดล ใช้ประกอบการตัดสินใจ ไม่ใช่คำตัดสินสุดท้าย

## 8. แหล่งข้อมูล

- M. Siddhartha, *Malicious URLs Dataset*, Kaggle, 2021 — https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset
- ข้อมูลทดสอบภายนอก: PhishTank (https://phishtank.org), URLhaus (https://urlhaus.abuse.ch), DeepURLBench (https://huggingface.co/datasets/DeepInstinct/DeepURLBench)
