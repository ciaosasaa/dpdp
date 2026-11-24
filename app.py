"""Prototipe web DPdp: deteksi penyakit daun padi dari foto.

Jalankan:
    cd <PROYEK>
    <VENV> -m streamlit run app.py

Agar bisa diakses perangkat lain, pakai ./run_web.sh (app + tunnel publik).

Model dipilih otomatis: checkpoint terbaik di runs/ berdasarkan val macro-F1,
atau dari env var DPDP_CHECKPOINT.
"""

import base64
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from predict import find_best_checkpoint, load_predictor, predict  # noqa: E402

ASSETS = Path(__file__).parent / "assets"
CSS = (ASSETS / "style.css").read_text(encoding="utf-8")
SAMPLES = Path(__file__).parent / "assets" / "contoh"
# Jumlah foto di dataset sumber; dipakai hanya untuk bacaan teknis.
N_DATASET = 4794

# Foto sawah di belakang judul. Disematkan sebagai data URI karena Streamlit
# versi ini tidak melayani berkas statis lewat /static/.
HERO_IMG = Path(__file__).parent / "static" / "sawah-hd.jpg"
HERO_SRC = ""
if HERO_IMG.exists():
    HERO_SRC = "data:image/jpeg;base64," + base64.b64encode(HERO_IMG.read_bytes()).decode("ascii")

st.set_page_config(
    page_title="DPdp — Deteksi Penyakit Daun Padi",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_CKPT = find_best_checkpoint() or Path(__file__).parent / "runs" / "colab_mn12" / "best.pt"
CKPT = Path(os.environ.get("DPDP_CHECKPOINT", DEFAULT_CKPT))

DISEASE_INFO = {
    "Bacterial_Blight": {
        "gejala": (
            "Luka basah berwarna kuning keputihan di tepi daun, lalu meluas ke tengah "
            "dan ke bawah; daun layu lalu mengering (kresek). Bila batang atau pangkal "
            "daun dipotong dan dicelup ke air, keluar cairan bakteri keruh."
        ),
        "penyebab": (
            "Bakteri Xanthomonas oryzae pv. oryzae. Masuk lewat luka dan mulut daun, "
            "dibawa air hujan atau air irigasi, benih yang sudah terinfeksi, serta "
            "angin dan alat yang terkontaminasi."
        ),
        "saran": (
            "Pakai varietas tahan dan benih sehat bersertifikat; jangan memupuk nitrogen "
            "berlebihan; kelola air agar tidak terus tergenang; buang sisa tanaman "
            "sakit; lakukan rotasi tanam dan kapur bila tanah terlalu masam."
        ),
    },
    "Blast": {
        "gejala": (
            "Lesi berbentuk belah ketupat pada daun: titik tengah abu-abu keputihan "
            "dengan tepi cokelat kemerahan. Bila menyerang leher malai (neck blast), "
            "malai patah dan isinya hampa."
        ),
        "penyebab": (
            "Jamur Pyricularia oryzae (Magnaporthe oryzae). Berkembang cepat pada "
            "kelembapan tinggi, embun, dan hujan berkepanjangan; diperburuk pemupukan "
            "nitrogen berlebihan, jarak tanam rapat, dan varietas yang rentan."
        ),
        "saran": (
            "Tanam varietas tahan; kurangi pupuk nitrogen; jaga jarak tanam agar "
            "sirkulasi udara lancar; semprot fungisida (mis. tricyclazole) bila "
            "serangan berat; bersihkan sisa-sisa tanaman sakit."
        ),
    },
    "Brown_Spot": {
        "gejala": (
            "Bercak cokelat berbentuk oval hingga bulat pada daun, dengan titik tengah "
            "keabu-abuan dan tepi cokelat. Biji pun menjadi bercak dan warnanya gelap, "
            "sehingga mutu gabah menurun."
        ),
        "penyebab": (
            "Jamur Bipolaris oryzae (Cochliobolus miyabeanus). Sering muncul karena "
            "tanah miskin unsur hara, terutama kekurangan kalium, dan tanaman stres "
            "kekeringan; benih sakit ikut menularkan."
        ),
        "saran": (
            "Berikan KCl atau pupuk kandang dan pupuk berimbang untuk menambah kalium; "
            "pakai benih sehat dan varietas tahan; jaga kelembapan tanah agar tidak "
            "kering terjal; panen tepat waktu dan bersihkan sisa tanaman."
        ),
    },
    "Tungro": {
        "gejala": (
            "Daun menguning kecokelatan mulai dari ujung, pertumbuhan kerdil, anakan "
            "sedikit, dan malai tidak terisi penuh. Daun muda bisa berwarna kuning "
            "pucat serta melipat."
        ),
        "penyebab": (
            "Virus rice tungro bacilliform (RTBV) dan rice tungro spherical (RTSV) "
            "yang dibawa oleh wereng hijau (Nephotettix virescens). Wereng menghisap "
            "daun sehat lalu menularkan virus; tanam tidak serentak mempercepat penyebaran."
        ),
        "saran": (
            "Kendalikan wereng hijau sedini mungkin; tanam serentak dan gunakan "
            "varietas tahan; cabut dan musnahkan tanaman yang sudah sakit; atur waktu "
            "tanam agar tidak bersamaan dengan puncak populasi wereng."
        ),
    },
}

# Padanan Indonesia untuk nama kelas (label model tetap bahasa Inggris).
CLASS_GLOSS = {
    "Bacterial_Blight": "Hawar Bakteri",
    "Blast": "Penyakit Blas",
    "Brown_Spot": "Bercak Cokelat",
    "Tungro": "Penyakit Tungro",
}

STAGE_TICKS = (
    '<i class="tick tl"></i><i class="tick tr"></i>'
    '<i class="tick bl"></i><i class="tick br"></i>'
)

# Gambar spesimen bergerak mengikuti gulungan halaman, lalu dimiringkan dalam
# perspektif.
SCROLL_FX = """
<script>
(() => {
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const doc = window.parent && window.parent.document ? window.parent.document : document;
  const lerp = (a, b, t) => a + (b - a) * t;
  const viewH = () => (window.parent && window.parent !== window
    ? window.parent.innerHeight : window.innerHeight);

  // drift: pergeseran vertikal sebagai fraksi tinggi bingkai.
  // rx/ry: kemiringan maksimum saat bingkai di ujung gulungan.
  const JOBS = [
    { sel: '[data-hm="hero"]',        drift: 0.10, rx: 9,  ry: 6 },
    { sel: '[data-hm="stage"]',       drift: 0.14, rx: 12, ry: 9 },
    { sel: '[data-testid="stImage"]', drift: 0.10, rx: 10, ry: 7 },
  ];
  const state = new Map();

  const stat = (el) => {
    let s = state.get(el);
    if (!s) {
      s = { cur: { py: 0, rx: 0, ry: 0 }, tgt: { py: 0, rx: 0, ry: 0 }, off: { x: 0, y: 0 } };
      state.set(el, s);
    }
    return s;
  };

  doc.addEventListener('pointermove', (e) => {
    for (const j of JOBS) {
      for (const el of doc.querySelectorAll(j.sel)) {
        const r = el.getBoundingClientRect();
        if (!r.width) continue;
        const s = stat(el);
        s.off.x = ((e.clientX - r.left) / r.width - 0.5) * 5;
        s.off.y = ((e.clientY - r.top) / r.height - 0.5) * -4;
      }
    }
  });

  const tick = () => {
    const vh = viewH();
    for (const j of JOBS) {
      for (const el of doc.querySelectorAll(j.sel)) {
        const img = el.querySelector('img');
        if (!img) continue;
        const s = stat(el);
        const r = el.getBoundingClientRect();
        const p = Math.max(-1, Math.min(1,
          (r.top + r.height / 2 - vh / 2) / (vh / 2 + r.height / 2)));
        s.tgt.py = -p * r.height * j.drift;
        s.tgt.rx = p * j.rx;
        s.tgt.ry = p * j.ry;
        s.cur.py = lerp(s.cur.py, s.tgt.py, 0.09);
        s.cur.rx = lerp(s.cur.rx, s.tgt.rx + s.off.y, 0.09);
        s.cur.ry = lerp(s.cur.ry, s.tgt.ry + s.off.x, 0.09);
        img.style.setProperty('--py', s.cur.py.toFixed(1) + 'px');
        img.style.setProperty('--rx', s.cur.rx.toFixed(2) + 'deg');
        img.style.setProperty('--ry', s.cur.ry.toFixed(2) + 'deg');
      }
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
})();
</script>
"""


@st.cache_resource(show_spinner="Memuat model...")
def get_model(ckpt_str: str):
    return load_predictor(Path(ckpt_str))


def pretty(label: str) -> str:
    return label.replace("_", " ")


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def instrument_bar(meta, class_names):
    """Keping atas: wordmark + OTTO + chip INFO TEKNIS kecil di pojok kanan."""
    st.markdown(
        '<div class="plate">'
        '<div class="word">DP<span>dp</span></div>'
        '<div class="plate-right">'
        '<span class="plate-note">OTTO</span>'
        '<details class="tech-chip">'
        "<summary>INFO</summary>"
        '<div class="tech">'
        f'<div><span>Arsitektur</span><b>{meta["arch"]}</b></div>'
        f'<div><span>Epoch terpilih</span><b>{meta["epoch"]}</b></div>'
        f'<div><span>Val macro-F1</span><b>{meta["val_macro_f1"]:.3f}</b></div>'
        f'<div><span>Data Kaggle</span><b>{N_DATASET:,} foto</b></div>'
        f'<div><span>Kelas</span><b>{len(class_names)}</b></div>'
        '<div><span>Pembelajaran</span><b>transfer ImageNet</b></div>'
        '<div><span>Pemrosesan</span><b>lokal di CPU</b></div>'
        '<div><span>Sumber data</span><b>Kaggle (CC0)</b></div>'
        "</div>"
        "</details>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def masthead():
    st.markdown(
        '<div class="masthead">'
        f'<div class="hero" data-hm="hero"><img src="{HERO_SRC}" alt="Sawah padi"></div>'
        '<div class="scrim"></div>'
        '<div class="mast-body">'
        "<h1>Deteksi Penyakit Daun Padi</h1>"
        '<p class="lede">Unggah foto daun padi untuk mengenali jenis penyakitnya '
        "beserta saran penanganan. Empat kelas, satu model terukur.</p>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def stage(image, caption):
    """Panggung spesimen: gambar daun dalam bingkai terukur."""
    if image is None:
        inner = (
            '<div class="empty">BELUM ADA SPESIMEN<br>'
            "unggah foto atau pilih contoh di panel sebelah</div>"
        )
    else:
        if isinstance(image, (str, Path)):
            path = Path(image)
            data = path.read_bytes()
            ext = path.suffix.lower().lstrip(".")
        else:
            data = image
            ext = "jpg"
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
        b64 = base64.b64encode(data).decode("ascii")
        inner = f'<img src="data:{mime};base64,{b64}" alt="Daun padi">'

    st.markdown(
        f'<div class="stage">{STAGE_TICKS}'
        '<div class="stamp"><b>001</b><span>LEMBAR KERJA · DAUN PADI</span></div>'
        f'<div class="frame" data-hm="stage">{inner}</div>'
        '<div class="scale"><span></span><span>224 px</span><span>·</span>'
        '<span>skala normal</span></div>'
        f'<div class="cap">{caption}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def report(res, true_label=None):
    label = pretty(res["label"])
    conf = res["confidence"]
    now = datetime.now().strftime("%d %b %Y %H:%M")

    verdict = ""
    if true_label:
        ok = res["label"] == true_label
        verdict = (
            f'<div class="verdict {"good" if ok else "bad"}">'
            f'{"BENAR" if ok else "SALAH"} &mdash; kelas asli {pretty(true_label)}'
            "</div>"
        )
    elif not res["confident"]:
        verdict = (
            '<div class="verdict bad">KEYAKINAN DI BAWAH 50% &mdash; foto mungkin '
            "bukan daun padi, atau kelasnya tidak ada di data latih.</div>"
        )

    rows = []
    for i, r in enumerate(res["top_k"]):
        hot = " hot" if i == 0 else ""
        rows.append(
            f'<div class="trow">'
            f'<span class="tname{hot}">{pretty(r["label"])}</span>'
            f'<span class="tval{hot}">{pct(r["confidence"])}</span>'
            "</div>"
        )

    info = DISEASE_INFO.get(res["label"], {})

    st.markdown(
        '<div class="report">'
        f"{STAGE_TICKS}"
        f'<div class="rid"><span>LAPORAN SPESIMEN</span><span>{now}</span></div>'
        f'<div class="diag">{label}</div>'
        f'<div class="diag-sub">{CLASS_GLOSS.get(res["label"], "")}</div>'
        f"{verdict}"
        '<div class="measure">'
        '<div class="mhead"><span>KEYAKINAN</span>'
        f"<b>{pct(conf)}</b></div>"
        f'<div class="track"><i style="width:{conf * 100:.1f}%"></i>'
        '<u style="left:50%"></u></div>'
        "</div>"
        '<div class="field">'
        '<div class="fhead">GEJALA</div>'
        f'<div class="fbody">{info.get("gejala", "—")}</div>'
        "</div>"
        '<div class="field">'
        '<div class="fhead">PENYEBAB</div>'
        f'<div class="fbody">{info.get("penyebab", "—")}</div>'
        "</div>"
        '<div class="field">'
        '<div class="fhead">REKOMENDASI PENANGANAN</div>'
        f'<div class="fbody">{info.get("saran", "—")}</div>'
        "</div>"
        '<div class="readings">'
        '<div class="rhead">TIGA BACAAN TERATAS</div>'
        + "".join(rows)
        + "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def main():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    st.components.v1.html(SCROLL_FX, height=1, scrolling=False)

    if not CKPT.exists():
        st.error("Model tidak ditemukan. Set env var DPDP_CHECKPOINT ke lokasi best.pt.")
        st.stop()

    model, class_names, meta = get_model(str(CKPT))

    instrument_bar(meta, class_names)
    masthead()

    tab_upload, tab_sample = st.tabs(["Unggah Foto", "Coba Contoh"])

    with tab_upload:
        up = st.file_uploader(
            "PILIH FOTO DAUN",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="visible",
        )
        if up is not None:
            st.session_state["uploaded"] = up.getvalue()
            st.session_state["uploaded_name"] = up.name
            st.session_state["sample"] = None
            with st.spinner("Menganalisis..."):
                tmp = write_temp(up.getvalue(), up.name)
                try:
                    res = predict(tmp, model, class_names, img_size=meta.get("img_size", 224))
                finally:
                    tmp.unlink(missing_ok=True)
            st.session_state["result"] = (res, None)
        else:
            st.markdown(
                '<div class="panel-label">CARA PAKAI</div>'
                "<p style='margin:0;font-size:.86rem;line-height:1.7;"
                "color:var(--ink-soft)'>Letakkan foto daun di atas meja, "
                "lalu baca laporan di bawah. Foto diproses lokal di CPU.</p>",
                unsafe_allow_html=True,
            )

    with tab_sample:
        picks = {}
        for img_path in sorted(SAMPLES.glob("*.jpg")):
            label = img_path.stem
            if label in class_names:
                picks[label] = img_path

        if picks:
            st.markdown(
                '<div class="panel-label">SATU FOTO CONTOH TIAP KELAS · TEKAN UNTUK MENGANALISIS</div>',
                unsafe_allow_html=True,
            )
            cols = st.columns(len(picks))
            for col, (label, path) in zip(cols, picks.items()):
                with col:
                    st.image(str(path), use_container_width=True)
                    st.markdown(
                        f'<div class="sample-cap">{CLASS_GLOSS.get(label, pretty(label))}<br>'
                        f"<span>{pretty(label)}</span></div>",
                        unsafe_allow_html=True,
                    )
                    if st.button("ANALISIS", key=f"sample_{label}", type="primary",
                                 use_container_width=True):
                        with st.spinner("Menganalisis..."):
                            res = predict(path, model, class_names, img_size=meta.get("img_size", 224))
                        st.session_state["sample"] = str(path)
                        st.session_state["sample_label"] = label
                        st.session_state["uploaded"] = None
                        st.session_state["result"] = (res, label)
        else:
            st.error("Data contoh tidak ditemukan.")

    # ---- panggung: bahan dari tab yang benar-benar dipakai ----
    uploaded = st.session_state.get("uploaded")
    uploaded_name = st.session_state.get("uploaded_name")
    sample = st.session_state.get("sample")
    sample_label = st.session_state.get("sample_label")

    if uploaded is not None:
        specimen = uploaded
        caption = f"SPECIMEN · {uploaded_name} · {len(uploaded) // 1024} KB"
    elif sample:
        specimen = sample
        caption = f"SPECIMEN · {pretty(sample_label)} · split test"
    else:
        specimen = None
        caption = "SPECIMEN · menunggu input"


    # ---- meja: panggung + panel ----
    left, right = st.columns([1.25, 1], gap="large")

    with left:
        stage(specimen, caption)

    with right:
        st.markdown('<div class="panel-label">STATUS</div>', unsafe_allow_html=True)
        if "result" in st.session_state:
            res, true = st.session_state["result"]
            st.markdown(
                '<div class="panel" style="padding:1.1rem 1.2rem">'
                '<div class="panel-label" style="margin-bottom:.5rem">HASIL TERAKHIR</div>'
                f'<div style="font-size:1.05rem;font-weight:600;letter-spacing:-.02em">'
                f"{pretty(res['label'])}</div>"
                f'<div style="font-family:var(--mono);font-size:.72rem;'
                f'color:var(--ink-soft);margin-top:.2rem">{pct(res["confidence"])} keyakinan'
                "</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="panel" style="padding:1.1rem 1.2rem">'
                '<div class="panel-label" style="margin-bottom:.5rem">HASIL TERAKHIR</div>'
                '<div style="font-size:.84rem;color:var(--muted)">belum ada analisis</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="contact">UNTUK INFO LEBIH LANJUT HUBUNGI NOMOR DI '
            'BAWAH INI<br>'
            '<a href="tel:081909988779">0819-0998-8779</a> '
            '(Kementerian Pertanian RI)</div>',
            unsafe_allow_html=True,
        )
    # ---- laporan ----
    if "result" in st.session_state:
        res, true = st.session_state["result"]
        report(res, true_label=true)

    st.markdown(
        '<div class="foot">'
        f"<span>DPdp · {meta['arch'].upper()} · DATA RICE LEAF DISEASE (CC0)</span>"
        f"<span>{len(class_names)} KELAS · DATA KAGGLE {N_DATASET:,} FOTO · "
        f"VAL MACRO-F1 {meta['val_macro_f1']:.3f}</span>"
        "<span>FOTO SAWAH: BASILE MORIN · CC BY-SA 4.0</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def write_temp(data: bytes, name: str) -> Path:
    suffix = Path(name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return Path(tmp.name)


if __name__ == "__main__":
    main()
