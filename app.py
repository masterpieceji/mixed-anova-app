# ============================================================
#  Mixed-Design Repeated Measures ANOVA — Streamlit App
#  รองรับ pingouin ทั้ง 0.5.x และ 0.6.x
#  รัน:  streamlit run app.py
# ============================================================
import io
import numpy as np
import pandas as pd
import streamlit as st
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

st.set_page_config(page_title="Mixed Repeated Measures ANOVA", layout="wide")
sns.set_theme(style="whitegrid")

# ============================================================
#  ตัวช่วยรองรับความต่างของเวอร์ชัน pingouin
# ============================================================
COLMAP = {
    "p_unc": "p-unc", "p_corr": "p-corr", "p_adjust": "p-adjust",
    "p_gg_corr": "p-GG-corr", "p_GG_corr": "p-GG-corr",
    "p_hf_corr": "p-HF-corr", "p_HF_corr": "p-HF-corr",
    "p_spher": "p-spher", "W_spher": "W-spher",
    "ddof1": "DF1", "ddof2": "DF2", "DF": "DF1",
    "eps_gg": "eps", "np_2": "np2",
}

def norm_cols(d: pd.DataFrame) -> pd.DataFrame:
    """แปลงชื่อคอลัมน์ผลลัพธ์ให้เป็นมาตรฐานเดียวกัน (รูปแบบขีดกลาง)"""
    d = d.rename(columns={k: v for k, v in COLMAP.items() if k in d.columns})
    return d.loc[:, ~d.columns.duplicated()]

def rget(row, *names, default=np.nan):
    """ดึงค่าจากชื่อคอลัมน์แรกที่มีอยู่จริงใน Series"""
    for n in names:
        if n in row.index and pd.notna(row[n]):
            return row[n]
    return default

def pairwise(**kw):
    """pairwise_tests (ใหม่) หรือ pairwise_ttests (เก่า)"""
    fn = getattr(pg, "pairwise_tests", None) or getattr(pg, "pairwise_ttests")
    return norm_cols(fn(**kw))

def sphericity_result(data, dv, subject, within):
    """คืนค่า (ผ่านหรือไม่, W, chi2, dof, p) แบบทนทุกเวอร์ชัน"""
    r = pg.sphericity(data, dv=dv, subject=subject, within=within)
    try:
        return bool(r.spher), float(r.W), float(r.chi2), r.dof, float(r.pval)
    except AttributeError:
        return bool(r[0]), float(r[1]), float(r[2]), r[3], float(r[4])

def fmt(x, nd=3):
    return f"{x:.{nd}f}" if pd.notna(x) and np.isfinite(x) else "—"

# ============================================================
#  Header
# ============================================================
st.title("Mixed-Design Repeated Measures ANOVA")
st.caption(f"Between-subjects × Within-subjects  •  pingouin v{pg.__version__}")

# ============================================================
#  1. นำเข้าข้อมูล
# ============================================================
with st.sidebar:
    st.header("1. นำเข้าข้อมูล")
    up = st.file_uploader("อัปโหลด CSV / Excel", type=["csv", "xlsx", "xls"])
    demo = st.checkbox("ใช้ข้อมูลตัวอย่าง", value=up is None)

@st.cache_data
def make_demo(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g, base, slope in [("Control", 50, 0.5), ("Treatment", 50, 4.0)]:
        for i in range(30):
            b = rng.normal(0, 5)
            for t, name in enumerate(["Pre", "Post", "FollowUp"]):
                rows.append({"id": f"{g[0]}{i:02d}", "group": g, "time": name,
                             "score": base + b + slope * t + rng.normal(0, 3)})
    return pd.DataFrame(rows)

@st.cache_data
def load(file) -> pd.DataFrame:
    return pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)

if up is not None:
    try:
        raw = load(up)
    except Exception as e:
        st.error(f"อ่านไฟล์ไม่สำเร็จ: {e}")
        st.stop()
elif demo:
    raw = make_demo()
else:
    st.info("กรุณาอัปโหลดไฟล์ หรือติ๊ก 'ใช้ข้อมูลตัวอย่าง' ที่แถบด้านซ้าย")
    st.stop()

st.subheader("ตัวอย่างข้อมูลที่นำเข้า")
st.dataframe(raw.head(10), use_container_width=True)

# ============================================================
#  2. กำหนดโครงสร้างข้อมูล
# ============================================================
def idx_of(cols, name, fallback=0):
    return cols.index(name) if name in cols else min(fallback, len(cols) - 1)

with st.sidebar:
    st.header("2. โครงสร้างข้อมูล")
    cols = list(raw.columns)
    default_long = {"id", "time"}.issubset(set(cols))
    layout = st.radio("รูปแบบข้อมูล", ["Long format", "Wide format"],
                      index=0 if default_long else 1)

    if layout == "Long format":
        subj    = st.selectbox("Subject ID", cols, index=idx_of(cols, "id", 0))
        between = st.selectbox("Between-subjects factor (กลุ่ม)", cols, index=idx_of(cols, "group", 1))
        within  = st.selectbox("Within-subjects factor (เวลา/เงื่อนไข)", cols, index=idx_of(cols, "time", 2))
        dv      = st.selectbox("Dependent variable (ตัวแปรตาม)", cols, index=idx_of(cols, "score", 3))
        if len({subj, between, within, dv}) < 4:
            st.error("กรุณาเลือกคอลัมน์ให้ไม่ซ้ำกัน"); st.stop()
        df = raw[[subj, between, within, dv]].copy()
        within_order = None
    else:
        subj    = st.selectbox("Subject ID", cols)
        between = st.selectbox("Between-subjects factor", [c for c in cols if c != subj])
        rep     = st.multiselect("คอลัมน์การวัดซ้ำ (เรียงตามลำดับ)",
                                 [c for c in cols if c not in (subj, between)])
        if len(rep) < 2:
            st.warning("เลือกคอลัมน์การวัดซ้ำอย่างน้อย 2 คอลัมน์"); st.stop()
        within, dv = "time", "value"
        df = raw.melt(id_vars=[subj, between], value_vars=rep,
                      var_name=within, value_name=dv)
        within_order = rep

    st.header("3. ตัวเลือกการวิเคราะห์")
    alpha = st.number_input("ระดับนัยสำคัญ (α)", 0.001, 0.20, 0.05, 0.005, format="%.3f")
    padj  = st.selectbox("การปรับค่า p (post-hoc)",
                         ["bonf", "holm", "fdr_bh", "sidak", "none"], index=0)
    show_traj = st.checkbox("แสดงกราฟเส้นทางรายบุคคล", value=True)

# ---- ทำความสะอาดข้อมูล ----
df[subj]    = df[subj].astype(str)
df[between] = df[between].astype(str)
df[dv]      = pd.to_numeric(df[dv], errors="coerce")
if within_order:
    df[within] = pd.Categorical(df[within], categories=within_order, ordered=True)
else:
    df[within] = df[within].astype(str)
df = df.dropna(subset=[dv, subj, between, within])

if df.empty:
    st.error("ไม่มีข้อมูลเหลือหลังทำความสะอาด — ตรวจสอบการเลือกคอลัมน์อีกครั้ง"); st.stop()

# ---- ตรวจความครบถ้วนของการวัดซ้ำ ----
k_within = df[within].nunique()
if k_within < 2:
    st.error("Within-subjects factor ต้องมีอย่างน้อย 2 ระดับ"); st.stop()
if df[between].nunique() < 2:
    st.error("Between-subjects factor ต้องมีอย่างน้อย 2 กลุ่ม"); st.stop()

counts = df.groupby(subj, observed=True)[within].nunique()
incomplete = counts[counts < k_within].index.tolist()
if incomplete:
    st.warning(f"ตัดผู้เข้าร่วม {len(incomplete)} รายที่ข้อมูลไม่ครบทุกเงื่อนไขออก (listwise deletion)")
    df = df[~df[subj].isin(incomplete)]

dups = df.duplicated(subset=[subj, within]).sum()
if dups:
    st.warning(f"พบข้อมูลซ้ำ {dups} แถว (subject × condition) — ใช้ค่าเฉลี่ยแทน")
    df = df.groupby([subj, between, within], observed=True, as_index=False)[dv].mean()

if df[subj].nunique() < 3:
    st.error("จำนวนผู้เข้าร่วมน้อยเกินไปสำหรับการวิเคราะห์"); st.stop()

order = (list(df[within].cat.categories)
         if isinstance(df[within].dtype, pd.CategoricalDtype)
         else sorted(df[within].unique()))

c1, c2, c3 = st.columns(3)
c1.metric("จำนวนผู้เข้าร่วม", df[subj].nunique())
c2.metric(f"กลุ่ม ({between})", df[between].nunique())
c3.metric(f"จุดวัดซ้ำ ({within})", k_within)

# ============================================================
#  แท็บผลลัพธ์
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 สถิติพรรณนา", "✅ ข้อตกลงเบื้องต้น", "📈 Mixed ANOVA", "🔍 Post-hoc", "📉 กราฟ"])

# ---------------------------------------------------- สถิติพรรณนา
with tab1:
    desc = (df.groupby([between, within], observed=True)[dv]
              .agg(n="count", Mean="mean", SD="std", Median="median",
                   Min="min", Max="max").reset_index())
    desc["SE"] = desc["SD"] / np.sqrt(desc["n"])
    desc["CI95_low"]  = desc["Mean"] - 1.96 * desc["SE"]
    desc["CI95_high"] = desc["Mean"] + 1.96 * desc["SE"]
    st.dataframe(desc.round(3), use_container_width=True)

    st.markdown("#### ตารางค่าเฉลี่ย (Group × Time)")
    st.dataframe(
        df.pivot_table(index=between, columns=within, values=dv,
                       aggfunc="mean", observed=True).round(3),
        use_container_width=True)

# ---------------------------------------------------- ข้อตกลงเบื้องต้น
with tab2:
    a1, a2 = st.columns(2)

    with a1:
        st.markdown("#### 1) Normality — Shapiro–Wilk (รายเซลล์)")
        rows = []
        for (g, t), sub in df.groupby([between, within], observed=True):
            if len(sub) >= 3:
                W, p = stats.shapiro(sub[dv])
                rows.append({between: g, within: t, "n": len(sub),
                             "W": round(W, 4), "p": round(p, 4),
                             "ผลลัพธ์": "✅ ปกติ" if p > alpha else "⚠️ ไม่ปกติ"})
        norm_tbl = pd.DataFrame(rows)
        st.dataframe(norm_tbl, use_container_width=True)
        if not norm_tbl.empty and (norm_tbl["p"] <= alpha).any():
            st.caption("⚠️ บางเซลล์ไม่เป็นโค้งปกติ — ANOVA ค่อนข้างทนทาน (robust) หากขนาดกลุ่มใกล้เคียงกันและ n ≥ 20")

        st.markdown("#### 2) Homogeneity of Variance — Levene")
        lev_rows = []
        for t, sub in df.groupby(within, observed=True):
            grps = [g[dv].values for _, g in sub.groupby(between, observed=True)]
            if len(grps) >= 2 and all(len(x) >= 2 for x in grps):
                F, p = stats.levene(*grps, center="median")
                lev_rows.append({within: t, "F": round(F, 4), "p": round(p, 4),
                                 "ผลลัพธ์": "✅ เท่ากัน" if p > alpha else "⚠️ ไม่เท่ากัน"})
        st.dataframe(pd.DataFrame(lev_rows), use_container_width=True)

    with a2:
        st.markdown("#### 3) Sphericity — Mauchly's Test")
        if k_within == 2:
            st.info("มีการวัดซ้ำเพียง 2 ระดับ — เป็นไปตามข้อตกลงโดยอัตโนมัติ")
        else:
            try:
                ok, W, chi2, dof, pval = sphericity_result(df, dv, subj, within)
                st.dataframe(pd.DataFrame([{
                    "W": round(W, 4), "chi²": round(chi2, 4), "df": dof,
                    "p": round(pval, 4),
                    "ผลลัพธ์": "✅ ผ่าน" if ok else "⚠️ ไม่ผ่าน"}]),
                    use_container_width=True)
                if not ok:
                    st.warning("ไม่ผ่านข้อตกลง → ให้อ่านค่า **p-GG-corr** (Greenhouse–Geisser) ในตาราง ANOVA แทน")
            except Exception as e:
                st.caption(f"คำนวณไม่ได้: {e}")

        st.markdown("#### 4) Box's M — ความเท่ากันของ Covariance Matrices")
        try:
            wide = (df.pivot_table(index=[subj, between], columns=within,
                                   values=dv, observed=True)
                      .dropna().reset_index())
            dvs = [c for c in wide.columns if c not in (subj, between)]
            box = norm_cols(pg.box_m(wide, dvs=dvs, group=between))
            st.dataframe(box.round(4), use_container_width=True)
            pcol = "p-unc" if "p-unc" in box.columns else ("pval" if "pval" in box.columns else None)
            if pcol and box[pcol].iloc[0] <= 0.001:
                st.caption("⚠️ p ≤ .001 — พิจารณาใช้ Pillai's Trace (MANOVA) ประกอบ")
        except Exception as e:
            st.caption(f"คำนวณไม่ได้: {e}")

# ---------------------------------------------------- Mixed ANOVA
with tab3:
    try:
        aov = norm_cols(pg.mixed_anova(data=df, dv=dv, within=within,
                                       between=between, subject=subj,
                                       correction=True, effsize="np2"))
    except Exception as e:
        st.error(f"วิเคราะห์ไม่สำเร็จ: {e}"); st.stop()

    st.markdown("#### ตารางผลการวิเคราะห์")
    st.dataframe(aov.round(4), use_container_width=True)
    with st.expander("ชื่อคอลัมน์ที่ได้จริง (สำหรับตรวจสอบ)"):
        st.write(list(aov.columns))

    st.markdown("#### สรุปผลเชิงข้อความ")
    for _, r in aov.iterrows():
        src  = r["Source"]
        p_gg = rget(r, "p-GG-corr")
        p    = p_gg if (src == within and pd.notna(p_gg)) else rget(r, "p-unc", "p")
        eta  = rget(r, "np2", "n2", "eta-square")
        df1, df2 = rget(r, "DF1", "ddof1", "DF"), rget(r, "DF2", "ddof2")
        F = rget(r, "F")
        if pd.isna(p):
            continue
        sig = "**มีนัยสำคัญ**" if p < alpha else "ไม่มีนัยสำคัญ"
        mag = ("ใหญ่" if eta >= .14 else "ปานกลาง" if eta >= .06 else "เล็ก") if pd.notna(eta) else "—"
        note = " *(ปรับด้วย GG)*" if (src == within and pd.notna(p_gg)) else ""
        st.write(f"- **{src}** — F({fmt(df1,0)}, {fmt(df2,0)}) = {fmt(F)}, "
                 f"p = {fmt(p,4)}{note} → {sig}  •  η²p = {fmt(eta)} (ขนาดอิทธิพล{mag})")

    inter = aov[aov["Source"].astype(str).str.lower() == "interaction"]
    if not inter.empty:
        p_int = rget(inter.iloc[0], "p-unc", "p")
        if pd.notna(p_int) and p_int < alpha:
            st.success("🔎 พบ **Interaction** อย่างมีนัยสำคัญ → ควรตีความผ่าน Simple Main Effects (แท็บ Post-hoc) มากกว่าดู main effects โดยลำพัง")
        else:
            st.info("ไม่พบ Interaction ที่มีนัยสำคัญ → สามารถตีความ Main Effects ได้โดยตรง")

# ---------------------------------------------------- Post-hoc
with tab4:
    st.markdown("#### Pairwise Comparisons")
    try:
        ph = pairwise(data=df, dv=dv, within=within, between=between,
                      subject=subj, padjust=padj, effsize="hedges")
        st.dataframe(ph.round(4), use_container_width=True)
    except Exception as e:
        ph = pd.DataFrame()
        st.error(f"คำนวณ post-hoc ไม่สำเร็จ: {e}")

    st.markdown("#### Simple Main Effects")
    s1, s2 = st.columns(2)

    with s1:
        st.caption(f"เปรียบเทียบ **{within}** ภายในแต่ละ **{between}**")
        out = []
        for g, sub in df.groupby(between, observed=True):
            try:
                a = norm_cols(pg.rm_anova(sub, dv=dv, within=within, subject=subj,
                                          correction=True, effsize="np2"))
                a.insert(0, between, g); out.append(a)
            except Exception:
                pass
        st.dataframe(pd.concat(out).round(4) if out else pd.DataFrame(),
                     use_container_width=True)

    with s2:
        st.caption(f"เปรียบเทียบ **{between}** ณ แต่ละ **{within}**")
        out = []
        for t, sub in df.groupby(within, observed=True):
            try:
                a = norm_cols(pg.anova(sub, dv=dv, between=between, effsize="np2"))
                a.insert(0, within, t); out.append(a)
            except Exception:
                pass
        st.dataframe(pd.concat(out).round(4) if out else pd.DataFrame(),
                     use_container_width=True)

# ---------------------------------------------------- กราฟ
with tab5:
    g1, g2 = st.columns(2)

    with g1:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        sns.pointplot(data=df, x=within, y=dv, hue=between, order=order,
                      errorbar=("ci", 95), dodge=0.15, capsize=0.08, ax=ax)
        ax.set_title("Interaction Plot (Mean ± 95% CI)")
        ax.legend(title=between, frameon=True)
        st.pyplot(fig); plt.close(fig)

    with g2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.2))
        sns.boxplot(data=df, x=within, y=dv, hue=between, order=order, ax=ax2)
        sns.stripplot(data=df, x=within, y=dv, hue=between, order=order,
                      dodge=True, alpha=0.35, size=3, legend=False, ax=ax2)
        ax2.set_title("การกระจายของข้อมูล (Group × Time)")
        st.pyplot(fig2); plt.close(fig2)

    if show_traj:
        pos = {v: i for i, v in enumerate(order)}
        fig3, ax3 = plt.subplots(figsize=(8, 4.4))
        palette = dict(zip(sorted(df[between].unique()),
                           sns.color_palette(n_colors=df[between].nunique())))
        for (g, s), ind in df.groupby([between, subj], observed=True):
            ind = ind.sort_values(within, key=lambda c: c.map(pos))
            ax3.plot(ind[within].map(pos), ind[dv], color=palette[g], alpha=0.18, lw=0.8)
        means = df.groupby([between, within], observed=True)[dv].mean().reset_index()
        for g, sub in means.groupby(between, observed=True):
            sub = sub.sort_values(within, key=lambda c: c.map(pos))
            ax3.plot(sub[within].map(pos), sub[dv], color=palette[g],
                     lw=3, marker="o", markersize=7, label=g)
        ax3.set_xticks(range(len(order))); ax3.set_xticklabels(order)
        ax3.set_xlabel(within); ax3.set_ylabel(dv)
        ax3.set_title("เส้นทางรายบุคคล + ค่าเฉลี่ยรายกลุ่ม")
        ax3.legend(title=between)
        st.pyplot(fig3); plt.close(fig3)

# ============================================================
#  Export
# ============================================================
st.divider()
try:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        desc.to_excel(w, sheet_name="Descriptives", index=False)
        aov.to_excel(w, sheet_name="MixedANOVA", index=False)
        if not ph.empty:
            ph.to_excel(w, sheet_name="PostHoc", index=False)
        df.to_excel(w, sheet_name="CleanedData", index=False)
    st.download_button("⬇️ ดาวน์โหลดผลลัพธ์ทั้งหมด (.xlsx)", buf.getvalue(),
                       "mixed_anova_results.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
except Exception as e:
    st.caption(f"สร้างไฟล์ดาวน์โหลดไม่สำเร็จ: {e}")