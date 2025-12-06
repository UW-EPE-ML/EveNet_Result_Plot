# EveNet_Result_Plot

# ⭐ **Final Naming Table (Nature-Style, Jargon-Free, Consistent)**

## **1. Model Variants**

We evaluate four variants of EveNet that differ in their pretraining strategy:

* **Fully pretrained EveNet model**
  *Self-supervised + supervised pretraining*
  **Short-form:** `EveNet–Full`

* **Self-supervised–only EveNet model**
  *Stage-1 SSL pretraining only*
  **Short-form:** `EveNet–SSL`

* **Ablation-pretrained EveNet variant**
  *Pretrained without the supervised generative objective*
  **Short-form:** `EveNet–Abl.`

* **EveNet trained from random initialization**
  *No pretraining*
  **Short-form:** `EveNet–Scratch`

These names refer exclusively to the **pretrained backbone weights**.

| Concept                            | **Long-form (for main text)**                                    | **Short-form (for figures and tables)** | Notes                                         |
|------------------------------------|------------------------------------------------------------------|-----------------------------------------|-----------------------------------------------|
| Fully pretrained model (“Nominal”) | **fully pretrained model** (self-supervised + supervised stages) | **Full**                                | Nature prefers plain English (“full model”).  |
| Self-supervised only (“SSL”)       | **self-supervised–only pretrained model**                        | **SSL**                                 | Standard ML term; safe.                       |
| Ablation-pretrained model          | **ablation-pretrained variant** (pretrained with segmentation)   | **Abl.** or **Abl-Seg**                 | Use “Abl.” unless you need explicit “Abl-SG.” |
| Random initialization (“Scratch”)  | **model trained from random initialization**                     | **Scratch**                             | Universally understood.                       |

---

## **2. Task Heads**

| Concept                         | **Long-form (for main text)**        | **Short-form (for captions)** | Notes                                                    |
|---------------------------------|--------------------------------------|-------------------------------|----------------------------------------------------------|
| Classification head             | **classification head**              | **Cls**                       | “Classification” is accessible; “Cls” is concise.        |
| Assignment head                 | **object–resonance assignment head** | **Asn**                       | Avoid jargon (“assignment” alone may confuse beginners). |
| Segmentation head               | **final-state segmentation head**    | **Seg**                       | Standard terminology.                                    |
| Supervised generative head      | **supervised generative head**       | **SG**                        | “SG” clearer than “SGEN” for figures.                    |
| Self-supervised generative head | **self-supervised generative head**  | **SSG**                       | Parallel with SG.                                        |

**Combinations for caption labels:**
Cls + Asn, Cls + SG, Asn + Seg, etc.
(Best practice: Title-case them, not all caps.)

| Category                      | Long-form (main text)                      | Short-form (figures/tables) |
|-------------------------------|--------------------------------------------|-----------------------------|
| Fully pretrained EveNet       | fully pretrained EveNet model              | `EveNet–Full`               |
| SSL-only EveNet               | self-supervised–only EveNet model          | `EveNet–SSL`                |
| Ablation EveNet               | ablation-pretrained EveNet variant         | `EveNet–Abl.`               |
| Scratch EveNet                | EveNet trained from random initialization  | `EveNet–Scratch`            |
| Classification head           | classification head                        | `Cls`                       |
| Assignment head               | object–resonance assignment head           | `Asn`                       |
| Segmentation head             | final-state segmentation head              | `Seg`                       |
| Supervised generation         | supervised generative head                 | `SG`                        |
| Self-supervised generation    | self-supervised generative head            | `SSG`                       |
| Quantum-correlation benchmark | top-quark quantum-correlation analysis     | `QC`                        |
| Rare-Higgs benchmark          | rare Higgs-boson decay benchmark (H→2a→4b) | `RH` / `H→2a→4b`            |
| Anomaly detection             | dimuon anomaly-detection benchmark         | `AD`                        |

---

## **3. Downstream Benchmarks**

| Study                                                     | **Long-form (main text)**                                                               | **Short-form (figures)** | Notes                                                    |
|-----------------------------------------------------------|-----------------------------------------------------------------------------------------|--------------------------|----------------------------------------------------------|
| Top-quark quantum-correlation / entanglement study (“QE”) | **top-quark quantum-correlation analysis** (includes spin correlation and entanglement) | **QC**                   | Avoid “QE” (“quantum efficiency”). QC is clear.          |
| Exotic Higgs / BSM decay benchmark                        | **rare Higgs-boson decay benchmark (H→2a→4b)**                                          | **RH** or **H→2a→4b**    | “Rare Higgs decay” avoids jargon like “exotic” or “BSM”. |
| Anomaly detection (“AD”)                                  | **dimuon anomaly-detection benchmark using real collision data**                        | **AD**                   | Clear even outside HEP.                                  |

---

# ⭐ **4. Combined Overview Table (Full Final Table)**

Below is the single integrated table you can drop into your Supplementary Methods or into a Consistency Appendix.

---

## **📘 Final Jargon-Free Naming Table for the Foundation-Model Paper**

| Category            | Concept                             | **Long-form description (main text)**                       | **Short-form (for figures/tables)** |
|---------------------|-------------------------------------|-------------------------------------------------------------|-------------------------------------|
| **Model Variant**   | Fully pretrained                    | fully pretrained model                                      | **Full**                            |
|                     | Self-supervised only                | self-supervised–only pretrained model                       | **SSL**                             |
|                     | Ablation-pretrained                 | ablation-pretrained variant (without supervised generation) | **Abl.** / **Abl-SG**               |
|                     | Scratch                             | model trained from random initialization                    | **Scratch**                         |
| **Task Head**       | Classification                      | classification head                                         | **Cls**                             |
|                     | Assignment                          | object–resonance assignment head                            | **Asn**                             |
|                     | Segmentation                        | final-state segmentation head                               | **Seg**                             |
|                     | Supervised generation               | supervised generative head                                  | **SG**                              |
|                     | Self-supervised generation          | self-supervised generative head                             | **SSG**                             |
| **Downstream Task** | Quantum correlation (formerly “QE”) | top-quark quantum-correlation analysis                      | **QC**                              |
|                     | Rare Higgs decay (formerly “BSM”)   | rare Higgs-boson decay benchmark (H→2a→4b)                  | **RH** or **H→2a→4b**               |
|                     | Anomaly detection                   | dimuon anomaly-detection benchmark using real data          | **AD**                              |

## 🔗 Plot gallery website

This repository now includes a small static gallery that renders the paper plots into PNGs and publishes them to GitHub Pages. The CI workflow builds the gallery on every push to `main`.

### Local preview

```bash
python -m pip install -r requirements.txt
python website/build_gallery.py --output-dir site --format png --dpi 200
python -m http.server --directory site 8000
```

Open http://localhost:8000 to browse the same gallery that will appear on GitHub Pages.


