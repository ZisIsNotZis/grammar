# grammar

Masked-token (denoising) sequence models over a 90-symbol tokenised **source-code**
corpus: how far can a pure-convolution, U-Net-shaped encoder–decoder push
masked-token reconstruction of code, with no recurrence and no attention?

## What this is / why it exists

This is a 2026 personal research scratchpad, not a library. It attacks one
question: **if you hand a segmentation-style encoder–decoder a sequence of code
tokens and ask it to fill in the blanks, how good does it get, and which
architectural choices matter?** The bet behind it — inherited from the
`salsanext`/`encnet`-style segmentation nets the author was reading at the time —
is that an image-style U-Net over token embeddings plus 1×1 group-convolution
"bottleneck" blocks is a competitive sequence model, and that the useful knob is
*topology* (how deep, where the skip connections go, how the down/up-sampling is
parameterised) rather than width or attention.

The directory is called `grammar` because the corpus is Python source and the
objective is to learn its grammar by reconstruction. That reading is inferred
purely from the code — **not one notebook contains a prose description of the
project.** The whole directory has seven markdown cells: three are model
definitions pasted in as markdown instead of code, and four are the one-line
result annotations quoted below.

## The model

Every notebook redefines the same small set of hand-rolled `torch.nn` containers.
There is no shared module; each notebook is self-contained. The canonical
definitions (as they appear in `Untitled-Copy2.ipynb`) are:

| Name | What it is |
| --- | --- |
| `F` | Thin wrapper turning a function into a `Module` (`s.f(x)` or `s.f(x, s.a)`). |
| `EncDec(n, a, b, c)` | The core U-Net: `n` encoder stages built by factory `a`, a bottleneck `b`, then `n` decoder stages `c` applied in reverse with a skip add — `x = c[i](b(x) + skip)`. In some notebooks the signature is `EncDec(n, a, b)` with the same body. |
| `Add` / `Mul` | `Sequential` that fans out and reduces with `add_` / `mul_` (multi-branch residual blocks). |
| `Res` | `Sequential` with a residual add: `super().forward(x) + x`. |
| `Rec(n, f)` | Builds a nested residual stack by folding `f` over itself `n` times. |
| `Nn(i, o, b, h, a)` | 1×1 bottleneck block: `Conv1d(i, i*h, 1) → BatchNorm1d → act → Conv1d(i*h, o, 1)`. |
| `SwiGLU(i, o, b, h, a)` | Same but gated (two 1×1 branches multiplied), pre-norm `BatchNorm1d(affine=False)`. |
| `Enc2` / `Dec2` | Expanded separable-conv encoder/decoder stage (two-branch `_a`/`_b`/`ab` wiring plus `a_`/`b_`/`__` shortcuts), with `MaxPool1d(2)` on the encoder side and `ConvTranspose1d(m, m, 3, 2, 1, 1)` on the decoder side. |
| `EncDec6` | A partially-written deeper variant (only `m65`/`m54` stages are defined). |

The assembled models all follow the same shape:

```
Embedding(90, m) → transpose(1,2) → EncDec(...) → (BatchNorm → act) → Conv1d(m, 90, 1)
```

i.e. tokens → `m`-dim embeddings → a U-Net that halves sequence length per stage
via `MaxPool1d(2)` and doubles it back via `ConvTranspose1d(3, 2, 1, 1)` → a
per-position 90-way classifier. Depth `n` ranges from 2 to 9. Two down-sampling
idioms are compared: stride-2 `ConvTranspose1d` versus
`F(lambda x: x.repeat_interleave(2, 2))`.

## The training loop

Identical in every notebook, ~10 lines:

```python
o = torch.optim.Adam(m.parameters())
O = torch.optim.lr_scheduler.OneCycleLR(o, .01, len(a)*10//(8192*a.size(1)),
                                        pct_start=.1, div_factor=9, final_div_factor=99)
for x in tqdm(range(O.total_steps//100)):
    for x in range(100):
        y = a[torch.randint(len(a), [8192])].long()          # random windows
        x = m(y.put(torch.randint(y.numel(), [y.numel()//2]),
                    torch.randint(90, [y.numel()//2])))       # mask 1/2 with random ids
        functional.cross_entropy(x, y).backward()
        o.step(); o.zero_grad(); O.step()
    print(int((x.argmax(1)==y).sum()*1000/y.numel()), end=' ')  # top-1 accuracy in per-mille
```

So: **denoising autoencoding, not autoregressive LM.** Random windows of length
32–512, a fraction of positions (1/2 in `Untitled-Copy2`, 3/4 in
`Untitled-Copy10`/`Untitled.ipynb`) overwritten with uniformly random token ids,
`cross_entropy` on all positions, Adam + `OneCycleLR`. Reported metric is top-1
token accuracy, printed in per-mille (`*1000`).

Several notebooks append that number, one float per line, to a log file:
`open('10.txt','a')`, `open('10b.txt','a')`, `open('10c.txt','a')`.

## The data

`Untitled-Copy4`, `-Copy10`, `-Copy2`, `-Copy5`, `-Copy6`, `-Copy7`, `-Copy8`,
`-Copy9` and `Untitled.ipynb` all load

```python
a = np.memmap('../dat/py/all.90', mode='r')
```

relative to the notebook's parent directory, then window it with
`as_strided`/`unfold`. `all.90` is a **90-symbol tokenised Python-source corpus**
(the vocab size 90 is baked into every `Embedding(90, ...)` and the final
classifier). **It is not included here and is not recoverable from this
directory** — `../dat/py/` lives in the author's private parent archive. Nothing
in these notebooks regenerates it, so the training loops cannot be re-run
end-to-end without that file.

### The three `.txt` files are training logs, not a corpus

`10.txt` (683 lines), `10b.txt` (576), `10c.txt` (573) each contain one float per
line — top-1 accuracy values appended by the loops above. They are kept because
they are small (34 KB total) and they are **the only recorded results in the
directory**. They are read-only evidence; nothing in the repo writes to them any
more (the source notebooks did, via `mode='a'`).

Sample from `10.txt`:

```
0.7937841796875
0.7913525390625
0.792958984375
```

## The only written results

Four annotation cells survive in the notebooks:

* `Untitled-Copy10.ipynb`: `2->0.43 4->0.54 8->0.58 16->0.62 32->0.67 64->0.72 128->0.78`
  — accuracy rising with sequence length 2→128.
* `Untitled-Copy2.ipynb`: `CALD>CDA 0>Separable RES>0` / `CLA>LAC RES>0` / `LAL>L RES>0`
  — ordering of architecture variants (and `RES>0` = residual connections help).
* `Untitled-Copy2.ipynb`: a bare `800`.
* `Untitled-Copy4.ipynb`: a bare `67`.

No other outputs are stored: **every notebook in this directory has all cell
outputs cleared** (0 stream / 0 execute_result / 0 images). No markdown explains
any of this.

## How to run

Notebook-only. There is **no single entrypoint** and no package to install.

```bash
pip install torch numpy tqdm jupyterlab      # or: pip install torch numpy tqdm jupyter
jupyter lab
```

Then open a notebook and run all cells.

### Read this before running

* **Every notebook is CUDA-only.**
  `torch.set_default_tensor_type(torch.cuda.FloatTensor)` is set at the top, and
  there is no `.cpu()` fallback. On a CPU-only machine you must delete or edit
  that line.
* **The data is missing.** You must place a 90-symbol tokenised corpus at
  `../dat/py/all.90` (i.e. one directory above this repo) or rewrite the
  `np.memmap('../dat/py/all.90')` path in each notebook.
* Some cells `try: torch.load('10h.pk')` / `'10g.pk'` / `'10d.pk'` / `'10.pk'` /
  `'tmp.pk'`. **None of those checkpoints exist here**; the `FileNotFoundError`
  branch rebuilds a model from scratch, which is the intended path when you have
  no checkpoint. `Untitled-Copy2.ipynb` cell 3 is the exception — it calls
  `torch.load('tmp.pk')` with no guard and will fail.
* Log-appending cells (`open('10.txt','a')`) will keep growing the `.txt` files.
  Redirect them if you want the shipped logs to stay byte-identical.
* `matplotlib` is not imported anywhere in this directory, so `tqdm.notebook`
  plus a live Jupyter session is the only progress feedback.

### Which notebook should I read?

**Canonical: [`Untitled-Copy2.ipynb`](Untitled-Copy2.ipynb)** — 23 cells, ~1000
code lines, the largest and most self-contained. It defines the clean
`F`/`EncDec`/`Add`/`Res` set, then sweeps depth, kernel size, separable vs dense
convs, residual vs plain, width (`m`), and parameter counts, with the ablation
annotations in place. Start here.

Close second: **[`Untitled-Copy10.ipynb`](Untitled-Copy10.ipynb)** (~756 lines) —
the cleanest reusable toolkit (`F`, `Add`, `Mul`, `Res`, `Rec`, `EncDec`, `Nn`,
`SwiGLU`) and the only notebook with the sequence-length scaling annotation. Uses
a deep `Rec(9, ...)` U-Net with depthwise group convolutions and 3/4 masking.

Quick map of the rest: `Untitled.ipynb` and `Untitled-Copy1.ipynb` are earlier
versions of the same masked-token model; `Untitled-Copy3`/`-Copy5`/`-Copy6`/
`-Copy7`/`-Copy8`/`-Copy9`/`-Copy11` are single-configuration variants;
`Untitled-Copy4.ipynb` has an `EncDec` with an extra `d` factory and `Mul`;
`Untitled1.ipynb` / `Untitled1-Copy1.ipynb` are an unrelated micro-experiment
distilling a dense `Conv1d(16,16,3)` into separable conv stacks by MSE;
`Untitled2.ipynb` is another single run; `Untitled3.ipynb` is two lines
(`from time import sleep`) and is empty scratch; `Untitled4.ipynb` holds
`EncDec6` plus one run.

## What is here vs. not

**Here**

* 17 `Untitled*.ipynb` notebooks, ~3,456 effective code lines, Jupyter `nbformat` 4.5, `python3` kernel.
* 3 training-log `.txt` files (1,832 lines of floats) — the only recorded results in numeric form.
* The complete hand-rolled architecture zoo described above, re-defined per notebook.

**Not here**

* **No written-up results.** No paper, no report, no README-style notes, no
  tables, no plots. All cell outputs were cleared before archiving. The `10*.txt`
  logs and four one-line annotations are the entire recorded outcome.
* **No checkpoints.** `10.pk`, `10d.pk`, `10g.pk`, `10h.pk`, `tmp.pk` are
  referenced but absent, so the sweeps cannot be resumed or the numbers
  reproduced without retraining.
* **No data.** `../dat/py/all.90` is outside the archive and is not generated here.
* **No packaging.** No `setup.py`, `pyproject.toml`, tests, CLI, or importable module.
* **No git history** in the source — the directory was extracted from the
  author's private archive with no `.git`.
* **No explanation.** Zero prose describing the hypothesis; the "grammar" framing
  above is reconstructed from the corpus path and the 90-way classifier.

## Relationship to `nn.py` / `cznn`

Checked and **not applicable**: this directory contains **no `.py` files at all**,
no notebook imports `nn` or `from nn import *`, and there is no reference to
`cznn` anywhere. Nothing was deleted. The container classes (`F`, `Add`, `Mul`,
`Res`, `Rec`, `EncDec`, `Nn`, `SwiGLU`) are hand-written inline in the notebooks
and are self-contained — they are *stylistically* close to the pure-`torch.nn`
composable-block idiom of the author's published
[`cznn`](https://github.com/ZisIsNotZis/cznn) library (`Add`/`Mul`/`Res` fold
into `Sequential`), but they are not loaded from it and do not depend on it. If
these experiments are ever productised, that is the natural landing place.

## Security / redaction

The publishable copy (this repo) differs from the source archive (`$HOME/my/grammar`)
in exactly two ways:

1. `Untitled-Copy1.ipynb` cells 0–3 were **removed**. They contained a decoded
   SSO bearer JWT (`{"sub":"sso","id":49,"phone":…,"tenant":24,"role":{…"ADMIN"…},
   "iat":1646363402,"exp":1646813453}`) and the `b64decode` call used to read it —
   leftovers from an unrelated debugging session, not part of this research. The
   token is signed (HS256) and carries a phone number and an ADMIN role, so it is
   removed rather than genericised. A redaction note replaces the cells. The
   source file in the source archive is untouched; **if that token was ever
   live, rotate it.**
2. Absolute home-directory paths: **none were found** in this directory, so
   none needed genericising. All data references are already relative
   (`../dat/py/all.90`). (The only `$HOME/...` strings anywhere in this repo are
   in this README, describing the source archive.)

Note that a naive credential-gate regex (key/token assignments, bare IPv4, PEM
headers) finds **nothing** in either the source or the published copy — it does
not match a bare JWT assigned to a short variable name. The leak was found with a
JWT-shaped scan (`eyJ` + two dot-separated base64url segments + a signature).

## Status

`Status: research/archive` — a 2026 personal research scratchpad, published as-is
for provenance and readability. Not maintained, not reproducible without the
missing corpus.

**License: none.** No `LICENSE` file was present in the source and none is added
here. Treat as all-rights-reserved / personal research; add an explicit license
before relying on it or reusing the code.
