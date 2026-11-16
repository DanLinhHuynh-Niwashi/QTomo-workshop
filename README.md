# Workshop: từ mã research đến API — Quantum process tomography với `KrausTomography`

Trong workshop này bạn đi từ một thí nghiệm viết "thẳng" (không class) tới một lớp `KrausTomography`
có API rõ ràng, có test, rồi dùng chính API đó để chạy lại thí nghiệm và vẽ hình của paper
*"Advancing quantum process tomography through universal compilation"*
([arXiv:2504.14958](https://arxiv.org/abs/2504.14958)).

**Bài toán vật lý (ngắn gọn):** học các toán tử Kraus $\{K_i\}$ sao cho với mọi probe state $\rho$,
$\mathcal{E}^\dagger\, K(\rho)\, \mathcal{E} \approx \rho$, trong đó $K(\rho) = \sum_i K_i \rho K_i^\dagger$ và
$\mathcal{E}$ là một unitary Haar-random cần tìm. Tối ưu bằng gradient Riemann + Adam trên TensorFlow.

**Mục tiêu kỹ thuật phần mềm:** tách "bài toán" khỏi "thuật toán", gom trạng thái vào một class, API nhất quán,
tái lập được bằng `seed`, không side effect khi import, có test, lưu kết quả và vẽ hình tách bạch.

---

## 1. Cài đặt

### 1.1. Yêu cầu
- **Python 3.10 – 3.12** (`python --version`). Windows: khi cài nhớ tick *"Add python.exe to PATH"*.
- **Git**.
- Editor có hỗ trợ Jupyter, ví dụ **VS Code** + extension *Python* và *Jupyter*.

### 1.2. Lấy mã nguồn

```bash
git clone <URL-repo-workshop> workshop
```

```bash
cd workshop
```

Bạn làm bài trên nhánh **`main`** (mặc định). Nhánh `solution` chứa đáp án — xem [mục 5](#5-đáp-án).

### 1.3. Môi trường ảo và thư viện

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo (làm lại mỗi lần mở terminal mới):

| Hệ điều hành | Lệnh |
|---|---|
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

Cài thư viện (TensorFlow khá nặng, mất vài phút) và cài package của workshop ở chế độ editable:

```bash
pip install -r requirements.txt
```

```bash
pip install -e .
```

Đăng ký kernel Jupyter trỏ vào `.venv` (một lần duy nhất):

```bash
python -m ipykernel install --sys-prefix --name python3
```

### 1.4. Kiểm tra

```bash
python -m pytest -m "not slow"
```

Kết quả mong đợi: không có `failed` / `error`. Có một số test **SKIPPED** với lý do `chưa viết: TODO …` —
đó là phần bạn sẽ làm, hoàn toàn bình thường.

Kiểm tra `python` đúng là của môi trường ảo:

```bash
python -c "import sys; print(sys.prefix)"
```

Đường dẫn in ra phải kết thúc bằng `workshop/.venv` (hoặc `workshop\.venv`).

### 1.5. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| PowerShell chặn `Activate.ps1` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` rồi kích hoạt lại |
| `ModuleNotFoundError: No module named 'qtomo_skeleton'` | Chưa kích hoạt `.venv`, hoặc chưa `pip install -e .` |
| `AttributeError: _ARRAY_API not found` khi import TensorFlow | Đang dùng Python hệ thống thay vì `.venv` — thường do đã **di chuyển / đổi tên thư mục** sau khi tạo `.venv`. Xoá `.venv` và tạo lại |
| Notebook báo `ModuleNotFoundError` dù terminal chạy được | Kernel của notebook không phải `.venv`: chạy lại lệnh `ipykernel install` ở 1.3 và chọn kernel đó |
| `UnicodeDecodeError` khi chạy `jupyter execute` trên Windows | Đặt `PYTHONUTF8=1` trước (PowerShell: `$env:PYTHONUTF8 = "1"`) |
| `InvalidArgumentError: Input matrix is not invertible` | Đừng train bằng trạng thái thuần *chính xác* như $\lvert 0\rangle\langle 0\rvert$; dùng trạng thái Haar (mặc định) |

> Đừng di chuyển hay đổi tên thư mục `workshop/` sau khi tạo `.venv`: môi trường ảo ghi cứng đường dẫn tuyệt đối.

---

## 2. Cấu trúc thư mục

```text
workshop/
├── README.md                       # tài liệu này
├── pyproject.toml                  # cấu hình package (src layout) + pytest
├── requirements.txt                # thư viện cần cài
├── src/qtomo_skeleton/             # package bạn làm việc cùng
│   ├── tomography.py               #   ★ KrausTomography — BẠN ĐIỀN CÁC TODO Ở ĐÂY
│   ├── generator.py                #   probe state, unitary Haar, Kraus ngẫu nhiên
│   ├── quantum_ops.py              #   U ρ U†, Kraus map, kênh nhiễu
│   ├── metric.py                   #   fidelity, infidelity (loss), MSE, …
│   ├── gradient.py                 #   gradient Euclid / Riemann
│   ├── optimizer.py                #   SGD, Adam cho tham số phức
│   ├── linalg.py                   #   helper đại số tuyến tính (chuẩn hoá Kraus, …)
│   ├── writer/                     #   lưu / đọc kết quả một run dưới dạng .npz
│   └── plot/                       #   vẽ hình từ kết quả đã lưu (matplotlib)
├── experiment_run.ipynb            # ① thí nghiệm "unpack" — không class, đọc để hiểu luồng
├── experiment_run_packed.ipynb     # ③ cùng thí nghiệm viết bằng API — TODO P1–P5
├── tests/                          # pytest
│   ├── test_skeleton.py            #   ★ test KrausTomography (TODO 1–4)
│   ├── test_experiment_notebook.py #   chạy notebook với cấu hình nhỏ
│   └── test_*.py                   #   test các module còn lại
└── output/                         # (tự sinh, không commit) .npz và hình do notebook tạo
```

Những gì **đã có sẵn** và bạn chỉ cần dùng: `generator`, `quantum_ops`, `metric`, `gradient`, `optimizer`,
`linalg`, `writer`, `plot`. Những gì **bạn viết**: thân các method trong `tomography.py` (bước ②) và các ô
TODO trong `experiment_run_packed.ipynb` (bước ③).

---

## 3. Quy trình thực hành

Mở notebook trong VS Code / Jupyter từ thư mục `workshop/`, chọn kernel `.venv`, rồi *Restart & Run All*.

### Bước ① — Đọc và chạy `experiment_run.ipynb` (thí nghiệm "unpack")

Toàn bộ những gì `KrausTomography` làm được viết thẳng ra bằng hàm thuần, chia thành các khối:

| Khối | Nội dung | Sẽ trở thành |
|---|---|---|
| `[B1]`–`[B3]` | seed → RNG, trạng thái train, Kraus khởi tạo | `KrausTomography.__init__` (đã viết sẵn) |
| `[B4]` | unitary đích, `data_process_fn`, `target_process_fn` | **ngoài class** — người dùng truyền vào |
| `[B5]`–`[B7]` | hàm loss, vòng lặp tối ưu, kết quả train | `fit` (TODO 3), `is_fitted` (TODO 1) |
| `[B8]` | đánh giá trên trạng thái mới | `evaluate` (TODO 4) |
| `[B9]` | đóng gói kết quả | `TomographyResult.from_model(...)` (đã có trong `writer`) |

- **Phần 1:** một run, mỗi khối một ô.
- **Phần 2:** ghép các khối thành *hàm* `run_unpacked`, quét 1–2 qubit × nhiều run, lưu `.npz`.
- **Phần 3:** đọc lại từ đĩa, vẽ **Fig. 2** (fidelity theo iteration) và **Fig. 3** (ma trận mật độ),
  lưu vào `output/experiment_run/figures/`.

Câu hỏi nên tự trả lời khi đọc: biến nào cần "sống" qua nhiều bước (→ thuộc tính của object)?
Giá trị nào là lựa chọn của người dùng (→ tham số)? Phần nào là bài toán, phần nào là thuật toán?

### Bước ② — Điền `src/qtomo_skeleton/tomography.py`

`KrausTomography.__init__` đã viết sẵn. Các method `is_fitted`, `set_logging`, `fit`, `evaluate` đã có tên,
tham số, type hint, docstring; thân hàm chỉ là `raise NotImplementedError` kèm **TODO 1–4** chỉ rõ khối
`[B?]` cần chuyển vào và những gì phải đổi (biến cục bộ → `self.…`, `print` → `logging`, …).

Kiểm tra liên tục bằng:

```bash
python -m pytest tests/test_skeleton.py -v
```

Test của method chưa viết báo **SKIPPED** (kèm tên TODO); viết xong method nào thì test của method đó chạy
thật — PASSED, hoặc FAILED kèm lý do. Xong bước này khi không còn SKIPPED nào liên quan tới TODO.

### Bước ③ — `experiment_run_packed.ipynb`: luyện dùng API

Cùng thí nghiệm, cùng hình, nhưng viết bằng `KrausTomography` của bạn cùng `writer` và `plot`:

| TODO | Việc cần làm | Thay cho |
|---|---|---|
| P1 | dựng model `KrausTomography(...)` | `[B1]`–`[B3]` |
| P2 | `tomo.fit(...)` | `[B5]`–`[B7]` |
| P3 | `tomo.evaluate(...)` cho từng trạng thái test | `[B8]` |
| P4 | `Evaluation.from_model(...)`, `TomographyResult.from_model(...)` | `[B9]` |
| P5 | hàm `run_packed(...)` cho một run | `run_unpacked(...)` |

Phần quét + lưu (`ExperimentWriter`) và Phần 3 (đọc lại, vẽ Fig. 2, Fig. 3) đã cho sẵn. Mỗi TODO có một ô
**kiểm tra** ngay sau; chạy qua được thì làm tiếp. Với cùng seed, kết quả phải **trùng** bản unpack ở bước ①.
Ô đầu tiên báo lỗi rõ ràng nếu `tomography.py` chưa làm xong.

Gợi ý: đọc docstring bằng `help(KrausTomography.fit)`, `help(TomographyResult.from_model)`, …

### Bước ④ — Kiểm tra toàn bộ

```bash
python -m pytest
```

(~1–2 phút, gồm cả test chạy notebook.) Khi bước ② xong, không còn test nào SKIPPED vì TODO.

Chạy notebook không cần giao diện (từ `workshop/`, đã kích hoạt `.venv`; Windows nhớ `PYTHONUTF8=1`):

```bash
jupyter execute --inplace experiment_run_packed.ipynb
```

---

## 4. Quy ước code

- **Không side effect khi import**: không seed toàn cục, không tạo thư mục, không `print` trong thư viện —
  dùng `logging`.
- **Tái lập**: mọi thứ ngẫu nhiên đi qua `np.random.Generator`; lớp nhận `seed=`. Không dùng `np.random.seed`.
- **Kiểm tra input** bằng `ValueError` / `RuntimeError`, không dùng `assert` trong thư viện.
- **API nhất quán**: `num_qubits` (không phải `n`), `None` thay cho giá trị đặc biệt như `-1`, tham số tuỳ chọn
  quan trọng là keyword-only.
- **Kết quả tách khỏi hình**: tham số thí nghiệm (`seed`, `run`, …) nằm trong `metadata` của file `.npz`,
  không mã hoá vào tên file rồi parse lại; hàm vẽ trả `Figure`, không tự `show()` hay tự ghi file.
- Fidelity / loss là **số phức** trong TensorFlow: lấy phần thực bằng `float(tf.math.real(x))`.
- File `.npz` mở được bằng NumPy thuần: `np.load("output/experiment_run/1q_run0.npz")`.

Lưu ý về thuật toán: với 2 qubit và cấu hình nhỏ (3 probe state), việc train thường dừng ở cực tiểu địa phương
(loss 0.05–0.9) — đó là đặc tính của thuật toán, không phải lỗi. Các kiểm tra trong notebook chỉ đòi hỏi
trường hợp 1 qubit hội tụ.

---

## 5. Đáp án

Nhánh **`solution`** có thêm:

- `src/qtomo_skeleton/tomography-solution.py` — đáp án bước ②;
- `experiment_run_packed_solution.ipynb` — đáp án bước ③ (đã chạy sẵn, có hình).

Hãy tự làm trước. Khi muốn so sánh:

```bash
git fetch origin
```

```bash
git diff main origin/solution --stat
```

```bash
git show origin/solution:src/qtomo_skeleton/tomography-solution.py
```
