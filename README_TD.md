# OVD Watchdog – Hệ thống giám sát dựa trên OVD + Rule Engine

Hệ thống giám sát video thời gian thực sử dụng **Grounding DINO** (object detection) kết hợp **ByteTrack** (tracking) và **rule engine** deterministic để phát hiện các tình huống vi phạm an toàn theo quy tắc do người dùng định nghĩa.

Hiện hỗ trợ hai loại quy tắc chính:
- **Helmetless person riding forklift**
- **Helmetless person entering restricted zone** (ví dụ: khu vực sau nhà máy)

## Đặc điểm nổi bật

- Phát hiện thưa (sparse detection) → theo dõi dày (dense tracking)
- Hỗ trợ rule dưới dạng **JSON DSL** (khuyến nghị) hoặc YAML cũ
- Tạo rule tự động từ ngôn ngữ tự nhiên bằng LLM (qua `--rule-text`)
- Xác nhận sự kiện theo cơ chế **tentative → confirmed → resolved**
- Ghi lại đoạn video trước/sau sự kiện (ring buffer)
- Hiển thị trực quan: bounding box, track ID, ROI, trạng thái sự kiện
- Có thể lưu video đầu ra có vẽ overlay

## Yêu cầu hệ thống

- Python 3.10
- GPU NVIDIA (khuyến nghị mạnh) – nếu không có thì chạy được trên CPU nhưng rất chậm
- Các thư viện chính:
  - opencv-python
  - torch + torchvision
  - groundingdino (hoặc fork tương thích)
  - numpy, pyyaml, jsonschema, ...

## Cài đặt nhanh

```bash
# 1. Clone repo
## Download this source
cd ovd-project

# 2. Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate    # Linux/macOS
# hoặc venv\Scripts\activate  # Windows

# 3. Cài đặt dependencies
pip install -r requirements.txt
```

## Cách chạy
### 1. Chạy với rule đã có sẵn (JSON)

```bash
# Ví dụ phát hiện người không mũ bảo hộ vào vùng cấm
python main.py \
  --input data/test_videos/test_factory.mp4 \
  --rule configs/rules/helmetless_area_entrance.json \
  --detection-interval 5 \
  --display \
  --output data/output/incident_highlight.mp4
```
### 2. Tạo rule từ văn bản tự nhiên (sử dụng LLM)

```bash
python main.py --input data\test_videos\no.mp4 --rule-text "Helmetless person entering zone"

#Lưu ý: Chức năng --rule-text hiện đang gọi LLM (thường là OpenAI) để sinh file JSON rule tạm thời.
```
### 3. Các tham số dòng lệnh phổ biến
| Tham số | Mô tả | Mặc định | Ví dụ |
| :--- | :--- | :--- | :--- |
| `--input` | **Bắt buộc** – Đường dẫn tới file video hoặc địa chỉ RTSP URL. | — | `video.mp4` hoặc `rtsp://...` |
| `--rule` | Đường dẫn tới file cấu hình rule (định dạng JSON hoặc YAML). | — | `configs/rules/xxx.json` |
| `--rule-text` | Sử dụng LLM để tạo rule tự động từ mô tả tiếng Anh. | — | `"Helmetless person entering zone"` |
| `--detection-interval` | Tần suất phát hiện vật thể (tính theo số frame). Số nhỏ giúp tracking mượt hơn nhưng tốn tài nguyên. | `30` | `5` |
| `--display` | Bật cửa sổ OpenCV để theo dõi trực quan quá trình xử lý. | Tắt | `--display` |
| `--output` | Đường dẫn lưu video kết quả (bao gồm bounding box và thông tin phân tích). | Không lưu | `output.mp4` |

### Cấu trúc thư mục (Project Structure)

Dưới đây là sơ đồ tổ chức mã nguồn của dự án:

```text
.
├── configs/
│   └── rules/                  # Các file cấu hình Rule (JSON/YAML) mẫu
├── data/
│   ├── test_videos/            # Thư mục chứa các video đầu vào để thử nghiệm
│   └── output/                 # Nơi lưu trữ video kết quả + các đoạn clip sự cố (incidents)
├── src/                        # Mã nguồn chính của ứng dụng
│   ├── core/                   # Các thành phần cốt lõi của hệ thống
│   │   ├── detect/             # GroundingDINO wrapper (Xử lý nhận diện vật thể)
│   │   ├── track/              # ByteTrack (Theo dõi đối tượng qua các frame)
│   │   ├── rules/              # RuleEngine & Rule Model (Logic kiểm tra quy tắc)
│   │   ├── record/             # Ring buffer & Incident recorder (Ghi đè bộ nhớ đệm & lưu sự cố)
│   │   └── notify/             # Hệ thống thông báo (Email, Slack, Telegram...)
│   ├── models/                 # Chứa các trọng số hoặc định nghĩa model AI
│   ├── utils/                  # Các hàm tiện ích (xử lý ảnh, vẽ bounding box,...)
│   └── rule_builder/           # Module chuyển đổi từ ngôn ngữ tự nhiên (LLM) sang JSON rule
└── main.py                     # File thực thi chính (Entry point) của chương trình
```

## In Progress / Đang phát triển

### Triển khai chạy trên Jetson Orin Nano (Edge Deployment)

**Mục tiêu**:  
Chạy toàn bộ pipeline OVD Watchdog (GroundingDINO + ByteTrack + Rule Engine) trên Jetson Orin Nano để giám sát thời gian thực tại chỗ, giảm độ trễ và không phụ thuộc cloud.

**Các bước đang thực hiện / cần làm**:

1. **Remote access & Development trên Jetson**
   - Đã cài **Tailscale** trên Jetson Nano/Orin để SSH từ máy tính cá nhân.
   - Kết nối ổn định: `tailscale up` → SSH bằng `ssh user@100.x.x.x`
   - (Tùy chọn) Cài VS Code Remote-SSH để code trực tiếp trên Jetson.

2. **Chuẩn bị môi trường Jetson**
   - JetPack phiên bản: 5.1.3 hoặc 6.0 (khuyến nghị mới nhất tương thích Orin Nano)
   - Cài Docker + NVIDIA Container Toolkit:
     ```bash
     sudo apt update
     sudo apt install -y docker.io nvidia-container-toolkit
     sudo systemctl enable docker
     sudo nvidia-ctk runtime configure --runtime=docker
     sudo systemctl restart docker

---

## English Version

# OVD Watchdog – Video Surveillance System Based on OVD + Rule Engine

A real-time video monitoring system that uses **Grounding DINO** (object detection) combined with **ByteTrack** (tracking) and a deterministic rule engine to identify safety violations defined by users.

### Key features

- Sparse detection → dense tracking
- Rule definitions in **JSON DSL** (recommended) or legacy YAML
- Automatically generate rules from natural language via LLM (using `--rule-text`)
- Event confirmation pipeline: **tentative → confirmed → resolved**
- Save video segments before/after incidents (ring buffer)
- Visual overlays: bounding boxes, track IDs, ROIs, event states
- Optional output video with drawn overlays

### System requirements

- Python 3.10
- NVIDIA GPU (strongly recommended) – CPU only is possible but very slow
- Core libraries:
  - opencv-python
  - torch + torchvision
  - groundingdino (or compatible fork)
  - numpy, pyyaml, jsonschema, …

### Quick setup

```bash
# 1. Clone repo
## Download this source
cd ovd-project

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux/macOS
# or venv\\Scripts\\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### How to run

#### 1. Run with existing rule (JSON)

```bash
# Example detecting helmetless person entering restricted zone
python main.py \\
  --input data/test_videos/test_factory.mp4 \\
  --rule configs/rules/helmetless_area_entrance.json \\
  --detection-interval 5 \\
  --display \\
  --output data/output/incident_highlight.mp4
```

#### 2. Generate rule from natural language (using LLM)

```bash
python main.py --input data\\test_videos\\no.mp4 --rule-text "Helmetless person entering zone"

#Note: the --rule-text feature currently calls an LLM (usually OpenAI) to
#produce a temporary JSON rule file.
```

#### 3. Common command-line parameters
| Parameter | Description | Default | Example |
| :--- | :--- | :--- | :--- |
| `--input` | **Required** – path to video file or RTSP URL. | — | `video.mp4` or `rtsp://...` |
| `--rule` | Path to rule configuration file (JSON or YAML). | — | `configs/rules/xxx.json` |
| `--rule-text` | Use LLM to auto-generate rule from English description. | — | `"Helmetless person entering zone"` |
| `--detection-interval` | Object detection frequency (in frames). Lower values give smoother
tracking but use more resources. | `30` | `5` |
| `--display` | Enable OpenCV window for visual monitoring. | Off | `--display` |
| `--output` | Output video path (with bounding boxes and analytics overlay). | No save | `output.mp4` |

### Project Structure

Below is the directory layout of the project.

```text
.
├── configs/
│   └── rules/                  # sample rule configuration files (JSON/YAML)
├── data/
│   ├── test_videos/            # input videos for testing
│   └── output/                 # output videos and incident clips
├── src/                        # main application source code
│   ├── core/                   # core components of the system
│   │   ├── detect/             # GroundingDINO wrapper (object detection)
│   │   ├── track/              # ByteTrack (object tracking between frames)
│   │   ├── rules/              # RuleEngine & Rule Model (rule logic)
│   │   ├── record/             # Ring buffer & Incident recorder
│   │   └── notify/             # Notification system (Email, Slack, Telegram...)
│   ├── models/                 # AI models/weights definitions
│   ├── utils/                  # utility functions (image processing, drawing boxes...)
│   └── rule_builder/           # module converting natural language to JSON rules via LLM
└── main.py                     # main executable (entry point)
```

### In Progress / Under development

#### Deploying on Jetson Orin Nano (Edge Deployment)

**Goal**:  
Run the entire OVD Watchdog pipeline (GroundingDINO + ByteTrack + Rule Engine) on Jetson Orin Nano for on‑site real‑time monitoring, reducing latency and avoiding cloud dependency.

**Ongoing / pending steps**:

1. **Remote access & development on Jetson**
   - Installed **Tailscale** on Jetson Nano/Orin for SSH from personal machine.
   - Stable connection: `tailscale up` → SSH with `ssh user@100.x.x.x`
   - (Optional) Install VS Code Remote-SSH to code directly on Jetson.

2. **Preparing Jetson environment**
   - JetPack version: 5.1.3 or 6.0 (latest recommended for Orin Nano)
   - Install Docker + NVIDIA Container Toolkit:
     ```bash
     sudo apt update
     sudo apt install -y docker.io nvidia-container-toolkit
     sudo systemctl enable docker
     sudo nvidia-ctk runtime configure --runtime=docker
     sudo systemctl restart docker
     ```


---

## Japanese Version

# OVDウォッチドッグ – OVD＋ルールエンジンベースの監視システム

ユーザー定義の安全違反を検出するために、**Grounding DINO**（物体検出）と**ByteTrack**（追跡）、および決定論的ルールエンジンを組み合わせたリアルタイムビデオ監視システム。

### 主要機能

- まばらな検出 → 密な追跡
- **JSON DSL**（推奨）またはレガシーYAML形式のルール定義
- 自然言語からLLMを使ってルールを自動生成（`--rule-text`を使用）
- イベント確認のパイプライン: **tentative → confirmed → resolved**
- インシデント前後のビデオを保存（リングバッファ）
- バウンディングボックス、トラックID、ROI、イベント状態などの視覚オーバーレイ
- 描画オーバーレイ付きの出力ビデオも可能

### システム要件

- Python 3.10
- NVIDIA GPU（強く推奨）。GPUがない場合でもCPUのみで動作するが非常に遅い。
- 主要ライブラリ:
  - opencv-python
  - torch + torchvision
  - groundingdino（または互換フォーク）
  - numpy、pyyaml、jsonschema、…

### クイックセットアップ

```bash
# 1. リポジトリをクローン
## このソースをダウンロード
cd ovd-project

# 2. 仮想環境を作成（推奨）
python -m venv venv
source venv/bin/activate    # Linux/macOS
# または venv\\Scripts\\activate  # Windows

# 3. 依存関係をインストール
pip install -r requirements.txt
```

### 実行方法

#### 1. 既存のルールで実行（JSON）

```bash
# 例: 禁止区域に入る安全ヘルメット未着用者を検出
python main.py \\
  --input data/test_videos/test_factory.mp4 \\
  --rule configs/rules/helmetless_area_entrance.json \\
  --detection-interval 5 \\
  --display \\
  --output data/output/incident_highlight.mp4
```

#### 2. 自然言語からルールを生成（LLM使用）

```bash
python main.py --input data\\test_videos\\no.mp4 --rule-text "Helmetless person entering zone"

#注意: --rule-text機能は現在、LLM（通常はOpenAI）を呼び出して
#一時的なJSONルールファイルを生成します。
```

#### 3. 一般的なコマンドラインパラメータ
| パラメータ | 説明 | デフォルト | 例 |
| :--- | :--- | :--- | :--- |
| `--input` | **必須** – ビデオファイルまたはRTSP URLへのパス。 | — | `video.mp4` または `rtsp://...` |
| `--rule` | ルール設定ファイルへのパス（JSONまたはYAML）。 | — | `configs/rules/xxx.json` |
| `--rule-text` | 英語の説明からLLMを使用してルールを自動生成。 | — | `"Helmetless person entering zone"` |
| `--detection-interval` | フレーム単位の物体検出頻度。小さい値は滑らかなトラッキング
を提供しますが、リソースを多く消費します。 | `30` | `5` |
| `--display` | OpenCVウィンドウで視覚監視を有効にする。 | オフ | `--display` |
| `--output` | 出力ビデオパス（バウンディングボックスと解析オーバーレイ付き）。 | 保存しない | `output.mp4` |

### プロジェクト構造

上記のベトナム語セクションと同じディレクトリ構成。

```text
.
├── configs/
│   └── rules/                  # sample rule configuration files (JSON/YAML)
├── data/
│   ├── test_videos/            # input videos for testing
│   └── output/                 # output videos and incident clips
├── src/                        # main application source code
│   ├── core/                   # core components of the system
│   │   ├── detect/             # GroundingDINO wrapper (object detection)
│   │   ├── track/              # ByteTrack (object tracking between frames)
│   │   ├── rules/              # RuleEngine & Rule Model (rule logic)
│   │   ├── record/             # Ring buffer & Incident recorder
│   │   └── notify/             # Notification system (Email, Slack, Telegram...)
│   ├── models/                 # AI models/weights definitions
│   ├── utils/                  # utility functions (image processing, drawing boxes...)
│   └── rule_builder/           # module converting natural language to JSON rules via LLM
└── main.py                     # main executable (entry point)
```

### 進行中 / 開発中

#### Jetson Orin Nanoへのデプロイ（エッジ展開）

**目標**:  
OVD Watchdogパイプライン（GroundingDINO + ByteTrack + Rule Engine）をJetson Orin Nanoで実行し、オンサイトでのリアルタイム監視を行い、レイテンシを減らしクラウド依存を避ける。

**進行中 / 実施予定のステップ**:

1. **Jetsonでのリモートアクセスと開発**
   - Jetson Nano/Orinに **Tailscale** をインストールし、個人PCからSSH可能にした。
   - 安定接続： `tailscale up` → `ssh user@100.x.x.x`
   - （任意）Jetson上で直接コード編集するためVS Code Remote-SSHをインストール。

2. **Jetson環境の準備**
   - JetPackバージョン：5.1.3または6.0（Orin Nano向け最新推奨）
   - Docker + NVIDIA Container Toolkitをインストール：
     ```bash
     sudo apt update
     sudo apt install -y docker.io nvidia-container-toolkit
     sudo systemctl enable docker
     sudo nvidia-ctk runtime configure --runtime=docker
     sudo systemctl restart docker
     ```
