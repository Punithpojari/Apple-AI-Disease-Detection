# AppleCare AI

AppleCare AI uses a pretrained EfficientNetB0 model as its primary and only
scan classifier. It was compared with the supplied MobileNetV2 model using
validation macro F1 and a cleaned held-out test subset. The React app sends images to the existing Node API,
which forwards them to the local Flask model service and saves the returned
class scores in scan history. Gemini and OpenAI are not used; if the local
model service is unavailable, a scan fails instead of falling back to another
provider or generating a result.

## Requirements

- Node.js 20.19+ or 22.12+
- npm
- Python and the packages in `model-service/requirements.txt`
- Optional: MongoDB for persistent scan history

Use Python 3.11 for the model environment; the configured workspace interpreter
may be newer than the TensorFlow wheel support range.

## Install

Install the JavaScript dependencies:

```bash
npm install
npm --prefix backend install
npm --prefix frontend install
```

Create a Python virtual environment and install the model-service requirements:

```bash
python3.11 -m venv model-service/.venv
model-service/.venv/bin/python -m pip install -r model-service/requirements.txt
```

On Windows, run `py -3.11 -m venv model-service\.venv`, then use
`model-service\.venv\Scripts\python.exe` in place of
`model-service/.venv/bin/python`.

## Configure

Copy `.env.example` to `backend/.env`:

```dotenv
PORT=4000
CLIENT_ORIGIN=http://localhost:5173,http://127.0.0.1:5173
MODEL_SERVICE_URL=http://127.0.0.1:5001
MODEL_SERVICE_TIMEOUT_MS=30000
MONGODB_URI=
MAX_UPLOAD_MB=8
```

The browser only calls the Node API; model files and configuration remain on
the server. `MODEL_SERVICE_URL` should point to the Flask service. Automatic
startup prefers the interpreter in `model-service/.venv` when present; set
`MODEL_PYTHON` to override it. Set `MODEL_PORT` to change the model service
port; keep `MODEL_SERVICE_URL` in sync. Set `MODEL_SERVICE_AUTOSTART=false` to
start the model service separately.

## Run locally

With the Python environment and model requirements installed, start all three
services:

```bash
npm run dev
```

On Windows PowerShell, set the interpreter before startup:

```powershell
$env:MODEL_PYTHON = "model-service\.venv\Scripts\python.exe"
npm run dev
```

- Web app: http://localhost:5173
- Node API: http://localhost:4000
- Model service: http://127.0.0.1:5001
- Health check: http://localhost:4000/api/health

To run services separately:

```bash
model-service/.venv/bin/python model-service/app.py
npm --prefix backend run dev
npm --prefix frontend run dev
```

Without `MONGODB_URI`, history uses an in-memory store (up to 20 scans) and is
cleared when the backend restarts. With a URI, MongoDB persistence continues to
use the existing scan-history behavior. MongoDB connection failures stop the
API from starting rather than silently switching stores.

## Model output and limitations

The classifier returns scores for exactly four training labels:

- `APPLE ROT LEAVES`
- `HEALTHY LEAVES`
- `LEAF BLOTCH`
- `SCAB LEAVES`

The app preserves those labels rather than claiming that a model class is a
confirmed botanical diagnosis. The top class is marked uncertain below a
0.60 score. Scores come from the model's softmax output; they have not been
calibrated and are not diagnostic probabilities. The model was trained on leaf
images but does not check whether an uploaded image is an apple leaf or assess
image quality. It does not generate symptoms, causes, or treatment advice.

The original report's **55.6% accuracy on 63 images** was contaminated by
duplicated images with conflicting labels across splits. After excluding 12
visually duplicated/conflicting source IDs, EfficientNetB0 scored **66.7%
(40/60)** versus **58.3% (35/60)** for the original model on the cleaned test
subset. The test is small and the labels are not expert-verified, so this is a
preliminary observed difference, not a guarantee of generalization. Scores can
be wrong, including when high. Use this prototype for screening only, and
consult a local agricultural extension service or plant pathologist before
choosing treatment.

## History and dashboard

- Dashboard metrics, class distribution, health-status mix, and last-seven-day
  activity chart are calculated from the existing API history, never sample
  data. Recent activity uses the browser's local calendar days.
- The dashboard's class chart counts each scan's highest-ranked class,
  including low-score/uncertain results.
- Scan history is searchable, filterable, sortable by date or model score,
  deletable with confirmation, and exports a CSV report.
- The disease library remains educational reference material; it is separate
  from and does not expand the model's four supported output labels.
- The About Project page describes the actual inference workflow and verified
  model limitations. The System Status page reads `/api/health` and reports
  the model readiness and configured history-storage mode.

## API

- `POST /api/analyze-leaf` — multipart form with an `image` field (JPEG, PNG,
  or WebP). Returns the saved scan, including `analysis.predicted_class`,
  `analysis.class_scores`, and `analysis.confidence_score`.
- `GET /api/history` — recent analyses, including image previews; at most 200
  MongoDB records are returned.
- `GET /api/history/:id` — retrieve one analysis.
- `DELETE /api/history/:id` — delete one analysis.
- `GET /api/health` — API and local-model readiness status.
- Model service `POST /predict` — internal multipart inference endpoint.

Uploads are limited to one image and 8 MB by default. The API checks the
declared MIME type and image signature, and stores the original photo and a
small preview for history. Requests are rate-limited and response security
headers are enabled.

## Test and build

```bash
npm --prefix backend test
npm --prefix frontend test
npm run build
```

The backend tests cover model-service requests, class-score validation, and
history image handling. Frontend utility tests cover date-range filtering and
CSV escaping. A real scan requires TensorFlow and the supplied model to be
installed and loaded by the local service.

## Image credits

- “Fresh ripe apples on a tree” by AS Photography, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Fresh_ripe_apples_on_a_tree_(54015357813).jpg), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (resized to WebP for the application).
- “Apple Tree Leaf” by Konstantin Tsabolov, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Apple_Tree_Leaf_(35128940).jpeg), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) (resized to WebP for the application).

## Operational limitations

Camera capture requires browser permission and a secure context (HTTPS, except
`localhost`). Persistent history requires a configured MongoDB instance. The
unauthenticated demo is intended for local evaluation, not multi-user
production. Before deployment, add user authentication, per-user history
access controls, retention/privacy policies, production model serving, and
monitoring.

## Earlier model research

The original standalone Flask app and model experiments are preserved in
[`app/`](./app/), [`v1_baseline/`](./v1_baseline/), and
[`v2_pipeline/`](./v2_pipeline/). The repository's earlier project overview,
including the v2 experiment results and dataset findings, is preserved in
[`docs/model-research.md`](./docs/model-research.md).
