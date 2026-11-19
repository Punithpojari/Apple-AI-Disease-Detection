import { useCallback, useEffect, useState } from "react";
import { AlertCircle, CheckCircle2, CircleHelp, LoaderCircle, RefreshCw, Server } from "lucide-react";
import PageHeader from "../components/PageHeader.jsx";
import { getApiHealth } from "../services/api.js";

export default function SystemStatus() {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const currentHealth = await getApiHealth();
      setHealth(currentHealth);
      setError("");
    } catch (requestError) {
      setHealth(null);
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 30_000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const apiOnline = health?.status === "ok";
  const modelReady = apiOnline && health.modelReady === true;
  const persistence = health?.historyPersistence;

  return (
    <div>
      <PageHeader
        eyebrow="LIVE SERVICE CHECK"
        title="System status"
        description="Current readiness reported by the running app and its health endpoint."
        action={<button className="button button-outline" onClick={refresh} disabled={loading}><RefreshCw size={15} className={loading ? "spin" : ""} /> Retry check</button>}
      />
      {error && <div className="notice notice-error" role="alert"><AlertCircle size={16} />{error}</div>}
      <section className="status-grid" aria-live="polite">
        <ServiceCard title="Web application" detail="React interface" state="ready" />
        <ServiceCard title="Express API" detail="GET /api/health" state={loading && !health ? "checking" : apiOnline ? "ready" : "unavailable"} />
        <ServiceCard title="Flask model service" detail={health?.model || "EfficientNetB0"} state={loading && !health ? "checking" : modelReady ? "ready" : apiOnline ? "unavailable" : "unknown"} />
        <ServiceCard title="Scan history" detail={persistence === "mongodb" ? "MongoDB persistence" : persistence === "memory" ? "In-memory storage" : "Storage status unavailable"} state={persistence === "mongodb" ? "ready" : persistence === "memory" ? "limited" : loading ? "checking" : "unknown"} />
      </section>
      <section className="panel status-details">
        <div className="panel-head"><div><div className="eyebrow">WHAT THIS MEANS</div><h2>Health checks, not predictions</h2></div><Server size={19} /></div>
        <p>This page reads the Express API health endpoint. The API reports whether the configured local model service answers with the expected model name and four class labels; it does not run a new leaf prediction.</p>
        {persistence === "memory" && <div className="status-storage-note"><CircleHelp size={16} /><span>History is currently in memory and will be cleared if the backend restarts. Configure MongoDB to persist scans across restarts.</span></div>}
        {persistence === "mongodb" && <div className="status-storage-note status-storage-ready"><CheckCircle2 size={16} /><span>MongoDB history persistence is configured for this backend.</span></div>}
        {!apiOnline && !loading && <div className="status-storage-note"><AlertCircle size={16} /><span>The API health check failed. Model and storage availability cannot be confirmed until the API responds.</span></div>}
        <small>{health?.provider && `Inference provider: ${health.provider}. `}{health?.model && `Configured model: ${health.model}.`}</small>
      </section>
    </div>
  );
}

function ServiceCard({ title, detail, state }) {
  const stateText = {
    ready: "Connected",
    checking: "Checking",
    unavailable: "Unavailable",
    limited: "Limited persistence",
    unknown: "Unknown",
  }[state];
  const StateIcon = state === "ready" ? CheckCircle2 : state === "checking" ? LoaderCircle : state === "limited" || state === "unknown" ? CircleHelp : AlertCircle;

  return (
    <article className={`panel status-card status-card-${state}`}>
      <div className="status-card-top"><span className="status-service-icon"><Server size={18} /></span><span className="status-state"><StateIcon size={14} className={state === "checking" ? "spin" : ""} />{stateText}</span></div>
      <h2>{title}</h2>
      <p>{detail}</p>
    </article>
  );
}
