import { Activity, ArrowRight, ArrowUpRight, ImagePlus, Leaf, ScanLine, Sprout } from "lucide-react";
import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Link } from "react-router-dom";
import PageHeader from "../components/PageHeader.jsx";
import StatusBadge from "../components/StatusBadge.jsx";
import { useHistory } from "../hooks/useHistory.js";
import { buildRecentActivity, filterScansByRange } from "../utils/dashboardMetrics.js";

const chartColors = ["#35845a", "#db9755", "#a9c5ad", "#7d8f78", "#c5d5c8", "#67786e"];

function formatDate(value) {
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", year: "numeric" }).format(new Date(value));
}

function formatRecentDate(value) {
  const date = new Date(value);
  return Number.isFinite(date.getTime())
    ? new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(date)
    : "Date unavailable";
}

export default function Dashboard() {
  const { scans, loading, error } = useHistory();
  const [range, setRange] = useState("all");
  const filteredScans = useMemo(() => filterScansByRange(scans, range), [range, scans]);
  const healthy = filteredScans.filter((scan) => scan.analysis.health_status === "healthy").length;
  const potential = filteredScans.filter((scan) => scan.analysis.health_status === "potential_disease").length;
  const otherStatuses = filteredScans.length - healthy - potential;
  const classCounts = filteredScans.reduce((counts, scan) => {
    const classTitle = scan.analysis.predicted_class?.title || scan.analysis.predicted_disease;
    if (classTitle && classTitle !== "Uncertain") counts[classTitle] = (counts[classTitle] || 0) + 1;
    return counts;
  }, {});
  const chartData = Object.entries(classCounts).map(([name, value]) => ({ name, value }));
  const healthData = [
    { name: "Healthy", value: healthy, color: "#35845a" },
    { name: "Potential disease", value: potential, color: "#db9755" },
    { name: "Uncertain / other", value: otherStatuses, color: "#a9b5aa" },
  ].filter((item) => item.value > 0);
  const activityData = useMemo(() => buildRecentActivity(scans), [scans]);
  const activityMax = Math.max(1, ...activityData.map((item) => item.count));
  const latestScan = scans[0];
  const rangeLabel = range === "all" ? "All time" : `Last ${range} days`;

  return (
    <div>
      <PageHeader eyebrow="ORCHARD OVERVIEW" title="Welcome back, Punith" description="Here’s a look at your recent apple leaf assessments." action={<Link className="button button-dark" to="/scanner"><ScanLine size={16} /> New analysis</Link>} />
      {error && <div className="notice notice-error">{error}</div>}
      <section className="stats-grid">
        <StatCard label="Total analyses" value={loading ? "—" : filteredScans.length} note={range === "all" ? "Saved in scan history" : rangeLabel} icon={ScanLine} tone="green" />
        <StatCard label="Healthy assessments" value={loading ? "—" : healthy} note={loading ? "Loading history" : filteredScans.length ? `${Math.round((healthy / filteredScans.length) * 100)}% of this period` : "No assessments yet"} icon={Sprout} tone="soft" />
        <StatCard label="Potential disease" value={loading ? "—" : potential} note="Worth another look" icon={Leaf} tone="amber" />
        <div className="stat-card stat-recent-activity"><div className="stat-card-top"><span>Recent activity</span><span className="stat-icon"><Activity size={18} /></span></div><strong className="stat-value">{loading ? "—" : latestScan ? formatRecentDate(latestScan.createdAt) : "—"}</strong><span className="stat-note">{loading ? "Loading latest scan" : latestScan ? latestScan.analysis.predicted_class?.title || latestScan.analysis.predicted_disease || "Assessment uncertain" : "No scans yet"}</span></div>
      </section>
      <section className="dashboard-panels">
        <div className="panel recent-panel"><div className="panel-head"><div><div className="eyebrow">YOUR LATEST CHECK-INS</div><h2>Recent analyses</h2></div><Link to="/history" className="panel-link">View history <ArrowRight size={15} /></Link></div>
          {loading ? <div className="empty-state compact">Loading your scan history…</div> : filteredScans.length ? <div className="recent-list">{filteredScans.slice(0, 5).map((scan) => <Link className="recent-row" to={`/analysis/${scan.id}`} key={scan.id}><img src={scan.imageData} alt="" /><span className="recent-details"><strong>{scan.analysis.predicted_class?.title || scan.analysis.predicted_disease || "Assessment uncertain"}</strong><small>{formatDate(scan.createdAt)} · {scan.model}</small></span><StatusBadge status={scan.analysis.health_status} /><ArrowUpRight className="recent-arrow" size={16} /></Link>)}</div> : scans.length ? <div className="empty-state compact">No analyses in this time range. Try a wider range.</div> : error ? <div className="empty-state compact">History could not be loaded. Retry from scan history.</div> : <div className="empty-state"><span className="empty-icon"><ImagePlus size={22} /></span><strong>Your first scan starts here</strong><p>Upload or take a leaf photo to rank the model’s four classes.</p><Link className="button button-dark button-small" to="/scanner">Scan a leaf <ArrowRight size={14} /></Link></div>}
        </div>
        <div className="panel chart-panel"><div className="panel-head"><div><div className="eyebrow">A QUICK PICTURE</div><h2>Top model classes</h2></div><div className="range-switch" role="group" aria-label="Dashboard date range">{[["7", "7D"], ["30", "30D"], ["all", "All"]].map(([value, label]) => <button key={value} className={range === value ? "range-active" : ""} aria-pressed={range === value} onClick={() => setRange(value)}>{label}</button>)}</div></div>
          {loading ? <div className="chart-empty">Loading scan distribution…</div> : chartData.length ? <><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={chartData} dataKey="value" nameKey="name" innerRadius={66} outerRadius={92} paddingAngle={4} stroke="none">{chartData.map((entry, index) => <Cell key={entry.name} fill={chartColors[index % chartColors.length]} />)}</Pie><Tooltip /></PieChart></ResponsiveContainer><div className="chart-center"><strong>{filteredScans.length}</strong><span>ranked<br />images</span></div></div><div className="chart-legend">{chartData.slice(0, 4).map((entry, index) => <div key={entry.name}><i style={{ background: chartColors[index % chartColors.length] }} /><span>{entry.name}</span><strong>{entry.value}</strong></div>)}</div><p className="chart-note">Counts each scan’s highest-ranked class, including uncertain results.</p></> : <div className="chart-empty"><div className="chart-empty-illustration"><Leaf size={23} /></div><strong>Your insights will grow here.</strong><p>Complete a few scans to see the model’s top-class counts across your history.</p></div>}
        </div>
      </section>
      <section className="dashboard-secondary-panels">
        <div className="panel dashboard-chart-card">
          <div className="panel-head"><div><div className="eyebrow">ASSESSMENT MIX</div><h2>Healthy vs. potential disease</h2></div></div>
          {loading ? <div className="chart-empty">Loading assessment mix…</div> : healthData.length ? <><div className="chart-wrap health-chart-wrap"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={healthData} dataKey="value" nameKey="name" innerRadius={53} outerRadius={76} paddingAngle={4} stroke="none">{healthData.map((entry) => <Cell key={entry.name} fill={entry.color} />)}</Pie><Tooltip /></PieChart></ResponsiveContainer><div className="chart-center"><strong>{filteredScans.length}</strong><span>saved<br />scans</span></div></div><div className="chart-legend">{healthData.map((entry) => <div key={entry.name}><i style={{ background: entry.color }} /><span>{entry.name}</span><strong>{entry.value}</strong></div>)}</div></> : <div className="chart-empty"><strong>No assessments yet.</strong><p>Run a scan to see health-status counts from your actual history.</p></div>}
          <p className="chart-note">{rangeLabel}; uncertain and other statuses are shown separately.</p>
        </div>
        <div className="panel dashboard-chart-card">
          <div className="panel-head"><div><div className="eyebrow">LAST 7 DAYS</div><h2>Recent scan activity</h2></div></div>
          {loading ? <div className="chart-empty">Loading recent activity…</div> : scans.length ? <div className="activity-chart-wrap"><ResponsiveContainer width="100%" height="100%"><BarChart data={activityData} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}><CartesianGrid vertical={false} stroke="#edf1ed" /><XAxis dataKey="label" axisLine={false} tickLine={false} tick={{ fontSize: 9, fill: "#829087" }} /><YAxis allowDecimals={false} domain={[0, activityMax]} axisLine={false} tickLine={false} tick={{ fontSize: 9, fill: "#829087" }} /><Tooltip /><Bar dataKey="count" name="Scans" fill="#4c8b5f" radius={[4, 4, 0, 0]} maxBarSize={28} /></BarChart></ResponsiveContainer></div> : <div className="chart-empty"><strong>No scan activity yet.</strong><p>Your saved scans will appear here by day.</p></div>}
          <p className="chart-note">Counts saved scans by local calendar day.</p>
        </div>
      </section>
      <section className="dashboard-banner"><div className="banner-leaf"><Leaf size={24} /></div><div><strong>A helpful first look, never the final word.</strong><p>The classifier returns one of four class scores and does not explain visible evidence. Confirm results with a local agricultural expert.</p></div><Link to="/diseases">Browse the library <ArrowRight size={15} /></Link></section>
    </div>
  );
}

function StatCard({ label, value, note, icon: Icon, tone }) {
  return <div className={`stat-card stat-${tone}`}><div className="stat-card-top"><span>{label}</span><span className="stat-icon"><Icon size={18} /></span></div><strong className="stat-value">{value}</strong><span className="stat-note">{note}</span></div>;
}
