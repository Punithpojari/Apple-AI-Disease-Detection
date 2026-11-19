import { Activity, BookOpen, ChevronDown, CircleHelp, Info, LayoutDashboard, Menu, PanelLeftClose, PanelLeftOpen, ScanLine, Server, X } from "lucide-react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useEffect, useRef, useState } from "react";
import Brand from "./Brand.jsx";
import { getApiHealth } from "../services/api.js";

const links = [
  { to: "/dashboard", label: "Overview", icon: LayoutDashboard },
  { to: "/scanner", label: "AI Leaf Scanner", icon: ScanLine },
  { to: "/history", label: "Scan history", icon: Activity },
  { to: "/diseases", label: "Disease library", icon: BookOpen },
  { to: "/about", label: "About project", icon: Info },
  { to: "/status", label: "System status", icon: Server },
];

export default function AppShell() {
  const [open, setOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(() => window.matchMedia("(max-width: 760px)").matches);
  const [serviceStatus, setServiceStatus] = useState("checking");
  const location = useLocation();
  const menuButtonRef = useRef(null);
  const closeButtonRef = useRef(null);
  const sidebarRef = useRef(null);

  useEffect(() => {
    const mediaQuery = window.matchMedia("(max-width: 760px)");
    const updateViewport = (event) => {
      setIsMobile(event.matches);
      if (!event.matches) setOpen(false);
    };
    mediaQuery.addEventListener("change", updateViewport);
    return () => mediaQuery.removeEventListener("change", updateViewport);
  }, []);

  useEffect(() => {
    if (!isMobile || !open) return undefined;

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeButtonRef.current?.focus();

    const keepFocusInDrawer = (event) => {
      if (event.key === "Escape") {
        setOpen(false);
        return;
      }
      if (event.key !== "Tab" || !sidebarRef.current) return;

      const focusable = sidebarRef.current.querySelectorAll('a[href], button:not([disabled])');
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    };
    window.addEventListener("keydown", keepFocusInDrawer);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", keepFocusInDrawer);
      menuButtonRef.current?.focus();
    };
  }, [isMobile, open]);

  useEffect(() => {
    setOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    let active = true;
    const checkModel = () => getApiHealth()
      .then((health) => { if (active) setServiceStatus(health.modelReady ? "configured" : "setup-needed"); })
      .catch(() => { if (active) setServiceStatus("unavailable"); });
    checkModel();
    const timer = window.setInterval(checkModel, 30_000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  return (
    <div className="app-shell">
      {isMobile && open && <button className="mobile-backdrop" onClick={() => setOpen(false)} aria-label="Close navigation" />}
      <aside
        ref={sidebarRef}
        id="app-navigation"
        className={`sidebar ${open ? "sidebar-open" : ""} ${collapsed && !isMobile ? "sidebar-collapsed" : ""}`}
        role={isMobile && open ? "dialog" : undefined}
        aria-label="Main navigation"
        aria-modal={isMobile && open ? "true" : undefined}
      >
        <div className="sidebar-brand">
          <Brand />
          <button
            className="icon-button sidebar-collapse"
            type="button"
            onClick={() => setCollapsed((value) => {
              if (!value) setHelpOpen(false);
              return !value;
            })}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
          </button>
          <button ref={closeButtonRef} className="icon-button mobile-close" onClick={() => setOpen(false)} aria-label="Close menu"><X size={19} /></button>
        </div>
        <div className="workspace-select">
          <span className="workspace-avatar">P</span>
          <span className="workspace-copy"><strong>Punith workspace</strong><small>Local session</small></span>
        </div>
        <div className="nav-caption">WORKSPACE</div>
        <nav className="sidebar-nav" aria-label="Workspace">
          {links.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} title={label} aria-label={label} onClick={() => setOpen(false)} className={({ isActive }) => `nav-link ${isActive ? "nav-link-active" : ""}`}>
              <Icon size={18} strokeWidth={1.8} /><span>{label}</span>
              {label === "AI Leaf Scanner" && <span className="nav-new">AI</span>}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <button className="sidebar-help" type="button" aria-label="Workspace help" title="Workspace help" aria-expanded={helpOpen} onClick={() => setHelpOpen((value) => !value)}>
            <span className="help-icon"><CircleHelp size={17} /></span>
            <span className="help-copy"><strong>Need a hand?</strong><small>{helpOpen ? "Quick links" : "Explore your workspace"}</small></span>
            <ChevronDown className={helpOpen ? "help-chevron help-chevron-open" : "help-chevron"} size={15} />
          </button>
          {helpOpen && (
            <div className="sidebar-help-links">
              <NavLink to="/scanner" onClick={() => setOpen(false)}>How to scan a leaf</NavLink>
              <NavLink to="/history" onClick={() => setOpen(false)}>Find a previous scan</NavLink>
              <NavLink to="/diseases" onClick={() => setOpen(false)}>Browse disease guides</NavLink>
            </div>
          )}
          <div className="profile-row" title="Punith workspace"><span className="profile-avatar">P</span><span className="profile-copy"><strong>Punith workspace</strong><small>No account connected</small></span></div>
        </div>
      </aside>
      <div className="shell-main" inert={isMobile && open ? "" : undefined}>
        <header className="topbar">
          <button ref={menuButtonRef} className="icon-button mobile-menu" onClick={() => setOpen(true)} aria-label="Open menu" aria-expanded={open} aria-controls="app-navigation"><Menu size={21} /></button>
          <div className="breadcrumb"><span>AppleCare AI</span><span className="crumb-slash">/</span><strong>{links.find((link) => location.pathname.startsWith(link.to))?.label || "Analysis"}</strong></div>
          <div className="topbar-right"><span className={`status-dot service-${serviceStatus}`} /> <span>{serviceStatus === "checking" ? "Checking model" : serviceStatus === "configured" ? "Local model ready" : serviceStatus === "setup-needed" ? "Model service needed" : "API unavailable"}</span><span className="topbar-divider" /><span className="avatar-small" aria-label="Punith">P</span></div>
        </header>
        <main className="main-content"><Outlet /></main>
      </div>
    </div>
  );
}
