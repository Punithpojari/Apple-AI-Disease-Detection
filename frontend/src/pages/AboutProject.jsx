import { ArrowRight, BookOpen, Cpu, Database, Leaf, ScanLine, ShieldCheck } from "lucide-react";
import { Link } from "react-router-dom";
import PageHeader from "../components/PageHeader.jsx";
import { MODEL_CLASS_NAMES } from "../utils/modelClasses.js";

const stack = [
  { name: "React + Vite", detail: "Responsive browser interface" },
  { name: "Node.js + Express", detail: "Image validation, API, and scan history" },
  { name: "Flask + TensorFlow", detail: "Serves the existing trained model locally" },
  { name: "MongoDB (optional)", detail: "Persistent history when configured" },
];

export default function AboutProject() {
  return (
    <div>
      <PageHeader
        eyebrow="ABOUT THE PROJECT"
        title="A thoughtful first look at leaf health."
        description="AppleCare AI is a screening prototype that connects a leaf photo to a local, pretrained EfficientNetB0 classifier."
        action={<Link className="button button-dark" to="/scanner"><ScanLine size={16} /> Try the scanner <ArrowRight size={15} /></Link>}
      />

      <section className="about-intro panel">
        <span className="about-intro-icon"><Leaf size={22} /></span>
        <div>
          <div className="eyebrow">PROJECT OBJECTIVE</div>
          <h2>Make an initial observation easier to review.</h2>
          <p>The app accepts an uploaded or camera-captured image, sends it through the Express API to the local Flask model service, and saves the real class scores in scan history. It does not train a model or call a hosted AI provider.</p>
        </div>
      </section>

      <section className="about-section">
        <div className="about-section-heading"><div className="eyebrow">HOW A SCAN WORKS</div><h2>Photo in. Scores out.</h2></div>
        <div className="about-workflow">
          <WorkflowStep number="01" title="Capture or upload" text="Choose a JPEG, PNG, or WebP photo, or capture one with a permitted device camera." icon={ScanLine} />
          <WorkflowStep number="02" title="Validate and infer" text="The backend checks the image and forwards its bytes to the EfficientNetB0 model." icon={Cpu} />
          <WorkflowStep number="03" title="Review and save" text="The app validates all four returned class scores, shows the result with limitations, and records the scan." icon={Database} />
        </div>
      </section>

      <section className="about-columns">
        <div className="panel about-card">
          <div className="about-section-heading"><div className="eyebrow">TECHNOLOGY</div><h2>Built with a small, clear stack.</h2></div>
          <div className="tech-stack">{stack.map((item) => <div className="tech-row" key={item.name}><strong>{item.name}</strong><span>{item.detail}</span></div>)}</div>
        </div>
        <div className="panel about-card">
          <div className="about-section-heading"><div className="eyebrow">MODEL OUTPUTS</div><h2>Four training labels</h2></div>
          <ul className="model-class-list">{MODEL_CLASS_NAMES.map((name) => <li key={name}>{name}</li>)}</ul>
          <p className="about-note">The disease library is educational context; it does not add categories to the classifier.</p>
        </div>
      </section>

      <section className="about-limitations">
        <span><ShieldCheck size={19} /></span>
        <div>
          <strong>Prototype screening only—not an agricultural diagnosis.</strong>
          <p>The selected EfficientNetB0 scored 66.7% on a cleaned 60-image test subset, compared with 58.3% for the original model. This small, non-expert-verified evaluation is preliminary. Scores are uncalibrated; the model cannot verify apple-leaf identity, inspect image quality, explain symptoms, or recommend treatment. Confirm concerns with a local agricultural expert.</p>
        </div>
        <Link to="/status">View live system status <ArrowRight size={14} /></Link>
      </section>

      <section className="about-library-link">
        <BookOpen size={17} />
        <span>Looking for general information about apple leaf conditions?</span>
        <Link to="/diseases">Browse the educational library <ArrowRight size={14} /></Link>
      </section>
    </div>
  );
}

function WorkflowStep({ number, title, text, icon: Icon }) {
  return (
    <article className="panel about-step">
      <div className="about-step-top"><span>{number}</span><Icon size={18} /></div>
      <h3>{title}</h3>
      <p>{text}</p>
    </article>
  );
}
