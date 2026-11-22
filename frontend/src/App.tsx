import { useState, useEffect } from 'react';
import { 
  BarChart2, Search, Shuffle, ShieldCheck, Zap, Award, 
  Brain, FileText, CheckCircle2, XCircle, 
  Layers, TrendingUp, Sparkles, Server, Filter, Scale
} from 'lucide-react';
import ModelComparison from './ModelComparison';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell 
} from 'recharts';

interface ModelAggregates {
  Model: string;
  [key: string]: any;
}

interface SampleComparison {
  id: string;
  input_text: string;
  context: string;
  target: string;
  metadata: any;
  prompt?: string;
  models: {
    [modelKey: string]: {
      output_text: string;
      exact_match: number;
      f1: number;
      bleu: number;
      rouge: number;
      judge_score: number;
      completeness: number;
      helpfulness: number;
      bert_score: number;
      reasoning_faithfulness: number;
      latency_seconds: number;
      tokens_out: number;
      is_hallucinated: boolean;
      is_failure: boolean;
      has_fn_call_leak?: boolean;
      is_answer_recoverable?: number;
      ungrounded_number_ratio?: number;
    }
  }
}

const METRIC_DEFINITIONS: { [key: string]: { title: string; category: string; description: string; unit: string; isLowerBetter?: boolean; format?: (v: number) => string; color: string } } = {
  // CORE 15 METRICS
  "Exact Match": { title: "Exact Match (EM)", category: "Accuracy & Match", description: "Strict string/numeric equality with ground truth target", unit: "%", format: (v) => `${(v * 100).toFixed(1)}%`, color: "#38bdf8" },
  "F1": { title: "Token F1 Score", category: "Accuracy & Match", description: "Harmonic mean of precision & recall over extracted tokens", unit: "%", format: (v) => `${(v * 100).toFixed(1)}%`, color: "#ffa116" },
  "BLEU": { title: "BLEU Score", category: "Accuracy & Match", description: "N-gram overlap precision score vs reference text", unit: "Score", format: (v) => v.toFixed(3), color: "#a855f7" },
  "ROUGE": { title: "ROUGE-L Score", category: "Accuracy & Match", description: "Longest common subsequence recall score", unit: "%", format: (v) => `${(v * 100).toFixed(1)}%`, color: "#2cbb5d" },
  "Judge Score": { title: "LLM Judge Score", category: "LLM Judge & Quality", description: "NeMo Evaluator LLM-as-a-Judge anchored score (1.0 to 5.0)", unit: "/ 5.0", format: (v) => `${v.toFixed(2)} / 5.0`, color: "#ffa116" },
  "Completeness": { title: "Completeness Score", category: "LLM Judge & Quality", description: "Evaluates whether all financial query aspects were addressed", unit: "/ 5.0", format: (v) => `${v.toFixed(2)} / 5.0`, color: "#2cbb5d" },
  "Helpfulness": { title: "Helpfulness Score", category: "LLM Judge & Quality", description: "Clear, structured, and actionable numerical output quality", unit: "/ 5.0", format: (v) => `${v.toFixed(2)} / 5.0`, color: "#38bdf8" },
  "Reasoning Faithfulness": { title: "Reasoning Faithfulness", category: "Faithfulness & Safety", description: "Fraction of ground-truth calculation steps present in chain-of-thought", unit: "%", format: (v) => `${(v * 100).toFixed(1)}%`, color: "#38bdf8" },
  "BERTScore": { title: "BERTScore", category: "Faithfulness & Safety", description: "Semantic embedding similarity of reasoning vs gold answer", unit: "Score", format: (v) => v.toFixed(4), color: "#a855f7" },
  "Hallucination Rate": { title: "Hallucination Rate", category: "Faithfulness & Safety", description: "Unverified numbers outside context whitelist (Lower is better!)", unit: "%", isLowerBetter: true, format: (v) => `${(v * 100).toFixed(1)}%`, color: "#ef4743" },
  "Failure Rate": { title: "Failure Rate", category: "Faithfulness & Safety", description: "Rate of invalid API calls, empty outputs, or refusals (Lower is better!)", unit: "%", isLowerBetter: true, format: (v) => `${(v * 100).toFixed(1)}%`, color: "#ef4743" },
  "Latency (s)": { title: "Generation Latency", category: "Performance & Speed", description: "Average generation time per request in seconds (Lower is better!)", unit: "s", isLowerBetter: true, format: (v) => `${v.toFixed(2)}s`, color: "#ffc01e" },
  "Throughput (req/s)": { title: "Request Throughput", category: "Performance & Speed", description: "Total completed inference requests per second", unit: "req/s", format: (v) => `${v.toFixed(3)}`, color: "#2cbb5d" },
  "Tokens/sec": { title: "Output Token Speed", category: "Performance & Speed", description: "Average output generation speed in tokens per second", unit: "tok/s", format: (v) => `${v.toFixed(1)} tok/s`, color: "#38bdf8" },
  "Category Accuracy": { title: "Overall Category Accuracy", category: "Accuracy & Match", description: "Tidy accuracy aggregated across task types and company disclosures", unit: "%", format: (v) => `${(v * 100).toFixed(1)}%`, color: "#a855f7" },

  // DERIVED 27 METRICS
  // Group 1: Format & Leakage
  "FN Call Leak Rate": { title: "FN Call Leak Rate", category: "Format & Leakage", description: "Share of model outputs leaking raw function-call JSON syntax (Lower is better!)", unit: "%", isLowerBetter: true, format: (v) => `${(v * 100).toFixed(1)}%`, color: "#f43f5e" },
  "Extraction Success Rate": { title: "Extraction Success Rate", category: "Format & Leakage", description: "Share of predictions with non-empty extracted final answer", unit: "%", format: (v) => `${(v * 100).toFixed(1)}%`, color: "#10b981" },
  "Final Answer Line Rate": { title: "Final Answer Line Rate", category: "Format & Leakage", description: "Share of outputs containing an explicit 'Final Answer:' header", unit: "%", format: (v) => `${(v * 100).toFixed(1)}%`, color: "#06b6d4" },
  "Answer Recoverable Rate": { title: "Answer Recoverable Rate", category: "Format & Leakage", description: "Share where ground-truth answer was stated in pre-JSON reasoning text", unit: "%", format: (v) => `${(v * 100).toFixed(1)}%`, color: "#8b5cf6" },

  // Group 2: Task Granularity
  "Label Exact Match": { title: "Label Exact Match", category: "Task Granularity", description: "Mean exact match accuracy on label/classification tasks (qa_label)", unit: "%", format: (v) => `${(v * 100).toFixed(1)}%`, color: "#3b82f6" },
  "Numeric Exact Match": { title: "Numeric Exact Match", category: "Task Granularity", description: "Mean exact match accuracy on numeric arithmetic tasks (qa_numeric_program)", unit: "%", format: (v) => `${(v * 100).toFixed(1)}%`, color: "#6366f1" },

  // Group 3: Judge Alignment
  "Judge EM Gap": { title: "Judge EM Gap", category: "Judge Alignment", description: "Normalized delta between LLM judge score and exact match (mean(judge)/5 - mean(EM))", unit: "Gap", isLowerBetter: true, format: (v) => v.toFixed(3), color: "#ec4899" },
  "Judge Composite": { title: "Judge Composite", category: "Judge Alignment", description: "Harmonic composite of mean judge score, completeness, and helpfulness", unit: "/ 5.0", format: (v) => `${v.toFixed(2)} / 5.0`, color: "#f59e0b" },
  "Judge Leniency Rate": { title: "Judge Leniency Rate", category: "Judge Alignment", description: "Share of samples awarded judge >= 4 despite 0 exact match (Lower is better!)", unit: "%", isLowerBetter: true, format: (v) => `${(v * 100).toFixed(1)}%`, color: "#f97316" },
  "Judge Fail Rate": { title: "Judge Fail Rate", category: "Judge Alignment", description: "Share of samples receiving a failing judge score <= 2 (Lower is better!)", unit: "%", isLowerBetter: true, format: (v) => `${(v * 100).toFixed(1)}%`, color: "#ef4444" },

  // Group 4: Reliability & Grounding
  "Reliable Accuracy": { title: "Reliable Accuracy", category: "Reliability & Grounding", description: "Share of predictions both Exact Match correct and completely non-hallucinated", unit: "%", format: (v) => `${(v * 100).toFixed(1)}%`, color: "#14b8a6" },
  "Hallucinated But Correct": { title: "Hallucinated But Correct", category: "Reliability & Grounding", description: "Share of correct answers that contained ungrounded reasoning (Lower is better!)", unit: "%", isLowerBetter: true, format: (v) => `${(v * 100).toFixed(1)}%`, color: "#f43f5e" },
  "Ungrounded Number Ratio": { title: "Ungrounded Number Ratio", category: "Reliability & Grounding", description: "Average ratio of distinct reasoning numbers not traceable to context/prompt", unit: "%", isLowerBetter: true, format: (v) => `${(v * 100).toFixed(1)}%`, color: "#e11d48" },
  "Ungrounded Flag Delta": { title: "Ungrounded Flag Delta", category: "Reliability & Grounding", description: "Ungrounded ratio difference between hallucinated and non-hallucinated rows", unit: "Delta", format: (v) => v.toFixed(4), color: "#a855f7" },
  "Hallucination Separation": { title: "Hallucination Separation", category: "Reliability & Grounding", description: "Separation power of hallucination flag: P(H | EM=0) - P(H | EM=1)", unit: "Delta", format: (v) => v.toFixed(4), color: "#8b5cf6" },
  "Label Hedge Rate": { title: "Label Hedge Rate", category: "Reliability & Grounding", description: "Share of label question responses expressing uncertainty or hedging phrases", unit: "%", format: (v) => `${(v * 100).toFixed(1)}%`, color: "#eab308" },

  // Group 5: Efficiency & Scaling
  "Latency per Correct": { title: "Latency per Correct", category: "Efficiency & Scaling", description: "Average generation latency in seconds required per unit of EM (Lower is better!)", unit: "s/EM", isLowerBetter: true, format: (v) => `${v.toFixed(2)}s`, color: "#f59e0b" },
  "Tokens per Correct": { title: "Tokens per Correct", category: "Efficiency & Scaling", description: "Average output tokens consumed per unit of exact match accuracy", unit: "tok/EM", isLowerBetter: true, format: (v) => `${v.toFixed(1)}`, color: "#eab308" },
  "Latency P95 (s)": { title: "Latency P95 (s)", category: "Efficiency & Scaling", description: "95th percentile generation latency across samples (Lower is better!)", unit: "s", isLowerBetter: true, format: (v) => `${v.toFixed(2)}s`, color: "#f97316" },
  "Latency CV": { title: "Latency CV", category: "Efficiency & Scaling", description: "Coefficient of variation of generation latency (std / mean)", unit: "CV", isLowerBetter: true, format: (v) => v.toFixed(3), color: "#f59e0b" },
  "Quality per Second": { title: "Quality per Second", category: "Efficiency & Scaling", description: "Mean judge score delivered per second of generation latency", unit: "pts/s", format: (v) => `${v.toFixed(3)}/s`, color: "#10b981" },
  "Output/Input Token Ratio": { title: "Output/Input Token Ratio", category: "Efficiency & Scaling", description: "Average ratio of generated output tokens to prompt input tokens", unit: "Ratio", format: (v) => v.toFixed(3), color: "#06b6d4" },
  "Verbosity-Accuracy Corr": { title: "Verbosity-Accuracy Corr", category: "Efficiency & Scaling", description: "Pearson correlation coefficient between output token count and exact match", unit: "r", format: (v) => v.toFixed(3), color: "#6366f1" },

  // Group 6: Robustness & CI
  "Macro Exact Match": { title: "Macro Exact Match", category: "Robustness & CI", description: "Macro-averaged exact match accuracy across all category subgroups", unit: "%", format: (v) => `${(v * 100).toFixed(1)}%`, color: "#3b82f6" },
  "Worst Category EM": { title: "Worst Category EM", category: "Robustness & CI", description: "Minimum exact match accuracy among categories with at least 5 samples", unit: "%", format: (v) => `${(v * 100).toFixed(1)}%`, color: "#ec4899" },
  "Category EM Std": { title: "Category EM Std", category: "Robustness & CI", description: "Standard deviation of exact match across categories (n >= 5, Lower is better!)", unit: "Std", isLowerBetter: true, format: (v) => v.toFixed(4), color: "#a855f7" },
  "EM 95% CI Halfwidth": { title: "EM 95% CI Halfwidth", category: "Robustness & CI", description: "Bootstrap 95% confidence interval half-width of mean EM (2000 resamples, seed 0)", unit: "±%", isLowerBetter: true, format: (v) => `±${(v * 100).toFixed(2)}%`, color: "#8b5cf6" }
};

const MODEL_COLOR_MAP: { [key: string]: string } = {
  // Evaluated Benchmark Models
  "Qwen3-8B-GGUF": "#ffa116",                     // Amber / Orange
  "Ministral-3-8B-Reasoning-2512-GGUF": "#2cbb5d", // Emerald Green
  "OREAL-DeepSeek-R1-Distill-Qwen-7B-GGUF": "#a855f7", // Violet / Purple
  "Meta-Llama-3.1-8B-Instruct-GGUF": "#38bdf8",   // Electric Cyan / Blue
  "meta-llama-3.1-8b-instruct": "#38bdf8",        // Electric Cyan / Blue
  "ministral-3-8b-instruct-2512": "#2cbb5d",      // Emerald Green
  // Legacy / Fallback Keys
  "llama3.1-8b": "#38bdf8",
  "deepseek-r1-7b": "#a855f7",
  "phi4-mini": "#10b981",
  "qwen2.5-7b": "#ffa116",
  "mistral-7b-v0.3": "#ef4743",
  "qwen": "#ffa116",
  "mistralai": "#ef4743"
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'metrics' | 'compare' | 'inspector' | 'company' | 'table'>('compare');
  const [masterData, setMasterData] = useState<ModelAggregates[]>([]);
  const [categoryData, setCategoryData] = useState<any[]>([]);
  const [modelsList, setModelsList] = useState<string[]>([]);
  const [metricFilter, setMetricFilter] = useState<string>('All');
  
  // Inspector state
  const [selectedModel, setSelectedModel] = useState<string>('llama3.1-8b');
  const [selectedSampleId, setSelectedSampleId] = useState<string>('item_001');
  const [sampleIdList, setSampleIdList] = useState<string[]>([]);
  const [sampleComparison, setSampleComparison] = useState<SampleComparison | null>(null);
  const [compareAllMode, setCompareAllMode] = useState<boolean>(true);
  const [loadingSample, setLoadingSample] = useState<boolean>(false);

  // Fetch initial master table and available models
  useEffect(() => {
    fetch('/api/master-table')
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data)) setMasterData(data);
      })
      .catch(err => console.error("Error loading master table:", err));

    fetch('/api/models')
      .then(res => res.json())
      .then(data => {
        if (data.models && Array.isArray(data.models)) {
          setModelsList(data.models);
          if (data.models.length > 0) setSelectedModel(data.models[0]);
        }
      })
      .catch(err => console.error("Error loading models:", err));

    fetch('/api/category-accuracy')
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data)) setCategoryData(data);
      })
      .catch(err => console.error("Error loading category accuracy:", err));

    fetch('/api/random-id')
      .then(res => res.json())
      .then(data => {
        if (data.all_ids) setSampleIdList(data.all_ids);
        if (data.id) setSelectedSampleId(data.id);
      })
      .catch(err => console.error("Error loading random id:", err));
  }, []);

  // Fetch sample comparison data whenever sample ID changes
  useEffect(() => {
    if (!selectedSampleId) return;
    setLoadingSample(true);
    fetch(`/api/sample/${selectedSampleId}`)
      .then(res => res.json())
      .then(data => {
        setSampleComparison(data);
        setLoadingSample(false);
      })
      .catch(err => {
        console.error("Error loading sample:", err);
        setLoadingSample(false);
      });
  }, [selectedSampleId]);

  const handlePickRandomSample = () => {
    fetch('/api/random-id')
      .then(res => res.json())
      .then(data => {
        if (data.id) {
          setSelectedSampleId(data.id);
        }
      })
      .catch(err => console.error(err));
  };

  const [tableFilter, setTableFilter] = useState<'all' | 'core' | 'derived'>('all');

  const CORE_CATEGORIES = ['Accuracy & Match', 'LLM Judge & Quality', 'Faithfulness & Safety', 'Performance & Speed'];
  const DERIVED_CATEGORIES = ['Format & Leakage', 'Task Granularity', 'Judge Alignment', 'Reliability & Grounding', 'Efficiency & Scaling', 'Robustness & CI'];

  const filteredMetrics = Object.keys(METRIC_DEFINITIONS).filter(metricKey => {
    if (metricFilter === 'All (42)' || metricFilter === 'All') return true;
    if (metricFilter === 'Core 15') return CORE_CATEGORIES.includes(METRIC_DEFINITIONS[metricKey].category);
    if (metricFilter === 'Derived 27') return DERIVED_CATEGORIES.includes(METRIC_DEFINITIONS[metricKey].category);
    return METRIC_DEFINITIONS[metricKey].category === metricFilter;
  });

  const getTopModelForMetric = (metricKey: string) => {
    if (!masterData || masterData.length === 0) return null;
    const isLower = METRIC_DEFINITIONS[metricKey]?.isLowerBetter;
    let best = masterData[0];
    for (const m of masterData) {
      const val = (m as any)[metricKey] ?? 0;
      const bestVal = (best as any)[metricKey] ?? 0;
      if (isLower) {
        if (val < bestVal) best = m;
      } else {
        if (val > bestVal) best = m;
      }
    }
    return best;
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg-dark)' }}>
      {/* Header Bar - LeetCode Style */}
      <header className="glass-panel" style={{ margin: '16px 24px 0 24px', padding: '12px 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{ background: '#2e2e2e', width: '40px', height: '40px', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid var(--border-light)' }}>
            <Brain size={24} color="#ffa116" />
          </div>
          <div>
            <h1 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.01em' }}>
              LLM Benchmarking & Evaluation Inspector
            </h1>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '8px', marginTop: '3px' }}>
              <span className="badge badge-neutral"><Server size={12} /> NeMo Evaluator Suite</span>
              <span className="badge badge-neutral"><Layers size={12} /> 42 Total Metrics (15 Core + 27 Derived)</span>
              <span className="badge badge-orange"><CheckCircle2 size={12} /> 5 Models Compared</span>
            </div>
          </div>
        </div>

        {/* Header Tab Navigation - Clean LeetCode style tabs */}
        <div style={{ display: 'flex', background: 'var(--bg-subtle)', borderRadius: '8px', padding: '3px', border: '1px solid var(--border-color)' }}>
          <button 
            onClick={() => setActiveTab('metrics')}
            style={{ 
              padding: '7px 14px', borderRadius: '6px', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: '0.82rem', display: 'flex', alignItems: 'center', gap: '6px',
              background: activeTab === 'metrics' ? '#333333' : 'transparent',
              color: activeTab === 'metrics' ? '#ffa116' : 'var(--text-secondary)',
              transition: 'all 0.15s ease'
            }}
          >
            <BarChart2 size={15} /> Metrics Graphs (42)
          </button>
          <button 
            onClick={() => setActiveTab('compare')}
            style={{ 
              padding: '7px 14px', borderRadius: '6px', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: '0.82rem', display: 'flex', alignItems: 'center', gap: '6px',
              background: activeTab === 'compare' ? '#333333' : 'transparent',
              color: activeTab === 'compare' ? '#ffa116' : 'var(--text-secondary)',
              transition: 'all 0.15s ease'
            }}
          >
            <Scale size={15} /> Compare 2 Models (Head-to-Head)
          </button>
          <button 
            onClick={() => setActiveTab('inspector')}
            style={{ 
              padding: '7px 14px', borderRadius: '6px', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: '0.82rem', display: 'flex', alignItems: 'center', gap: '6px',
              background: activeTab === 'inspector' ? '#333333' : 'transparent',
              color: activeTab === 'inspector' ? '#ffa116' : 'var(--text-secondary)',
              transition: 'all 0.15s ease'
            }}
          >
            <FileText size={15} /> Prediction Inspector
          </button>
          <button 
            onClick={() => setActiveTab('company')}
            style={{ 
              padding: '7px 14px', borderRadius: '6px', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: '0.82rem', display: 'flex', alignItems: 'center', gap: '6px',
              background: activeTab === 'company' ? '#333333' : 'transparent',
              color: activeTab === 'company' ? '#ffa116' : 'var(--text-secondary)',
              transition: 'all 0.15s ease'
            }}
          >
            <TrendingUp size={15} /> Company Breakdown
          </button>
          <button 
            onClick={() => setActiveTab('table')}
            style={{ 
              padding: '7px 14px', borderRadius: '6px', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: '0.82rem', display: 'flex', alignItems: 'center', gap: '6px',
              background: activeTab === 'table' ? '#333333' : 'transparent',
              color: activeTab === 'table' ? '#ffa116' : 'var(--text-secondary)',
              transition: 'all 0.15s ease'
            }}
          >
            <FileText size={15} /> Master Table (42)
          </button>
        </div>
      </header>

      {/* Main Container */}
      <main style={{ padding: '24px', flex: 1 }}>
        {/* Quick Stats Top Cards (Core + Derived Highlights) */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '16px', marginBottom: '24px' }}>
          <div className="glass-card" style={{ padding: '16px', display: 'flex', alignItems: 'center', gap: '14px' }}>
            <div style={{ background: '#2c2c2c', padding: '10px', borderRadius: '8px', color: '#38bdf8', border: '1px solid #383838' }}>
              <Award size={22} />
            </div>
            <div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Top EM Performer</div>
              <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '2px' }}>
                {masterData.length > 0 ? getTopModelForMetric("Exact Match")?.Model : "Loading..."}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#2cbb5d', fontWeight: 600, marginTop: '1px' }}>
                {masterData.length > 0 ? `${((getTopModelForMetric("Exact Match")?.["Exact Match"] ?? 0) * 100).toFixed(1)}% Exact Match` : ""}
              </div>
            </div>
          </div>

          <div className="glass-card" style={{ padding: '16px', display: 'flex', alignItems: 'center', gap: '14px' }}>
            <div style={{ background: '#2c2c2c', padding: '10px', borderRadius: '8px', color: '#ffa116', border: '1px solid #383838' }}>
              <Sparkles size={22} />
            </div>
            <div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Highest Judge Score</div>
              <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '2px' }}>
                {masterData.length > 0 ? getTopModelForMetric("Judge Score")?.Model : "Loading..."}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#ffa116', fontWeight: 600, marginTop: '1px' }}>
                {masterData.length > 0 ? `${(getTopModelForMetric("Judge Score")?.["Judge Score"] ?? 0).toFixed(2)} / 5.0 Rating` : ""}
              </div>
            </div>
          </div>

          <div className="glass-card" style={{ padding: '16px', display: 'flex', alignItems: 'center', gap: '14px' }}>
            <div style={{ background: '#2c2c2c', padding: '10px', borderRadius: '8px', color: '#14b8a6', border: '1px solid #383838' }}>
              <ShieldCheck size={22} />
            </div>
            <div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Most Reliable Acc</div>
              <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '2px' }}>
                {masterData.length > 0 ? getTopModelForMetric("Reliable Accuracy")?.Model : "Loading..."}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#14b8a6', fontWeight: 600, marginTop: '1px' }}>
                {masterData.length > 0 ? `${((getTopModelForMetric("Reliable Accuracy")?.["Reliable Accuracy"] ?? 0) * 100).toFixed(1)}% Reliable` : ""}
              </div>
            </div>
          </div>

          <div className="glass-card" style={{ padding: '16px', display: 'flex', alignItems: 'center', gap: '14px' }}>
            <div style={{ background: '#2c2c2c', padding: '10px', borderRadius: '8px', color: '#8b5cf6', border: '1px solid #383838' }}>
              <CheckCircle2 size={22} />
            </div>
            <div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Recoverable Answer</div>
              <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '2px' }}>
                {masterData.length > 0 ? getTopModelForMetric("Answer Recoverable Rate")?.Model : "Loading..."}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#8b5cf6', fontWeight: 600, marginTop: '1px' }}>
                {masterData.length > 0 ? `${((getTopModelForMetric("Answer Recoverable Rate")?.["Answer Recoverable Rate"] ?? 0) * 100).toFixed(1)}% Recoverable` : ""}
              </div>
            </div>
          </div>

          <div className="glass-card" style={{ padding: '16px', display: 'flex', alignItems: 'center', gap: '14px' }}>
            <div style={{ background: '#2c2c2c', padding: '10px', borderRadius: '8px', color: '#f43f5e', border: '1px solid #383838' }}>
              <Filter size={22} />
            </div>
            <div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Lowest Leak Rate</div>
              <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '2px' }}>
                {masterData.length > 0 ? getTopModelForMetric("FN Call Leak Rate")?.Model : "Loading..."}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#2cbb5d', fontWeight: 600, marginTop: '1px' }}>
                {masterData.length > 0 ? `${((getTopModelForMetric("FN Call Leak Rate")?.["FN Call Leak Rate"] ?? 0) * 100).toFixed(1)}% Leakage` : ""}
              </div>
            </div>
          </div>

          <div className="glass-card" style={{ padding: '16px', display: 'flex', alignItems: 'center', gap: '14px' }}>
            <div style={{ background: '#2c2c2c', padding: '10px', borderRadius: '8px', color: '#2cbb5d', border: '1px solid #383838' }}>
              <Zap size={22} />
            </div>
            <div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Fastest Generation</div>
              <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '2px' }}>
                {masterData.length > 0 ? getTopModelForMetric("Latency (s)")?.Model : "Loading..."}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#2cbb5d', fontWeight: 600, marginTop: '1px' }}>
                {masterData.length > 0 ? `${(getTopModelForMetric("Latency (s)")?.["Latency (s)"] ?? 0).toFixed(2)}s Latency` : ""}
              </div>
            </div>
          </div>
        </div>

        {/* TAB 1: SEPARATE BAR GRAPHS FOR ALL 42 METRICS */}
        {activeTab === 'metrics' && (
          <div>
            {/* Filter Bar for Metric Categories */}
            <div className="glass-panel" style={{ padding: '14px 20px', marginBottom: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Filter size={16} color="var(--accent-orange)" />
                <span style={{ fontWeight: 600, fontSize: '0.88rem', color: 'var(--text-primary)' }}>Filter Metric Category ({filteredMetrics.length} shown):</span>
              </div>
              <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                {[
                  'All (42)', 'Core 15', 'Derived 27',
                  'Accuracy & Match', 'LLM Judge & Quality', 'Faithfulness & Safety', 'Performance & Speed',
                  'Format & Leakage', 'Task Granularity', 'Judge Alignment', 'Reliability & Grounding', 'Efficiency & Scaling', 'Robustness & CI'
                ].map(cat => (
                  <button
                    key={cat}
                    onClick={() => setMetricFilter(cat)}
                    style={{
                      padding: '5px 11px', borderRadius: '6px', border: metricFilter === cat ? '1px solid var(--accent-orange)' : '1px solid var(--border-color)',
                      background: metricFilter === cat ? '#333333' : 'var(--bg-subtle)',
                      color: metricFilter === cat ? '#ffa116' : 'var(--text-secondary)',
                      cursor: 'pointer', fontSize: '0.78rem', fontWeight: 600, transition: 'all 0.15s ease'
                    }}
                  >
                    {cat}
                  </button>
                ))}
              </div>
            </div>

            {/* Grid of 15 Individual Bar Graphs */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '20px' }}>
              {filteredMetrics.map(metricKey => {
                const def = METRIC_DEFINITIONS[metricKey];
                const topModel = getTopModelForMetric(metricKey);
                
                return (
                  <div key={metricKey} className="glass-panel" style={{ padding: '18px', display: 'flex', flexDirection: 'column' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '10px' }}>
                      <div>
                        <h3 style={{ fontSize: '0.98rem', fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: def.color }}></span>
                          {def.title}
                        </h3>
                        <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '3px' }}>
                          {def.description}
                        </p>
                      </div>
                      {topModel && (
                        <div className="badge badge-orange" title="Highest scoring model on this metric">
                          🏆 {topModel.Model}
                        </div>
                      )}
                    </div>

                    {/* Chart Container */}
                    <div style={{ width: '100%', height: '210px', marginTop: '6px' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={masterData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#333333" />
                          <XAxis 
                            dataKey="Model" 
                            stroke="var(--text-secondary)" 
                            fontSize={11} 
                            tickLine={false}
                            interval={0}
                            angle={-15}
                            textAnchor="end"
                          />
                          <YAxis 
                            stroke="var(--text-secondary)" 
                            fontSize={11} 
                            tickLine={false} 
                            tickFormatter={(val) => def.isLowerBetter || def.unit === 's' || def.unit === 'Score' || def.unit === '/ 5.0' ? val : (val * 100).toFixed(0) + '%'}
                          />
                          <Tooltip 
                            contentStyle={{ background: '#262626', borderColor: '#3e3e3e', borderRadius: '6px' }}
                            formatter={(value: any) => [def.format ? def.format(Number(value)) : value, def.title]}
                          />
                          <Bar dataKey={metricKey} radius={[4, 4, 0, 0]}>
                            {masterData.map((entry, index) => (
                              <Cell 
                                key={`cell-${index}`} 
                                fill={MODEL_COLOR_MAP[entry.Model] || def.color} 
                                opacity={entry.Model === topModel?.Model ? 1.0 : 0.75}
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* TAB 2: HEAD-TO-HEAD COMPARISON OF TWO SPECIFIC MODELS */}
        {activeTab === 'compare' && (
          <ModelComparison
            masterData={masterData}
            categoryData={categoryData}
            modelsList={modelsList}
            metricDefinitions={METRIC_DEFINITIONS}
            modelColorMap={MODEL_COLOR_MAP}
            initialSampleId={selectedSampleId}
            sampleIdList={sampleIdList}
          />
        )}

        {/* TAB 3: PROMPT & PREDICTION INSPECTOR FOR JUDGES */}
        {activeTab === 'inspector' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {/* Control Bar */}
            <div className="glass-panel" style={{ padding: '16px 20px', display: 'flex', gap: '16px', alignItems: 'center', flexWrap: 'wrap', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', gap: '16px', alignItems: 'center', flexWrap: 'wrap', flex: 1 }}>
                {/* Random Button */}
                <button className="btn-random" onClick={handlePickRandomSample} title="Select a random data point for judges inspection">
                  <Shuffle size={16} /> Pick Random Sample
                </button>

                {/* Sample ID Search & Selector */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: 1, minWidth: '260px' }}>
                  <Search size={16} color="var(--text-secondary)" />
                  <select 
                    className="custom-select"
                    value={selectedSampleId} 
                    onChange={(e) => setSelectedSampleId(e.target.value)}
                    style={{ width: '100%', fontWeight: 600 }}
                  >
                    {sampleIdList.map(id => (
                      <option key={id} value={id}>{id}</option>
                    ))}
                  </select>
                </div>

                {/* Model Selector (For Single Model view mode) */}
                {!compareAllMode && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Model:</span>
                    <select 
                      className="custom-select"
                      value={selectedModel} 
                      onChange={(e) => setSelectedModel(e.target.value)}
                      style={{ fontWeight: 600 }}
                    >
                      {modelsList.map(m => (
                        <option key={m} value={m}>{m}</option>
                      ))}
                    </select>
                  </div>
                )}
              </div>

              {/* Compare Mode Toggle */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px', background: 'var(--bg-subtle)', padding: '3px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                <button
                  onClick={() => setCompareAllMode(true)}
                  style={{
                    padding: '6px 12px', borderRadius: '4px', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: '0.8rem',
                    background: compareAllMode ? '#333333' : 'transparent',
                    color: compareAllMode ? '#ffa116' : 'var(--text-secondary)'
                  }}
                >
                  ⚡ Compare All 5 Models
                </button>
                <button
                  onClick={() => setCompareAllMode(false)}
                  style={{
                    padding: '6px 12px', borderRadius: '4px', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: '0.8rem',
                    background: !compareAllMode ? '#333333' : 'transparent',
                    color: !compareAllMode ? '#ffa116' : 'var(--text-secondary)'
                  }}
                >
                  Single Model View
                </button>
              </div>
            </div>

            {loadingSample ? (
              <div className="glass-panel" style={{ padding: '60px', textAlign: 'center', color: 'var(--text-secondary)' }}>
                Loading sample data point {selectedSampleId}...
              </div>
            ) : sampleComparison ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                {/* Context & Prompt Card */}
                <div className="glass-panel" style={{ padding: '20px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '14px', flexWrap: 'wrap', gap: '12px' }}>
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <h2 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#38bdf8' }}>{sampleComparison.id}</h2>
                        <span className="badge badge-neutral">{sampleComparison.metadata?.company || 'Category / Group'}</span>
                        <span className="badge badge-neutral">{sampleComparison.metadata?.task_type || 'qa_numeric_program'}</span>
                      </div>
                      <p style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)', marginTop: '8px' }}>
                        Question: "{sampleComparison.input_text}"
                      </p>
                    </div>

                    <div style={{ background: 'var(--bg-subtle)', border: '1px solid var(--border-light)', padding: '10px 16px', borderRadius: '6px', textAlign: 'right' }}>
                      <div style={{ fontSize: '0.72rem', color: '#2cbb5d', textTransform: 'uppercase', fontWeight: 700 }}>Ground Truth Target</div>
                      <div style={{ fontSize: '1.3rem', fontWeight: 800, color: 'var(--text-primary)' }}>{sampleComparison.target}</div>
                      {sampleComparison.metadata?.program && (
                        <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', fontFamily: 'monospace', marginTop: '2px' }}>
                          Formula: {sampleComparison.metadata.program}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Financial Context Table */}
                  {sampleComparison.context && (
                    <div style={{ marginTop: '14px', background: '#181818', padding: '14px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                      <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px', textTransform: 'uppercase' }}>Financial Disclosure Context Table:</div>
                      <pre style={{ fontSize: '0.84rem', color: '#d4d4d4', whiteSpace: 'pre-wrap', lineHeight: '1.5' }}>
                        {sampleComparison.context}
                      </pre>
                    </div>
                  )}

                  {/* Complete Prompt Toggle */}
                  {sampleComparison.prompt && (
                    <details style={{ marginTop: '14px' }}>
                      <summary style={{ cursor: 'pointer', fontSize: '0.82rem', fontWeight: 600, color: '#38bdf8' }}>
                        📜 View Full LLM System Prompt & Instructions
                      </summary>
                      <pre style={{ marginTop: '8px', background: '#181818', padding: '12px', borderRadius: '6px', border: '1px solid var(--border-color)', fontSize: '0.8rem', color: '#9e9e9e', whiteSpace: 'pre-wrap' }}>
                        {sampleComparison.prompt}
                      </pre>
                    </details>
                  )}
                </div>

                {/* MODEL COMPARISON CARDS */}
                {compareAllMode ? (
                  <div>
                    <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-primary)' }}>
                      <Sparkles color="var(--accent-orange)" size={18} /> 
                      Side-by-Side Model Outputs on sample <span style={{ color: '#38bdf8' }}>{sampleComparison.id}</span>:
                    </h3>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '16px' }}>
                      {Object.keys(sampleComparison.models).map(mKey => {
                        const mData = sampleComparison.models[mKey];
                        const isEM = mData.exact_match === 1.0;
                        
                        return (
                          <div 
                            key={mKey} 
                            className="glass-panel" 
                            style={{ 
                              padding: '16px', 
                              border: isEM ? '1px solid rgba(44, 187, 93, 0.4)' : '1px solid rgba(239, 71, 67, 0.3)',
                              background: isEM ? 'rgba(44, 187, 93, 0.03)' : 'rgba(239, 71, 67, 0.03)',
                              display: 'flex', flexDirection: 'column', justifyContent: 'space-between'
                            }}
                          >
                            <div>
                              {/* Model Card Header */}
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                                <span style={{ fontWeight: 700, fontSize: '0.95rem', color: MODEL_COLOR_MAP[mKey] || 'var(--text-primary)' }}>
                                  {mKey}
                                </span>
                                {isEM ? (
                                  <span className="badge badge-emerald"><CheckCircle2 size={12} /> Exact Match 1.0</span>
                                ) : (
                                  <span className="badge badge-rose"><XCircle size={12} /> Mis-match 0.0</span>
                                )}
                              </div>

                              {/* Reasoning Output text */}
                              <div style={{ background: '#141414', padding: '12px', borderRadius: '6px', minHeight: '140px', fontSize: '0.82rem', color: '#d4d4d4', whiteSpace: 'pre-wrap', border: '1px solid #2a2a2a', marginBottom: '12px' }}>
                                {mData.output_text || "No output response generated."}
                              </div>
                            </div>

                            {/* Metric Badges Footnote */}
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', paddingTop: '10px', borderTop: '1px solid var(--border-color)' }}>
                              <span className="badge badge-amber" title="LLM Judge Score">
                                ⭐ {mData.judge_score.toFixed(1)} / 5
                              </span>
                              <span className="badge badge-cyan" title="Latency">
                                ⏱️ {mData.latency_seconds.toFixed(2)}s
                              </span>
                              <span className="badge badge-purple" title="BERTScore">
                                🎯 BERT: {mData.bert_score.toFixed(2)}
                              </span>
                              {mData.is_hallucinated && (
                                <span className="badge badge-rose">⚠️ Hallucinated</span>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ) : (
                  /* Single Model Inspection View */
                  sampleComparison.models[selectedModel] ? (
                    <div className="glass-panel" style={{ padding: '20px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: MODEL_COLOR_MAP[selectedModel] || 'var(--text-primary)' }}>
                          Model: {selectedModel} Response Inspection
                        </h3>
                        <div style={{ display: 'flex', gap: '8px' }}>
                          <span className={`badge ${sampleComparison.models[selectedModel].exact_match === 1.0 ? 'badge-emerald' : 'badge-rose'}`}>
                            Exact Match: {sampleComparison.models[selectedModel].exact_match}
                          </span>
                          <span className="badge badge-amber">
                            Judge Rating: {sampleComparison.models[selectedModel].judge_score} / 5.0
                          </span>
                        </div>
                      </div>

                      <div style={{ background: '#141414', padding: '14px', borderRadius: '6px', fontSize: '0.88rem', color: '#d4d4d4', whiteSpace: 'pre-wrap', border: '1px solid var(--border-color)', marginBottom: '14px' }}>
                        {sampleComparison.models[selectedModel].output_text}
                      </div>

                      {/* Detailed Metric Chips */}
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '10px' }}>
                        <div className="glass-card" style={{ padding: '10px', textAlign: 'center' }}>
                          <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>F1 Score</div>
                          <div style={{ fontSize: '1rem', fontWeight: 700, color: '#38bdf8', marginTop: '2px' }}>{sampleComparison.models[selectedModel].f1}</div>
                        </div>
                        <div className="glass-card" style={{ padding: '10px', textAlign: 'center' }}>
                          <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>BLEU Score</div>
                          <div style={{ fontSize: '1rem', fontWeight: 700, color: '#a855f7', marginTop: '2px' }}>{sampleComparison.models[selectedModel].bleu}</div>
                        </div>
                        <div className="glass-card" style={{ padding: '10px', textAlign: 'center' }}>
                          <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>ROUGE Score</div>
                          <div style={{ fontSize: '1rem', fontWeight: 700, color: '#2cbb5d', marginTop: '2px' }}>{sampleComparison.models[selectedModel].rouge}</div>
                        </div>
                        <div className="glass-card" style={{ padding: '10px', textAlign: 'center' }}>
                          <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>BERTScore</div>
                          <div style={{ fontSize: '1rem', fontWeight: 700, color: '#ffa116', marginTop: '2px' }}>{sampleComparison.models[selectedModel].bert_score}</div>
                        </div>
                        <div className="glass-card" style={{ padding: '10px', textAlign: 'center' }}>
                          <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>Latency</div>
                          <div style={{ fontSize: '1rem', fontWeight: 700, color: '#2cbb5d', marginTop: '2px' }}>{sampleComparison.models[selectedModel].latency_seconds}s</div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="glass-panel" style={{ padding: '30px', textAlign: 'center', color: 'var(--text-secondary)' }}>
                      No data recorded for model {selectedModel} on sample {selectedSampleId}.
                    </div>
                  )
                )}
              </div>
            ) : null}
          </div>
        )}

        {/* TAB 3: COMPANY ACCURACY BREAKDOWN */}
        {activeTab === 'company' && (
          <div className="glass-panel" style={{ padding: '20px' }}>
            <h2 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '14px', color: 'var(--text-primary)' }}>Company Breakdown Accuracy</h2>
            <div style={{ width: '100%', height: '400px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={categoryData.filter(d => d.group_type === 'company')} margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333333" />
                  <XAxis dataKey="group" stroke="var(--text-secondary)" fontSize={12} />
                  <YAxis stroke="var(--text-secondary)" fontSize={12} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip contentStyle={{ background: '#262626', borderColor: '#3e3e3e' }} formatter={(val: any) => [`${(Number(val) * 100).toFixed(1)}%`, 'Accuracy']} />
                  <Bar dataKey="accuracy" fill="#38bdf8" radius={[4, 4, 0, 0]}>
                    {categoryData.filter(d => d.group_type === 'company').map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={MODEL_COLOR_MAP[entry.Model] || '#38bdf8'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* TAB 4: MASTER TABLE */}
        {activeTab === 'table' && (
          <div className="glass-panel" style={{ padding: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '14px', marginBottom: '16px' }}>
              <div>
                <h2 style={{ fontSize: '1.15rem', fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Award size={18} color="var(--accent-orange)" />
                  Consolidated Benchmark Master Table ({tableFilter === 'all' ? 'All 42 Metrics' : tableFilter === 'core' ? '15 Core Metrics' : '27 Derived Metrics'})
                </h2>
                <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                  High-density multi-model benchmark matrix with automated score formatting and unit normalization.
                </p>
              </div>

              {/* View Toggle Buttons */}
              <div style={{ display: 'flex', background: 'var(--bg-subtle)', borderRadius: '6px', padding: '3px', border: '1px solid var(--border-color)', gap: '4px' }}>
                <button
                  onClick={() => setTableFilter('all')}
                  style={{
                    padding: '6px 12px', borderRadius: '4px', border: 'none', cursor: 'pointer', fontSize: '0.78rem', fontWeight: 600,
                    background: tableFilter === 'all' ? '#333333' : 'transparent',
                    color: tableFilter === 'all' ? 'var(--accent-orange)' : 'var(--text-secondary)',
                    transition: 'all 0.15s ease'
                  }}
                >
                  All 42 Metrics
                </button>
                <button
                  onClick={() => setTableFilter('core')}
                  style={{
                    padding: '6px 12px', borderRadius: '4px', border: 'none', cursor: 'pointer', fontSize: '0.78rem', fontWeight: 600,
                    background: tableFilter === 'core' ? '#333333' : 'transparent',
                    color: tableFilter === 'core' ? 'var(--accent-orange)' : 'var(--text-secondary)',
                    transition: 'all 0.15s ease'
                  }}
                >
                  Core 15 Metrics
                </button>
                <button
                  onClick={() => setTableFilter('derived')}
                  style={{
                    padding: '6px 12px', borderRadius: '4px', border: 'none', cursor: 'pointer', fontSize: '0.78rem', fontWeight: 600,
                    background: tableFilter === 'derived' ? '#333333' : 'transparent',
                    color: tableFilter === 'derived' ? 'var(--accent-orange)' : 'var(--text-secondary)',
                    transition: 'all 0.15s ease'
                  }}
                >
                  Derived 27 Metrics
                </button>
              </div>
            </div>

            <div style={{ overflowX: 'auto', border: '1px solid var(--border-color)', borderRadius: '8px' }}>
              <table className="table-custom" style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
                <thead>
                  <tr style={{ background: '#252525', borderBottom: '1px solid var(--border-color)' }}>
                    <th style={{ position: 'sticky', left: 0, background: '#252525', zIndex: 2, padding: '10px 14px', textAlign: 'left', minWidth: '180px' }}>
                      Model Name
                    </th>
                    {Object.keys(METRIC_DEFINITIONS)
                      .filter(k => {
                        const isCore = CORE_CATEGORIES.includes(METRIC_DEFINITIONS[k].category);
                        if (tableFilter === 'core') return isCore;
                        if (tableFilter === 'derived') return !isCore;
                        return true;
                      })
                      .map(k => (
                        <th key={k} style={{ padding: '10px 12px', textAlign: 'right', whiteSpace: 'nowrap', minWidth: '120px' }} title={METRIC_DEFINITIONS[k].description}>
                          <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{k}</div>
                          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 400 }}>{METRIC_DEFINITIONS[k].category}</div>
                        </th>
                      ))}
                  </tr>
                </thead>
                <tbody>
                  {masterData.map(m => (
                    <tr key={m.Model} style={{ borderBottom: '1px solid #2a2a2a' }}>
                      <td style={{ position: 'sticky', left: 0, background: 'var(--bg-subtle)', zIndex: 1, fontWeight: 700, padding: '10px 14px', color: MODEL_COLOR_MAP[m.Model] || 'var(--text-primary)' }}>
                        {m.Model}
                      </td>
                      {Object.keys(METRIC_DEFINITIONS)
                        .filter(k => {
                          const isCore = CORE_CATEGORIES.includes(METRIC_DEFINITIONS[k].category);
                          if (tableFilter === 'core') return isCore;
                          if (tableFilter === 'derived') return !isCore;
                          return true;
                        })
                        .map(k => {
                          const def = METRIC_DEFINITIONS[k];
                          const rawVal = m[k];
                          const formatted = (rawVal !== undefined && rawVal !== null)
                            ? (def.format ? def.format(rawVal) : String(rawVal))
                            : "-";
                          
                          const topModel = getTopModelForMetric(k);
                          const isBest = topModel?.Model === m.Model;

                          return (
                            <td key={k} style={{ padding: '10px 12px', textAlign: 'right', whiteSpace: 'nowrap' }}>
                              {isBest ? (
                                <span style={{ fontWeight: 700, color: '#2cbb5d' }}>
                                  {formatted}
                                </span>
                              ) : (
                                <span style={{ color: 'var(--text-secondary)' }}>
                                  {formatted}
                                </span>
                              )}
                            </td>
                          );
                        })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
