import { useState, useEffect, useMemo } from 'react';
import { 
  ArrowLeftRight, Trophy, CheckCircle2, 
  XCircle, Shuffle, Search, Award, Scale, Info,
  TrendingUp, BarChart2, Eye
} from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  Legend, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';

export interface ModelAggregates {
  Model: string;
  "Exact Match": number;
  F1: number;
  BLEU: number;
  ROUGE: number;
  "Judge Score": number;
  Completeness: number;
  Helpfulness: number;
  "Reasoning Faithfulness": number;
  BERTScore: number;
  "Hallucination Rate": number;
  "Failure Rate": number;
  "Latency (s)": number;
  "Throughput (req/s)": number;
  "Tokens/sec": number;
  "Category Accuracy": number;
}

export interface MetricDef {
  title: string;
  category: string;
  description: string;
  unit: string;
  isLowerBetter?: boolean;
  format?: (v: number) => string;
  color: string;
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
    }
  }
}

interface ModelComparisonProps {
  masterData: ModelAggregates[];
  categoryData: any[];
  modelsList: string[];
  metricDefinitions: { [key: string]: MetricDef };
  modelColorMap: { [key: string]: string };
  initialSampleId?: string;
  sampleIdList?: string[];
}

export default function ModelComparison({
  masterData,
  categoryData,
  modelsList,
  metricDefinitions,
  modelColorMap,
  initialSampleId = 'synthetic_bfsi_000284',
  sampleIdList = []
}: ModelComparisonProps) {
  // Available models from masterData
  const availableModels = useMemo(() => {
    if (masterData.length > 0) {
      return masterData.map(m => m.Model);
    }
    return modelsList;
  }, [masterData, modelsList]);

  // Selected models for head-to-head comparison
  const [modelA, setModelA] = useState<string>('qwen2.5-7b');
  const [modelB, setModelB] = useState<string>('llama3.1-8b');

  // Sync state if models load and defaults aren't available
  useEffect(() => {
    if (availableModels.length >= 2) {
      if (!availableModels.includes(modelA)) {
        setModelA(availableModels[0]);
      }
      if (!availableModels.includes(modelB)) {
        setModelB(availableModels[1] || availableModels[0]);
      }
    }
  }, [availableModels]);

  const [activeCategory, setActiveCategory] = useState<string>('All');
  const [activeChartTab, setActiveChartTab] = useState<'accuracy' | 'judge' | 'speed' | 'safety' | 'radar'>('accuracy');
  
  // Sample deep-dive inspection state
  const [sampleId, setSampleId] = useState<string>(initialSampleId);
  const [sampleData, setSampleData] = useState<SampleComparison | null>(null);
  const [loadingSample, setLoadingSample] = useState<boolean>(false);

  useEffect(() => {
    if (initialSampleId && !sampleId) {
      setSampleId(initialSampleId);
    }
  }, [initialSampleId]);

  useEffect(() => {
    if (!sampleId) return;
    setLoadingSample(true);
    fetch(`/api/sample/${sampleId}`)
      .then(res => res.json())
      .then(data => {
        setSampleData(data);
        setLoadingSample(false);
      })
      .catch(err => {
        console.error("Error loading sample comparison:", err);
        setLoadingSample(false);
      });
  }, [sampleId]);

  const handlePickRandomSample = () => {
    fetch('/api/random-id')
      .then(res => res.json())
      .then(data => {
        if (data.id) setSampleId(data.id);
      })
      .catch(err => console.error("Error picking random ID:", err));
  };

  const handleSwapModels = () => {
    const temp = modelA;
    setModelA(modelB);
    setModelB(temp);
  };

  const getColor = (m: string) => modelColorMap[m] || '#38bdf8';

  const colorA = getColor(modelA);
  const colorB = getColor(modelB);

  // Model records
  const dataA = useMemo(() => masterData.find(m => m.Model === modelA), [masterData, modelA]);
  const dataB = useMemo(() => masterData.find(m => m.Model === modelB), [masterData, modelB]);

  // Comparison metrics analysis
  const comparisonStats = useMemo(() => {
    if (!dataA || !dataB) return { winsA: 0, winsB: 0, ties: 0, metricRows: [], strengthsA: [], strengthsB: [] };

    let winsA = 0;
    let winsB = 0;
    let ties = 0;
    const strengthsA: { metric: string; detail: string }[] = [];
    const strengthsB: { metric: string; detail: string }[] = [];

    const metricRows = Object.keys(metricDefinitions).map(metricKey => {
      const def = metricDefinitions[metricKey];
      const valA = (dataA as any)[metricKey] ?? 0;
      const valB = (dataB as any)[metricKey] ?? 0;
      const isLower = !!def.isLowerBetter;

      let winner: 'A' | 'B' | 'tie' = 'tie';
      if (Math.abs(valA - valB) > 0.00001) {
        if (isLower) {
          winner = valA < valB ? 'A' : 'B';
        } else {
          winner = valA > valB ? 'A' : 'B';
        }
      }

      if (winner === 'A') winsA++;
      else if (winner === 'B') winsB++;
      else ties++;

      // Compute Delta
      const rawDiff = valA - valB;
      let deltaText = '';
      let deltaDirection: 'A' | 'B' | 'tie' = winner;

      if (def.unit === '%') {
        const diffPts = (valA - valB) * 100;
        deltaText = `${diffPts >= 0 ? '+' : ''}${diffPts.toFixed(1)}%`;
      } else if (def.unit === 's') {
        const diffSec = valA - valB;
        const ratio = valA > 0 && valB > 0 ? (valA > valB ? valA / valB : valB / valA) : 1;
        if (winner === 'A') {
          deltaText = `${Math.abs(diffSec).toFixed(2)}s faster (${ratio.toFixed(1)}x)`;
        } else if (winner === 'B') {
          deltaText = `${Math.abs(diffSec).toFixed(2)}s faster (${ratio.toFixed(1)}x)`;
        } else {
          deltaText = 'Identical speed';
        }
      } else if (def.unit === 'tok/s') {
        const diffTok = valA - valB;
        deltaText = `${diffTok >= 0 ? '+' : ''}${diffTok.toFixed(1)} tok/s`;
      } else if (def.unit === '/ 5.0') {
        const diffScore = valA - valB;
        deltaText = `${diffScore >= 0 ? '+' : ''}${diffScore.toFixed(2)} pts`;
      } else {
        deltaText = `${rawDiff >= 0 ? '+' : ''}${rawDiff.toFixed(3)}`;
      }

      // Add to strengths if notable win
      if (winner === 'A') {
        if (metricKey === 'Exact Match') strengthsA.push({ metric: 'Exact Match', detail: `${(valA * 100).toFixed(1)}% vs ${(valB * 100).toFixed(1)}% (+${((valA - valB) * 100).toFixed(1)}%)` });
        else if (metricKey === 'Judge Score') strengthsA.push({ metric: 'LLM Judge', detail: `${valA.toFixed(2)} vs ${valB.toFixed(2)} (+${(valA - valB).toFixed(2)})` });
        else if (metricKey === 'Latency (s)') strengthsA.push({ metric: 'Generation Speed', detail: `${valA.toFixed(2)}s vs ${valB.toFixed(2)}s (${(valB / valA).toFixed(1)}x faster)` });
        else if (metricKey === 'Hallucination Rate') strengthsA.push({ metric: 'Lower Hallucination', detail: `${(valA * 100).toFixed(1)}% vs ${(valB * 100).toFixed(1)}%` });
        else if (metricKey === 'Reasoning Faithfulness') strengthsA.push({ metric: 'Faithfulness', detail: `${(valA * 100).toFixed(1)}% vs ${(valB * 100).toFixed(1)}%` });
        else if (metricKey === 'Tokens/sec') strengthsA.push({ metric: 'Output Tok/s', detail: `${valA.toFixed(1)} vs ${valB.toFixed(1)} tok/s` });
      } else if (winner === 'B') {
        if (metricKey === 'Exact Match') strengthsB.push({ metric: 'Exact Match', detail: `${(valB * 100).toFixed(1)}% vs ${(valA * 100).toFixed(1)}% (+${((valB - valA) * 100).toFixed(1)}%)` });
        else if (metricKey === 'Judge Score') strengthsB.push({ metric: 'LLM Judge', detail: `${valB.toFixed(2)} vs ${valA.toFixed(2)} (+${(valB - valA).toFixed(2)})` });
        else if (metricKey === 'Latency (s)') strengthsB.push({ metric: 'Generation Speed', detail: `${valB.toFixed(2)}s vs ${valA.toFixed(2)}s (${(valA / valB).toFixed(1)}x faster)` });
        else if (metricKey === 'Hallucination Rate') strengthsB.push({ metric: 'Lower Hallucination', detail: `${(valB * 100).toFixed(1)}% vs ${(valA * 100).toFixed(1)}%` });
        else if (metricKey === 'Reasoning Faithfulness') strengthsB.push({ metric: 'Faithfulness', detail: `${(valB * 100).toFixed(1)}% vs ${(valA * 100).toFixed(1)}%` });
        else if (metricKey === 'Tokens/sec') strengthsB.push({ metric: 'Output Tok/s', detail: `${valB.toFixed(1)} vs ${valA.toFixed(1)} tok/s` });
      }

      return {
        key: metricKey,
        def,
        valA,
        valB,
        winner,
        deltaText,
        deltaDirection
      };
    });

    return { winsA, winsB, ties, metricRows, strengthsA, strengthsB };
  }, [dataA, dataB, metricDefinitions]);

  // Radar chart data comparing 6 holistic dimensions (0 - 100 normalized scale)
  const radarData = useMemo(() => {
    if (!dataA || !dataB) return [];

    return [
      {
        dimension: 'Exact Match',
        [modelA]: Math.round(((dataA["Exact Match"] ?? 0) * 100)),
        [modelB]: Math.round(((dataB["Exact Match"] ?? 0) * 100)),
        fullMark: 100
      },
      {
        dimension: 'LLM Judge',
        [modelA]: Math.round((((dataA["Judge Score"] ?? 0) / 5.0) * 100)),
        [modelB]: Math.round((((dataB["Judge Score"] ?? 0) / 5.0) * 100)),
        fullMark: 100
      },
      {
        dimension: 'Reasoning',
        [modelA]: Math.round(((dataA["Reasoning Faithfulness"] ?? 0) * 100)),
        [modelB]: Math.round(((dataB["Reasoning Faithfulness"] ?? 0) * 100)),
        fullMark: 100
      },
      {
        dimension: 'Anti-Hallucination',
        [modelA]: Math.max(0, Math.round((1 - (dataA["Hallucination Rate"] ?? 0)) * 100)),
        [modelB]: Math.max(0, Math.round((1 - (dataB["Hallucination Rate"] ?? 0)) * 100)),
        fullMark: 100
      },
      {
        dimension: 'Speed Tok/s',
        [modelA]: Math.min(100, Math.round(((dataA["Tokens/sec"] ?? 0) / 55) * 100)),
        [modelB]: Math.min(100, Math.round(((dataB["Tokens/sec"] ?? 0) / 55) * 100)),
        fullMark: 100
      },
      {
        dimension: 'Completeness',
        [modelA]: Math.round((((dataA["Completeness"] ?? 0) / 5.0) * 100)),
        [modelB]: Math.round((((dataB["Completeness"] ?? 0) / 5.0) * 100)),
        fullMark: 100
      }
    ];
  }, [dataA, dataB, modelA, modelB]);

  // Company breakdown comparison data
  const companyComparisonData = useMemo(() => {
    if (!categoryData || categoryData.length === 0) return [];
    
    // Extract unique companies
    const companyNames = Array.from(new Set(
      categoryData.filter(d => d.group_type === 'company').map(d => d.group)
    )) as string[];

    return companyNames.map(company => {
      const recA = categoryData.find(d => d.Model === modelA && d.group === company);
      const recB = categoryData.find(d => d.Model === modelB && d.group === company);
      const accA = recA ? Number(recA.accuracy) * 100 : 0;
      const accB = recB ? Number(recB.accuracy) * 100 : 0;
      return {
        company,
        [modelA]: Number(accA.toFixed(1)),
        [modelB]: Number(accB.toFixed(1)),
        diff: Number((accA - accB).toFixed(1))
      };
    });
  }, [categoryData, modelA, modelB]);

  // Filtered rows for the comparison table
  const filteredMetricRows = useMemo(() => {
    if (activeCategory === 'All') return comparisonStats.metricRows;
    return comparisonStats.metricRows.filter(r => r.def.category === activeCategory);
  }, [comparisonStats.metricRows, activeCategory]);

  // Chart datasets grouped by category
  const accuracyChartData = useMemo(() => {
    if (!dataA || !dataB) return [];
    const keys = ["Exact Match", "F1", "ROUGE", "Reasoning Faithfulness", "Category Accuracy"];
    return keys.map(k => ({
      metric: metricDefinitions[k]?.title || k,
      [modelA]: Number(((dataA as any)[k] * 100).toFixed(1)),
      [modelB]: Number(((dataB as any)[k] * 100).toFixed(1)),
    }));
  }, [dataA, dataB, modelA, modelB, metricDefinitions]);

  const judgeChartData = useMemo(() => {
    if (!dataA || !dataB) return [];
    const keys = ["Judge Score", "Completeness", "Helpfulness"];
    return keys.map(k => ({
      metric: metricDefinitions[k]?.title || k,
      [modelA]: Number((dataA as any)[k].toFixed(2)),
      [modelB]: Number((dataB as any)[k].toFixed(2)),
    }));
  }, [dataA, dataB, modelA, modelB, metricDefinitions]);

  const speedChartData = useMemo(() => {
    if (!dataA || !dataB) return [];
    return [
      {
        metric: "Latency in Seconds (Lower is better)",
        [modelA]: Number((dataA["Latency (s)"] ?? 0).toFixed(2)),
        [modelB]: Number((dataB["Latency (s)"] ?? 0).toFixed(2)),
      },
      {
        metric: "Output Generation (Tokens/sec)",
        [modelA]: Number((dataA["Tokens/sec"] ?? 0).toFixed(1)),
        [modelB]: Number((dataB["Tokens/sec"] ?? 0).toFixed(1)),
      }
    ];
  }, [dataA, dataB, modelA, modelB]);

  const safetyChartData = useMemo(() => {
    if (!dataA || !dataB) return [];
    return [
      {
        metric: "Hallucination Rate % (Lower is better)",
        [modelA]: Number(((dataA["Hallucination Rate"] ?? 0) * 100).toFixed(1)),
        [modelB]: Number(((dataB["Hallucination Rate"] ?? 0) * 100).toFixed(1)),
      },
      {
        metric: "Failure Rate % (Lower is better)",
        [modelA]: Number(((dataA["Failure Rate"] ?? 0) * 100).toFixed(1)),
        [modelB]: Number(((dataB["Failure Rate"] ?? 0) * 100).toFixed(1)),
      },
      {
        metric: "BERTScore Semantic Alignment",
        [modelA]: Number(((dataA["BERTScore"] ?? 0) * 100).toFixed(1)),
        [modelB]: Number(((dataB["BERTScore"] ?? 0) * 100).toFixed(1)),
      }
    ];
  }, [dataA, dataB, modelA, modelB]);

  // Preset pairings for quick testing
  const presets = [
    { name: "Qwen 2.5 vs Llama 3.1", a: "qwen2.5-7b", b: "llama3.1-8b" },
    { name: "Phi-4 Mini vs DeepSeek R1", a: "phi4-mini", b: "deepseek-r1-7b" },
    { name: "Phi-4 Mini vs Qwen 2.5", a: "phi4-mini", b: "qwen2.5-7b" },
    { name: "Mistral 7B vs Llama 3.1", a: "mistral-7b-v0.3", b: "llama3.1-8b" }
  ];

  const totalMetrics = comparisonStats.winsA + comparisonStats.winsB + comparisonStats.ties;
  const pctA = totalMetrics > 0 ? (comparisonStats.winsA / totalMetrics) * 100 : 50;
  const pctB = totalMetrics > 0 ? (comparisonStats.winsB / totalMetrics) * 100 : 50;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
      {/* SECTION 1: MODEL SELECTOR & COMPARISON HEADER BAR */}
      <div className="glass-panel" style={{ padding: '18px 24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px', marginBottom: '16px' }}>
          <div>
            <h2 style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Scale size={22} color="var(--accent-orange)" />
              Head-to-Head Model Benchmark Comparison
            </h2>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
              Select any two BFSI models to contrast side-by-side benchmark numbers, delta advantages, radar capabilities, and sample reasoning.
            </p>
          </div>

          {/* Quick Matchup Presets */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Quick Presets:</span>
            {presets.map(p => (
              <button
                key={p.name}
                onClick={() => { setModelA(p.a); setModelB(p.b); }}
                style={{
                  background: (modelA === p.a && modelB === p.b) ? 'rgba(255, 161, 22, 0.15)' : 'var(--bg-subtle)',
                  border: (modelA === p.a && modelB === p.b) ? '1px solid var(--accent-orange)' : '1px solid var(--border-color)',
                  color: (modelA === p.a && modelB === p.b) ? 'var(--accent-orange)' : 'var(--text-secondary)',
                  padding: '5px 10px',
                  borderRadius: '6px',
                  fontSize: '0.76rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                {p.name}
              </button>
            ))}
          </div>
        </div>

        {/* Dual Model Selectors with Central Swap Button */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: '16px', alignItems: 'center' }}>
          {/* Model A Selector Card */}
          <div style={{ background: 'var(--bg-subtle)', border: `1px solid ${colorA}50`, borderRadius: '8px', padding: '14px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: colorA, textTransform: 'uppercase', letterSpacing: '0.05em', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: colorA }}></span>
                Model A (Reference)
              </span>
              <span className="badge badge-neutral" style={{ fontSize: '0.7rem' }}>
                {comparisonStats.winsA} Metric Wins
              </span>
            </div>
            <select
              className="custom-select"
              value={modelA}
              onChange={(e) => setModelA(e.target.value)}
              style={{ width: '100%', fontSize: '0.95rem', fontWeight: 700, color: colorA, borderColor: `${colorA}80` }}
            >
              {availableModels.map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>

          {/* Swap Button & VS Indicator */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px' }}>
            <button
              onClick={handleSwapModels}
              className="btn-secondary"
              title="Swap Model A and Model B"
              style={{ borderRadius: '50%', width: '42px', height: '42px', padding: 0, justifyContent: 'center' }}
            >
              <ArrowLeftRight size={18} color="var(--accent-orange)" />
            </button>
            <span style={{ fontSize: '0.72rem', fontWeight: 800, color: 'var(--text-secondary)', letterSpacing: '0.08em' }}>
              VS
            </span>
          </div>

          {/* Model B Selector Card */}
          <div style={{ background: 'var(--bg-subtle)', border: `1px solid ${colorB}50`, borderRadius: '8px', padding: '14px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: colorB, textTransform: 'uppercase', letterSpacing: '0.05em', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: colorB }}></span>
                Model B (Challenger)
              </span>
              <span className="badge badge-neutral" style={{ fontSize: '0.7rem' }}>
                {comparisonStats.winsB} Metric Wins
              </span>
            </div>
            <select
              className="custom-select"
              value={modelB}
              onChange={(e) => setModelB(e.target.value)}
              style={{ width: '100%', fontSize: '0.95rem', fontWeight: 700, color: colorB, borderColor: `${colorB}80` }}
            >
              {availableModels.map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>
        </div>

        {modelA === modelB && (
          <div style={{ marginTop: '12px', background: 'rgba(239, 71, 67, 0.1)', border: '1px solid rgba(239, 71, 67, 0.3)', borderRadius: '6px', padding: '10px 14px', fontSize: '0.82rem', color: '#ef4743', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Info size={16} /> Both selectors currently have <strong>{modelA}</strong> selected. Choose a different model for Model B to see side-by-side differences!
          </div>
        )}
      </div>

      {/* SECTION 2: EXECUTIVE HEAD-TO-HEAD SCOREBOARD */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '18px' }}>
        {/* Model A Profile Card */}
        <div 
          className="glass-panel" 
          style={{ 
            padding: '20px', 
            borderLeft: `4px solid ${colorA}`,
            background: comparisonStats.winsA > comparisonStats.winsB ? 'linear-gradient(135deg, rgba(38,38,38,0.9) 0%, rgba(56,189,248,0.05) 100%)' : 'var(--bg-card)'
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600 }}>Model A Candidate</span>
              <h3 style={{ fontSize: '1.25rem', fontWeight: 800, color: colorA, marginTop: '2px' }}>{modelA}</h3>
            </div>
            {comparisonStats.winsA > comparisonStats.winsB && (
              <span className="badge badge-orange" style={{ padding: '6px 10px', fontSize: '0.82rem', fontWeight: 700 }}>
                <Trophy size={14} /> Overall Matchup Winner
              </span>
            )}
          </div>

          <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', marginBottom: '14px' }}>
            <span style={{ fontSize: '2rem', fontWeight: 900, color: 'var(--text-primary)' }}>{comparisonStats.winsA}</span>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>/ {totalMetrics} evaluated metric victories</span>
          </div>

          <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '12px' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Key Advantages:</span>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '8px' }}>
              {comparisonStats.strengthsA.length > 0 ? (
                comparisonStats.strengthsA.slice(0, 4).map((s, idx) => (
                  <div key={idx} style={{ fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-primary)' }}>
                    <CheckCircle2 size={14} color="#2cbb5d" />
                    <span><strong>{s.metric}:</strong> <span style={{ color: '#2cbb5d' }}>{s.detail}</span></span>
                  </div>
                ))
              ) : (
                <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>No unique metric advantages over challenger.</div>
              )}
            </div>
          </div>
        </div>

        {/* Win Distribution Meter & Head-to-Head Balance */}
        <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <div style={{ textAlign: 'center', marginBottom: '14px' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-secondary)', letterSpacing: '0.05em' }}>
              Metric Victory Share
            </span>
            <div style={{ fontSize: '1.4rem', fontWeight: 800, marginTop: '4px', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '12px' }}>
              <span style={{ color: colorA }}>{comparisonStats.winsA}</span>
              <span style={{ color: 'var(--text-secondary)', fontSize: '1rem', fontWeight: 400 }}>vs</span>
              <span style={{ color: colorB }}>{comparisonStats.winsB}</span>
            </div>
            {comparisonStats.ties > 0 && (
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>({comparisonStats.ties} metric ties)</span>
            )}
          </div>

          {/* Split Progress Bar */}
          <div style={{ width: '100%', height: '14px', background: '#222222', borderRadius: '8px', overflow: 'hidden', display: 'flex', border: '1px solid var(--border-color)' }}>
            <div style={{ width: `${pctA}%`, background: colorA, transition: 'width 0.4s ease' }} title={`${modelA}: ${pctA.toFixed(0)}% wins`}></div>
            <div style={{ width: `${pctB}%`, background: colorB, transition: 'width 0.4s ease' }} title={`${modelB}: ${pctB.toFixed(0)}% wins`}></div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '8px', fontSize: '0.75rem', fontWeight: 600 }}>
            <span style={{ color: colorA }}>{modelA}: {pctA.toFixed(1)}%</span>
            <span style={{ color: colorB }}>{modelB}: {pctB.toFixed(1)}%</span>
          </div>

          <div style={{ marginTop: '16px', background: 'var(--bg-subtle)', padding: '10px 14px', borderRadius: '6px', fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
            {comparisonStats.winsA > comparisonStats.winsB ? (
              <span><strong>{modelA}</strong> holds superior benchmark capability, winning <strong>{comparisonStats.winsA}</strong> out of {totalMetrics} automated and LLM-judge evaluated pillars.</span>
            ) : comparisonStats.winsB > comparisonStats.winsA ? (
              <span><strong>{modelB}</strong> holds superior benchmark capability, winning <strong>{comparisonStats.winsB}</strong> out of {totalMetrics} automated and LLM-judge evaluated pillars.</span>
            ) : (
              <span>Both models are evenly matched across the evaluated BFSI benchmark dimensions.</span>
            )}
          </div>
        </div>

        {/* Model B Profile Card */}
        <div 
          className="glass-panel" 
          style={{ 
            padding: '20px', 
            borderRight: `4px solid ${colorB}`,
            background: comparisonStats.winsB > comparisonStats.winsA ? 'linear-gradient(225deg, rgba(38,38,38,0.9) 0%, rgba(255,161,22,0.05) 100%)' : 'var(--bg-card)'
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600 }}>Model B Challenger</span>
              <h3 style={{ fontSize: '1.25rem', fontWeight: 800, color: colorB, marginTop: '2px' }}>{modelB}</h3>
            </div>
            {comparisonStats.winsB > comparisonStats.winsA && (
              <span className="badge badge-orange" style={{ padding: '6px 10px', fontSize: '0.82rem', fontWeight: 700 }}>
                <Trophy size={14} /> Overall Matchup Winner
              </span>
            )}
          </div>

          <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', marginBottom: '14px' }}>
            <span style={{ fontSize: '2rem', fontWeight: 900, color: 'var(--text-primary)' }}>{comparisonStats.winsB}</span>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>/ {totalMetrics} evaluated metric victories</span>
          </div>

          <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '12px' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Key Advantages:</span>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '8px' }}>
              {comparisonStats.strengthsB.length > 0 ? (
                comparisonStats.strengthsB.slice(0, 4).map((s, idx) => (
                  <div key={idx} style={{ fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-primary)' }}>
                    <CheckCircle2 size={14} color="#2cbb5d" />
                    <span><strong>{s.metric}:</strong> <span style={{ color: '#2cbb5d' }}>{s.detail}</span></span>
                  </div>
                ))
              ) : (
                <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>No unique metric advantages over reference.</div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* SECTION 3: VISUAL COMPARATIVE GRAPHS */}
      <div className="glass-panel" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px', marginBottom: '18px' }}>
          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <BarChart2 size={18} color="var(--accent-orange)" />
              Side-by-Side Visual Graph Comparison
            </h3>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
              Visualizing performance gaps across accuracy, LLM-judge quality, speed, safety, and radar trade-offs.
            </p>
          </div>

          {/* Graph Sub-tab Selectors */}
          <div style={{ display: 'flex', gap: '6px', background: 'var(--bg-subtle)', padding: '3px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
            <button
              onClick={() => setActiveChartTab('accuracy')}
              style={{
                padding: '6px 12px', borderRadius: '4px', border: 'none', cursor: 'pointer', fontSize: '0.78rem', fontWeight: 600,
                background: activeChartTab === 'accuracy' ? '#333333' : 'transparent',
                color: activeChartTab === 'accuracy' ? 'var(--accent-orange)' : 'var(--text-secondary)'
              }}
            >
              Accuracy (%)
            </button>
            <button
              onClick={() => setActiveChartTab('judge')}
              style={{
                padding: '6px 12px', borderRadius: '4px', border: 'none', cursor: 'pointer', fontSize: '0.78rem', fontWeight: 600,
                background: activeChartTab === 'judge' ? '#333333' : 'transparent',
                color: activeChartTab === 'judge' ? 'var(--accent-orange)' : 'var(--text-secondary)'
              }}
            >
              LLM Judge (/ 5.0)
            </button>
            <button
              onClick={() => setActiveChartTab('speed')}
              style={{
                padding: '6px 12px', borderRadius: '4px', border: 'none', cursor: 'pointer', fontSize: '0.78rem', fontWeight: 600,
                background: activeChartTab === 'speed' ? '#333333' : 'transparent',
                color: activeChartTab === 'speed' ? 'var(--accent-orange)' : 'var(--text-secondary)'
              }}
            >
              Latency & Tok/s
            </button>
            <button
              onClick={() => setActiveChartTab('safety')}
              style={{
                padding: '6px 12px', borderRadius: '4px', border: 'none', cursor: 'pointer', fontSize: '0.78rem', fontWeight: 600,
                background: activeChartTab === 'safety' ? '#333333' : 'transparent',
                color: activeChartTab === 'safety' ? 'var(--accent-orange)' : 'var(--text-secondary)'
              }}
            >
              Safety & Faithfulness
            </button>
            <button
              onClick={() => setActiveChartTab('radar')}
              style={{
                padding: '6px 12px', borderRadius: '4px', border: 'none', cursor: 'pointer', fontSize: '0.78rem', fontWeight: 600,
                background: activeChartTab === 'radar' ? '#333333' : 'transparent',
                color: activeChartTab === 'radar' ? 'var(--accent-orange)' : 'var(--text-secondary)'
              }}
            >
              🕸️ 6-Axis Radar
            </button>
          </div>
        </div>

        {/* Selected Chart Rendering */}
        <div style={{ width: '100%', height: '360px', marginTop: '10px' }}>
          {activeChartTab === 'radar' ? (
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="75%" data={radarData}>
                <PolarGrid stroke="#383838" />
                <PolarAngleAxis dataKey="dimension" stroke="#d4d4d4" fontSize={12} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#555555" fontSize={10} />
                <Radar name={modelA} dataKey={modelA} stroke={colorA} fill={colorA} fillOpacity={0.35} />
                <Radar name={modelB} dataKey={modelB} stroke={colorB} fill={colorB} fillOpacity={0.35} />
                <Tooltip contentStyle={{ background: '#262626', borderColor: '#3e3e3e', borderRadius: '6px' }} />
                <Legend wrapperStyle={{ fontSize: '13px', paddingTop: '10px' }} />
              </RadarChart>
            </ResponsiveContainer>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart 
                data={
                  activeChartTab === 'accuracy' ? accuracyChartData :
                  activeChartTab === 'judge' ? judgeChartData :
                  activeChartTab === 'speed' ? speedChartData :
                  safetyChartData
                } 
                margin={{ top: 20, right: 30, left: 10, bottom: 25 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#333333" />
                <XAxis 
                  dataKey="metric" 
                  stroke="var(--text-secondary)" 
                  fontSize={12} 
                  tickLine={false} 
                  interval={0}
                />
                <YAxis 
                  stroke="var(--text-secondary)" 
                  fontSize={12} 
                  tickLine={false}
                  tickFormatter={(v) => activeChartTab === 'accuracy' || activeChartTab === 'safety' ? `${v}%` : `${v}`}
                />
                <Tooltip 
                  contentStyle={{ background: '#262626', borderColor: '#3e3e3e', borderRadius: '6px' }}
                  formatter={(val: any, name: any) => [
                    activeChartTab === 'accuracy' || activeChartTab === 'safety' ? `${val}%` : `${val}`, 
                    name
                  ]}
                />
                <Legend wrapperStyle={{ fontSize: '13px', paddingTop: '10px' }} />
                <Bar dataKey={modelA} name={modelA} fill={colorA} radius={[4, 4, 0, 0]} />
                <Bar dataKey={modelB} name={modelB} fill={colorB} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* SECTION 4: ENTERPRISE COMPANY HEAD-TO-HEAD COMPARISON */}
      {companyComparisonData.length > 0 && (
        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', flexWrap: 'wrap', gap: '10px' }}>
            <div>
              <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <TrendingUp size={18} color="#2cbb5d" />
                Company Financial Disclosure Accuracy Head-to-Head
              </h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                Benchmarked accuracy across 6 domain company disclosures ({modelA} vs {modelB}).
              </p>
            </div>
            <div style={{ display: 'flex', gap: '14px', alignItems: 'center' }}>
              <span style={{ fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: colorA }}></span>
                {modelA}
              </span>
              <span style={{ fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: colorB }}></span>
                {modelB}
              </span>
            </div>
          </div>

          <div style={{ width: '100%', height: '280px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={companyComparisonData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333333" />
                <XAxis dataKey="company" stroke="var(--text-secondary)" fontSize={11} interval={0} />
                <YAxis stroke="var(--text-secondary)" fontSize={11} tickFormatter={(v) => `${v}%`} />
                <Tooltip 
                  contentStyle={{ background: '#262626', borderColor: '#3e3e3e', borderRadius: '6px' }}
                  formatter={(val: any, name: any) => [`${val}%`, name]}
                />
                <Bar dataKey={modelA} fill={colorA} radius={[4, 4, 0, 0]} />
                <Bar dataKey={modelB} fill={colorB} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* SECTION 5: COMPREHENSIVE NUMERICAL BENCHMARK TABLE */}
      <div className="glass-panel" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px', marginBottom: '16px' }}>
          <div>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Award size={18} color="var(--accent-orange)" />
              All 15 Evaluated Metrics Side-by-Side Numbers & Advantage Deltas
            </h3>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
              Granular numerical comparison with automated delta calculation and winner assignment.
            </p>
          </div>

          {/* Metric Category Filter Pills */}
          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
            {['All', 'Accuracy & Match', 'LLM Judge & Quality', 'Faithfulness & Safety', 'Performance & Speed'].map(cat => (
              <button
                key={cat}
                onClick={() => setActiveCategory(cat)}
                style={{
                  padding: '5px 10px',
                  borderRadius: '5px',
                  border: activeCategory === cat ? '1px solid var(--accent-orange)' : '1px solid var(--border-color)',
                  background: activeCategory === cat ? '#333333' : 'var(--bg-subtle)',
                  color: activeCategory === cat ? 'var(--accent-orange)' : 'var(--text-secondary)',
                  cursor: 'pointer',
                  fontSize: '0.76rem',
                  fontWeight: 600,
                  transition: 'all 0.15s ease'
                }}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        {/* Comparison Table */}
        <div style={{ overflowX: 'auto' }}>
          <table className="table-custom">
            <thead>
              <tr>
                <th style={{ width: '26%' }}>Benchmark Metric</th>
                <th style={{ width: '20%', color: colorA, background: 'rgba(56,189,248,0.04)' }}>
                  {modelA} (Value)
                </th>
                <th style={{ width: '18%', textAlign: 'center' }}>Head-to-Head Delta</th>
                <th style={{ width: '20%', color: colorB, background: 'rgba(255,161,22,0.04)' }}>
                  {modelB} (Value)
                </th>
                <th style={{ width: '16%', textAlign: 'center' }}>Category Winner</th>
              </tr>
            </thead>
            <tbody>
              {filteredMetricRows.map((row) => {
                const isWinnerA = row.winner === 'A';
                const isWinnerB = row.winner === 'B';
                const formattedA = row.def.format ? row.def.format(row.valA) : row.valA;
                const formattedB = row.def.format ? row.def.format(row.valB) : row.valB;

                return (
                  <tr key={row.key}>
                    <td>
                      <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <span style={{ fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: row.def.color }}></span>
                          {row.def.title}
                        </span>
                        <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                          {row.def.description}
                        </span>
                      </div>
                    </td>

                    {/* Model A Value */}
                    <td style={{ 
                      fontWeight: isWinnerA ? 800 : 500, 
                      color: isWinnerA ? colorA : 'var(--text-secondary)',
                      background: isWinnerA ? 'rgba(44, 187, 93, 0.06)' : 'transparent'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        {isWinnerA && <CheckCircle2 size={14} color="#2cbb5d" />}
                        <span>{formattedA}</span>
                      </div>
                    </td>

                    {/* Delta Badge */}
                    <td style={{ textAlign: 'center' }}>
                      {row.winner === 'tie' ? (
                        <span className="badge badge-neutral" style={{ fontSize: '0.72rem' }}>Tied (0.0)</span>
                      ) : (
                        <span 
                          className={isWinnerA ? 'badge badge-emerald' : 'badge badge-amber'}
                          style={{ fontSize: '0.75rem', fontWeight: 700 }}
                        >
                          {row.deltaText}
                        </span>
                      )}
                    </td>

                    {/* Model B Value */}
                    <td style={{ 
                      fontWeight: isWinnerB ? 800 : 500, 
                      color: isWinnerB ? colorB : 'var(--text-secondary)',
                      background: isWinnerB ? 'rgba(44, 187, 93, 0.06)' : 'transparent'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        {isWinnerB && <CheckCircle2 size={14} color="#2cbb5d" />}
                        <span>{formattedB}</span>
                      </div>
                    </td>

                    {/* Winner Badge */}
                    <td style={{ textAlign: 'center' }}>
                      {isWinnerA ? (
                        <span className="badge badge-cyan" style={{ fontWeight: 700 }}>
                          🏆 {modelA}
                        </span>
                      ) : isWinnerB ? (
                        <span className="badge badge-orange" style={{ fontWeight: 700 }}>
                          🏆 {modelB}
                        </span>
                      ) : (
                        <span className="badge badge-neutral">Tied</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* SECTION 6: INTERACTIVE SAMPLE PREDICTION HEAD-TO-HEAD */}
      <div className="glass-panel" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '14px', marginBottom: '16px' }}>
          <div>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Eye size={18} color="#38bdf8" />
              Micro Benchmark Inspection: Sample Responses Side-by-Side
            </h3>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
              Compare generated step-by-step reasoning and outputs of {modelA} vs {modelB} on real test items.
            </p>
          </div>

          {/* Sample Selector Controls */}
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
            <button className="btn-random" onClick={handlePickRandomSample} title="Load another random test sample">
              <Shuffle size={14} /> Random Test Sample
            </button>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Search size={14} color="var(--text-secondary)" />
              <select
                className="custom-select"
                value={sampleId}
                onChange={(e) => setSampleId(e.target.value)}
                style={{ fontWeight: 600, minWidth: '220px' }}
              >
                {sampleIdList.map(id => (
                  <option key={id} value={id}>{id}</option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {loadingSample ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
            Loading sample comparison for {sampleId}...
          </div>
        ) : sampleData ? (
          <div>
            {/* Prompt & Target Card */}
            <div style={{ background: 'var(--bg-subtle)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '16px', marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px' }}>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                    <span style={{ fontSize: '0.9rem', fontWeight: 700, color: '#38bdf8' }}>{sampleData.id}</span>
                    <span className="badge badge-neutral">{sampleData.metadata?.company || 'BFSI Company'}</span>
                    <span className="badge badge-neutral">{sampleData.metadata?.task_type || 'qa_numeric_program'}</span>
                  </div>
                  <div style={{ fontSize: '0.92rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                    Question: "{sampleData.input_text}"
                  </div>
                </div>

                <div style={{ background: '#1c1c1c', border: '1px solid #333', padding: '8px 16px', borderRadius: '6px', textAlign: 'right' }}>
                  <div style={{ fontSize: '0.68rem', color: '#2cbb5d', textTransform: 'uppercase', fontWeight: 700 }}>Ground Truth Target</div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--text-primary)' }}>{sampleData.target}</div>
                  {sampleData.metadata?.program && (
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', fontFamily: 'monospace', marginTop: '2px' }}>
                      {sampleData.metadata.program}
                    </div>
                  )}
                </div>
              </div>

              {sampleData.context && (
                <details style={{ marginTop: '10px' }}>
                  <summary style={{ cursor: 'pointer', fontSize: '0.78rem', color: 'var(--text-secondary)', fontWeight: 600 }}>
                    📄 View Reference Financial Disclosure Table Context
                  </summary>
                  <pre style={{ marginTop: '8px', background: '#141414', padding: '10px', borderRadius: '6px', fontSize: '0.78rem', color: '#b0b0b0', whiteSpace: 'pre-wrap', border: '1px solid var(--border-color)' }}>
                    {sampleData.context}
                  </pre>
                </details>
              )}
            </div>

            {/* Side-by-Side Model Responses */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '16px' }}>
              {/* Model A Output */}
              {sampleData.models[modelA] ? (
                <div 
                  style={{ 
                    background: 'var(--bg-subtle)', 
                    border: sampleData.models[modelA].exact_match === 1.0 ? '1px solid rgba(44, 187, 93, 0.4)' : '1px solid rgba(239, 71, 67, 0.3)', 
                    borderRadius: '8px', 
                    padding: '16px',
                    display: 'flex', flexDirection: 'column', justifyContent: 'space-between'
                  }}
                >
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                      <span style={{ fontWeight: 800, fontSize: '0.95rem', color: colorA }}>
                        Model A: {modelA}
                      </span>
                      {sampleData.models[modelA].exact_match === 1.0 ? (
                        <span className="badge badge-emerald"><CheckCircle2 size={12} /> Exact Match 1.0</span>
                      ) : (
                        <span className="badge badge-rose"><XCircle size={12} /> Mis-match 0.0</span>
                      )}
                    </div>

                    <div style={{ background: '#121212', padding: '12px', borderRadius: '6px', minHeight: '150px', fontSize: '0.82rem', color: '#d4d4d4', whiteSpace: 'pre-wrap', border: '1px solid #282828', marginBottom: '12px', lineHeight: '1.45' }}>
                      {sampleData.models[modelA].output_text || "No response recorded."}
                    </div>
                  </div>

                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', paddingTop: '10px', borderTop: '1px solid var(--border-color)' }}>
                    <span className="badge badge-amber">⭐ Judge: {sampleData.models[modelA].judge_score.toFixed(1)}/5</span>
                    <span className="badge badge-cyan">⏱️ {sampleData.models[modelA].latency_seconds.toFixed(2)}s</span>
                    <span className="badge badge-purple">🎯 BERT: {sampleData.models[modelA].bert_score.toFixed(2)}</span>
                    {sampleData.models[modelA].is_hallucinated && (
                      <span className="badge badge-rose">⚠️ Hallucinated</span>
                    )}
                  </div>
                </div>
              ) : (
                <div style={{ background: 'var(--bg-subtle)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '24px', textAlign: 'center', color: 'var(--text-secondary)' }}>
                  No output recorded for {modelA} on sample {sampleId}.
                </div>
              )}

              {/* Model B Output */}
              {sampleData.models[modelB] ? (
                <div 
                  style={{ 
                    background: 'var(--bg-subtle)', 
                    border: sampleData.models[modelB].exact_match === 1.0 ? '1px solid rgba(44, 187, 93, 0.4)' : '1px solid rgba(239, 71, 67, 0.3)', 
                    borderRadius: '8px', 
                    padding: '16px',
                    display: 'flex', flexDirection: 'column', justifyContent: 'space-between'
                  }}
                >
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                      <span style={{ fontWeight: 800, fontSize: '0.95rem', color: colorB }}>
                        Model B: {modelB}
                      </span>
                      {sampleData.models[modelB].exact_match === 1.0 ? (
                        <span className="badge badge-emerald"><CheckCircle2 size={12} /> Exact Match 1.0</span>
                      ) : (
                        <span className="badge badge-rose"><XCircle size={12} /> Mis-match 0.0</span>
                      )}
                    </div>

                    <div style={{ background: '#121212', padding: '12px', borderRadius: '6px', minHeight: '150px', fontSize: '0.82rem', color: '#d4d4d4', whiteSpace: 'pre-wrap', border: '1px solid #282828', marginBottom: '12px', lineHeight: '1.45' }}>
                      {sampleData.models[modelB].output_text || "No response recorded."}
                    </div>
                  </div>

                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', paddingTop: '10px', borderTop: '1px solid var(--border-color)' }}>
                    <span className="badge badge-amber">⭐ Judge: {sampleData.models[modelB].judge_score.toFixed(1)}/5</span>
                    <span className="badge badge-cyan">⏱️ {sampleData.models[modelB].latency_seconds.toFixed(2)}s</span>
                    <span className="badge badge-purple">🎯 BERT: {sampleData.models[modelB].bert_score.toFixed(2)}</span>
                    {sampleData.models[modelB].is_hallucinated && (
                      <span className="badge badge-rose">⚠️ Hallucinated</span>
                    )}
                  </div>
                </div>
              ) : (
                <div style={{ background: 'var(--bg-subtle)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '24px', textAlign: 'center', color: 'var(--text-secondary)' }}>
                  No output recorded for {modelB} on sample {sampleId}.
                </div>
              )}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
