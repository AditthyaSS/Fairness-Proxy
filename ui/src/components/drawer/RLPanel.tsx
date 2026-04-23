// src/components/drawer/RLPanel.tsx
import { usePipelineStore } from '../../store/usePipelineStore'
import {
    LineChart, Line, XAxis, YAxis, Tooltip,
    ReferenceLine, ResponsiveContainer, Legend,
    AreaChart, Area
} from 'recharts'
import { motion, AnimatePresence } from 'framer-motion'

const PHASE_COLORS: Record<string, string> = {
    heuristic_fallback: '#4a5568',
    ppo_warming:        '#7c3aed',
    trained_ppo:        '#00d084',
}

const PHASE_LABELS: Record<string, string> = {
    heuristic_fallback: 'Phase 1 — Heuristic Fallback',
    ppo_warming:        'Phase 2 — PPO Warming',
    trained_ppo:        'Phase 3 — Trained PPO',
}

export function RLPanel() {
    const rlStats = usePipelineStore((s) => s.rlStats)
    const accuracyGain = usePipelineStore((s) => s.accuracyGain)

    if (!rlStats) {
        return (
            <div className="drawer-empty">
                Run the pipeline with UCI Adult dataset to see learning metrics
            </div>
        )
    }

    // Build dual-line chart data
    const origCurve = rlStats.original_accuracy_curve ?? []
    const corrCurve = rlStats.corrected_accuracy_curve ?? []
    const gainCurve = rlStats.accuracy_gain_curve ?? []

    const accuracyData = origCurve.map((orig: number, i: number) => ({
        episode: i + 1,
        original:  parseFloat((orig * 100).toFixed(2)),
        proxy:     parseFloat(((corrCurve[i] ?? orig) * 100).toFixed(2)),
        gain:      parseFloat(((gainCurve[i] ?? 0) * 100).toFixed(2)),
    }))

    const rewardData = (rlStats.reward_curve ?? []).map((r: number, i: number) => ({
        i,
        reward: r,
    }))

    const improvement = (rlStats.cumulative_improvement ?? 0) * 100
    const isImproving = improvement > 0

    const agentPhase = rlStats.agent_phase ?? 'heuristic_fallback'
    const phaseProgress = rlStats.phase_progress ?? 0
    const totalLabeled = rlStats.total_labeled_episodes ?? 0

    return (
        <div className="rl-panel">

            {/* ── TOP STAT ROW ── */}
            <div className="rl-stat-row">

                <div className="rl-stat">
                    <span className="rl-stat-label">Labeled episodes</span>
                    <span className="rl-stat-value">
                        {totalLabeled}
                    </span>
                </div>

                <div className="rl-stat">
                    <span className="rl-stat-label">Avg reward (last 50)</span>
                    <span
                        className="rl-stat-value"
                        style={{ color: rlStats.avg_reward_last50 > -0.1 ? '#00d084' : '#ff4757' }}
                    >
                        {rlStats.avg_reward_last50.toFixed(4)}
                    </span>
                </div>

                {/* THE END CARD NUMBER */}
                <div className="rl-stat rl-stat-highlight">
                    <span className="rl-stat-label">Cumulative accuracy gain</span>
                    <AnimatePresence mode="wait">
                        <motion.span
                            key={improvement.toFixed(2)}
                            className="rl-stat-value rl-stat-big"
                            style={{ color: isImproving ? '#00d084' : '#ff4757' }}
                            initial={{ scale: 0.8, opacity: 0 }}
                            animate={{ scale: 1,   opacity: 1 }}
                            exit={{    scale: 1.2, opacity: 0 }}
                            transition={{ type: 'spring', stiffness: 300 }}
                        >
                            {isImproving ? '+' : ''}{improvement.toFixed(2)}%
                        </motion.span>
                    </AnimatePresence>
                </div>

                <div className="rl-stat">
                    <span className="rl-stat-label">Avg L_CF</span>
                    <span className="rl-stat-value">
                        {rlStats.avg_cf_variance_last50.toFixed(4)}
                    </span>
                </div>

            </div>

            {/* ── AGENT PHASE PROGRESS BAR ── */}
            <div className="rl-phase-row">
                <div className="rl-phase-labels">
                    {(['heuristic_fallback', 'ppo_warming', 'trained_ppo'] as const).map(phase => (
                        <span
                            key={phase}
                            className="rl-phase-label"
                            style={{
                                color: agentPhase === phase
                                    ? PHASE_COLORS[phase]
                                    : '#2e3440',
                                fontWeight: agentPhase === phase ? 700 : 400,
                            }}
                        >
                            {phase === 'heuristic_fallback' ? 'Heuristic' :
                             phase === 'ppo_warming'        ? 'Warming'   : 'Trained PPO'}
                        </span>
                    ))}
                </div>

                <div className="rl-phase-track">
                    <motion.div
                        className="rl-phase-fill"
                        style={{ background: PHASE_COLORS[agentPhase] }}
                        animate={{
                            width: agentPhase === 'heuristic_fallback'
                                ? `${phaseProgress * 33.3}%`
                                : agentPhase === 'ppo_warming'
                                ? `${33.3 + phaseProgress * 33.3}%`
                                : '100%'
                        }}
                        transition={{ duration: 0.8, ease: 'easeOut' }}
                    />
                </div>

                <div className="rl-phase-current">
                    {PHASE_LABELS[agentPhase]}
                    {agentPhase !== 'trained_ppo' && (
                        <span className="rl-phase-next">
                            {agentPhase === 'heuristic_fallback'
                                ? ` · ${Math.max(0, 10 - totalLabeled)} episodes to Phase 2`
                                : ` · ${Math.max(0, 50 - totalLabeled)} episodes to Phase 3`}
                        </span>
                    )}
                </div>
            </div>

            {/* ── DUAL LINE ACCURACY CHART — THE END CARD ── */}
            <div className="rl-chart-label">
                Accuracy over episodes
                <span className="rl-chart-sublabel">
                    — Original AI vs Fairness Proxy (ground truth from UCI labels)
                </span>
            </div>

            {accuracyData.length > 1 ? (
                <ResponsiveContainer width="100%" height={160}>
                    <LineChart data={accuracyData} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                        <XAxis
                            dataKey="episode"
                            tick={{ fill: '#4a5568', fontSize: 10 }}
                        />
                        <YAxis
                            domain={['auto', 'auto']}
                            tick={{ fill: '#4a5568', fontSize: 10 }}
                            tickFormatter={(v: number) => `${v.toFixed(0)}%`}
                            width={42}
                        />
                        <Tooltip
                            contentStyle={{
                                background: '#111318',
                                border: '1px solid #242830',
                                borderRadius: 6,
                                fontSize: 11,
                            }}
                            formatter={(v: number, name: string) => [
                                `${v.toFixed(2)}%`,
                                name === 'original' ? 'Original AI' : 'Fairness Proxy'
                            ]}
                        />
                        <Legend
                            formatter={(v: string) => v === 'original' ? 'Original AI' : 'Fairness Proxy'}
                            wrapperStyle={{ fontSize: 11, color: '#4a5568' }}
                        />
                        {/* Original AI — flat gray line */}
                        <Line
                            type="monotone"
                            dataKey="original"
                            stroke="#4a5568"
                            strokeWidth={2}
                            dot={false}
                            strokeDasharray="4 2"
                            animationDuration={600}
                        />
                        {/* Fairness Proxy — climbing purple line */}
                        <Line
                            type="monotone"
                            dataKey="proxy"
                            stroke="#7c3aed"
                            strokeWidth={2.5}
                            dot={false}
                            animationDuration={600}
                        />
                    </LineChart>
                </ResponsiveContainer>
            ) : (
                <div className="rl-chart-empty">
                    Run {Math.max(0, 2 - accuracyData.length)} more UCI Adult episodes
                    to see the accuracy chart
                </div>
            )}

            {/* ── PER-RUN ACCURACY GAIN INDICATOR ── */}
            <AnimatePresence>
                {accuracyGain !== null && (
                    <motion.div
                        className="rl-gain-banner"
                        style={{
                            borderColor: accuracyGain > 0 ? '#00d084' : '#ff4757',
                            background:  accuracyGain > 0 ? '#00d08411' : '#ff475711',
                        }}
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{    opacity: 0, y: -8 }}
                    >
                        <span style={{ color: accuracyGain > 0 ? '#00d084' : '#ff4757' }}>
                            {accuracyGain > 0 ? '▲' : '▼'} This decision:
                            {accuracyGain > 0 ? ' +' : ' '}{(accuracyGain * 100).toFixed(2)}% accuracy
                        </span>
                        <span className="rl-gain-explanation">
                            {accuracyGain > 0
                                ? 'Mitigation moved prediction closer to true outcome'
                                : 'No improvement this episode — agent learning from this'}
                        </span>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* ── REWARD CURVE ── */}
            <div className="rl-chart-label" style={{ marginTop: 16 }}>
                Reward curve (last 50 episodes)
            </div>
            {rewardData.length > 0 ? (
                <ResponsiveContainer width="100%" height={80}>
                    <AreaChart data={rewardData}>
                        <XAxis dataKey="i" hide />
                        <YAxis domain={['auto', 'auto']} hide />
                        <Tooltip
                            contentStyle={{
                                background: '#111318',
                                border: '1px solid #242830',
                                borderRadius: 6,
                                fontSize: 11,
                            }}
                            formatter={(v: number) => [v.toFixed(4), 'reward']}
                        />
                        <ReferenceLine y={0} stroke="#2e3440" strokeDasharray="3 3" />
                        <defs>
                            <linearGradient id="rewardGrad" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%"  stopColor="#7c3aed" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="#7c3aed" stopOpacity={0}   />
                            </linearGradient>
                        </defs>
                        <Area
                            type="monotone"
                            dataKey="reward"
                            stroke="#7c3aed"
                            strokeWidth={2}
                            fill="url(#rewardGrad)"
                            dot={false}
                            animationDuration={300}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            ) : (
                <div className="rl-chart-empty" style={{ height: 80 }}>
                    No reward data yet
                </div>
            )}

            {/* ── ACTION DISTRIBUTION ── */}
            <div className="rl-chart-label" style={{ marginTop: 16 }}>
                Action distribution (last 50)
            </div>
            <div className="rl-action-bars">
                {Object.entries(rlStats.action_distribution_last50).map(([action, count]) => {
                    const total = Object.values(rlStats.action_distribution_last50)
                        .reduce((a: number, b: number) => a + b, 0)
                    const pct = total > 0 ? (count / total) * 100 : 0
                    const color = action === 'PASS'
                        ? '#00d084' : action === 'MITIGATE'
                        ? '#f5a623' : '#ff4757'
                    return (
                        <div key={action} className="rl-action-bar-row">
                            <span className="rl-action-bar-label" style={{ color }}>
                                {action}
                            </span>
                            <div className="rl-action-bar-bg">
                                <motion.div
                                    className="rl-action-bar-fill"
                                    animate={{ width: `${pct}%` }}
                                    style={{
                                        background: color + '44',
                                        borderRight: `2px solid ${color}`
                                    }}
                                    transition={{ duration: 0.6 }}
                                />
                            </div>
                            <span className="rl-action-bar-pct">{pct.toFixed(0)}%</span>
                        </div>
                    )
                })}
            </div>

            {/* ── BOTTOM NOTE ── */}
            <div className="rl-note">
                Ground truth from UCI Adult <code>true_label</code>.
                Reward R = −(α·L_CF + β·Σ|wₚ|) + γ·accuracy_gain, γ=1.5.
                Agent transitions: heuristic → PPO warming (10 eps) → trained PPO (50 eps).
            </div>

        </div>
    )
}
