"""
app/api/routes/rl_status.py
=============================
Exposes real-time RL agent learning stats + accuracy curves.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.session import get_db, RLEpisode
from app.rl.ppo_agent import get_rl_agent

router = APIRouter()


@router.get("/stats")
async def get_rl_stats(db: AsyncSession = Depends(get_db)):
    """
    Returns live RL learning statistics:
    - Total episodes seen
    - Running average reward (last 50)
    - Verdict distribution over time
    - Agent mode (heuristic vs trained)
    - Accuracy curves (original vs corrected)
    """
    total = await db.scalar(select(func.count()).select_from(RLEpisode))

    recent = await db.execute(
        select(RLEpisode.reward, RLEpisode.action, RLEpisode.cf_variance,
               RLEpisode.protected_shap_sum)
        .order_by(RLEpisode.id.desc())
        .limit(50)
    )
    episodes = recent.fetchall()

    avg_reward = float(sum(e.reward for e in episodes) / len(episodes)) if episodes else 0.0
    avg_cf = float(sum(e.cf_variance for e in episodes) / len(episodes)) if episodes else 0.0
    avg_shap = float(sum(e.protected_shap_sum for e in episodes) / len(episodes)) if episodes else 0.0

    action_counts = {"PASS": 0, "MITIGATE": 0, "BLOCK": 0}
    for e in episodes:
        if e.action in action_counts:
            action_counts[e.action] += 1

    reward_curve = [round(e.reward, 4) for e in reversed(episodes)]

    agent = get_rl_agent()
    agent_mode = "trained_ppo" if agent.policy is not None else "heuristic_fallback"

    # ── Accuracy curves (ground truth) ────────────────────────────────
    accuracy_result = await db.execute(
        select(
            RLEpisode.id,
            RLEpisode.original_error,
            RLEpisode.corrected_error,
            RLEpisode.accuracy_gain,
        )
        .where(RLEpisode.original_error.isnot(None))
        .order_by(RLEpisode.id.asc())
        .limit(200)
    )
    acc_rows = accuracy_result.fetchall()

    original_accuracy_curve = [round(1 - r.original_error, 4) for r in acc_rows]
    corrected_accuracy_curve = [round(1 - r.corrected_error, 4) for r in acc_rows]
    accuracy_gain_curve = [round(r.accuracy_gain, 4) for r in acc_rows]

    # Compute cumulative improvement
    cum_result = await db.execute(
        select(func.avg(RLEpisode.accuracy_gain))
        .where(RLEpisode.accuracy_gain.isnot(None))
    )
    cumulative_improvement = cum_result.scalar()
    cumulative_improvement = round(float(cumulative_improvement), 5) if cumulative_improvement else 0.0

    # Agent phase progression
    total_with_labels = len(acc_rows)
    if total_with_labels < 10:
        agent_phase = "heuristic_fallback"
        phase_progress = total_with_labels / 10
    elif total_with_labels < 50:
        agent_phase = "ppo_warming"
        phase_progress = (total_with_labels - 10) / 40
    else:
        agent_phase = "trained_ppo"
        phase_progress = 1.0

    return {
        "total_episodes": total or 0,
        "avg_reward_last50": round(avg_reward, 4),
        "avg_cf_variance_last50": round(avg_cf, 4),
        "avg_protected_shap_last50": round(avg_shap, 4),
        "action_distribution_last50": action_counts,
        "reward_curve": reward_curve,
        "agent_mode": agent_mode,
        # Accuracy tracking
        "original_accuracy_curve": original_accuracy_curve,
        "corrected_accuracy_curve": corrected_accuracy_curve,
        "accuracy_gain_curve": accuracy_gain_curve,
        "cumulative_improvement": cumulative_improvement,
        "total_labeled_episodes": total_with_labels,
        "agent_phase": agent_phase,
        "phase_progress": round(phase_progress, 3),
    }


@router.get("/replay-buffer")
async def get_replay_buffer(limit: int = 100, db: AsyncSession = Depends(get_db)):
    """Returns recent RL episodes for training replay visualization."""
    result = await db.execute(
        select(RLEpisode).order_by(RLEpisode.id.desc()).limit(limit)
    )
    episodes = result.scalars().all()
    return [
        {
            "id": e.id,
            "request_id": e.request_id,
            "action": e.action,
            "reward": round(e.reward, 4),
            "cf_variance": round(e.cf_variance, 4),
            "protected_shap_sum": round(e.protected_shap_sum, 4),
            "accuracy_gain": round(e.accuracy_gain, 4) if e.accuracy_gain is not None else None,
        }
        for e in reversed(episodes)
    ]
