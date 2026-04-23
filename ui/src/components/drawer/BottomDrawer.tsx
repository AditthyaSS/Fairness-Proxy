import { motion, AnimatePresence } from 'framer-motion';
import { usePipelineStore } from '../../store/usePipelineStore';
import { ResponseJson } from './ResponseJson';
import { ShapChart } from './ShapChart';
import { TwinScoreChart } from './TwinScoreChart';
import { RLPanel } from './RLPanel';
import type { DrawerTab } from '../../types/api';

const TABS: { key: DrawerTab; label: string }[] = [
  { key: 'json', label: 'RESPONSE JSON' },
  { key: 'shap', label: 'XAI / SHAP' },
  { key: 'twins', label: 'TWIN SCORES' },
  { key: 'rl', label: 'RL LEARNING' },
];

export function BottomDrawer() {
  const drawerOpen = usePipelineStore((s) => s.drawerOpen);
  const drawerTab = usePipelineStore((s) => s.drawerTab);
  const setDrawerOpen = usePipelineStore((s) => s.setDrawerOpen);
  const setDrawerTab = usePipelineStore((s) => s.setDrawerTab);

  return (
    <div className="bottom-drawer-wrapper">
      <button
        className="drawer-toggle"
        onClick={() => setDrawerOpen(!drawerOpen)}
      >
        <span className="drawer-toggle-icon">{drawerOpen ? '▼' : '▲'}</span>
        <span>Pipeline Results</span>
      </button>

      <AnimatePresence>
        {drawerOpen && (
          <motion.div
            className="bottom-drawer"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 280, opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: 'easeInOut' }}
          >
            <div className="drawer-tabs">
              {TABS.map((tab) => (
                <button
                  key={tab.key}
                  className={`drawer-tab ${drawerTab === tab.key ? 'drawer-tab-active' : ''}`}
                  onClick={() => setDrawerTab(tab.key)}
                >
                  {tab.label}
                </button>
              ))}
            </div>
            <div className="drawer-content">
              {drawerTab === 'json' && <ResponseJson />}
              {drawerTab === 'shap' && <ShapChart />}
              {drawerTab === 'twins' && <TwinScoreChart />}
              {drawerTab === 'rl' && <RLPanel />}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
