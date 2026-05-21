"""
core/swarm_bridge.py — Adapter for Swarm Agents to Main EA.
"""

from typing import Dict, Any
import logging
import config

log = logging.getLogger("SwarmBridge")

class SwarmBridge:
    def __init__(self):
        self.enabled = getattr(config, 'SWARM_ENABLED', False)
        self.orchestrator = None
        
        if self.enabled:
            log.info("[Swarm] Swarm is ENABLED. Initializing Orchestrator and 14 Agents...")
            try:
                from crypto_swarm_trading.core.orchestrator import OrchestratorAgent
                self.orchestrator = OrchestratorAgent(config)
                self.orchestrator.initialize_agents()
                log.info("[Swarm] Orchestrator initialized successfully.")
            except Exception as e:
                log.error(f"[Swarm] Failed to initialize orchestrator: {e}")
                self.enabled = False
        else:
            log.info("[Swarm] Swarm is DISABLED in config.")

    def analyze(self, current_data: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the swarm per-tick and returns normalized consensus.
        """
        if not self.enabled or not self.orchestrator:
            return {'signal': 'HOLD', 'confidence': 0.0, 'agreement': 0.0}

        # Swarm requires 'normalized' and 'raw'
        swarm_input = {
            'normalized': features,
            'raw': current_data
        }

        try:
            consensus = self.orchestrator.run_tick(swarm_input)
            return {
                'signal': consensus.get('signal', 'HOLD'),
                'confidence': float(consensus.get('confidence', 0.0)),
                'agreement': float(consensus.get('agreement_level', 0.0))
            }
        except Exception as e:
            log.error(f"[Swarm] Execution failed: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'agreement': 0.0}
