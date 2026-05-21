"""
core/orchestrator.py — Supreme Orchestrator with Swarm Consensus Engine.

Features:
  - Initialize and manage all 20 swarm agents
  - Weighted voting consensus mechanism
  - Dynamic agent weight adjustment based on rolling accuracy
  - Auto-disable agents with repeated errors (> threshold)
  - Auto-re-enable after cooldown period
  - Emergency shutdown capability
  - Real-time performance monitoring per-agent
  - Fallback handling when agents are unavailable
"""

import time
import logging
from typing import Dict, List, Any
from crypto_swarm_trading.core.base_agent import AgentState
from crypto_swarm_trading.core.message_bus import MessageBus
from crypto_swarm_trading.core.database import DBManager
import math

class OrchestratorAgent:
    def __init__(self, config):
        self.config = config
        self.message_bus = MessageBus(getattr(config, 'MAX_QUEUE_SIZE', 5000))
        self.db = DBManager(getattr(config, 'DB_PATH', './data/swarm_trading.db'))
        self.logger = logging.getLogger("Orchestrator")
        self.logger.setLevel(getattr(config, 'LOG_LEVEL', 'INFO'))
        
        self.agents: Dict[str, Any] = {}
        self._agent_weights = dict(getattr(config, 'AGENT_WEIGHTS', {}))
        self._disabled_agents: Dict[str, float] = {}
        self._running = False
        self._cooldown_seconds = getattr(config, 'AGENT_COOLDOWN_SECONDS', 120)
        self._error_threshold = getattr(config, 'AGENT_ERROR_THRESHOLD', 5)

    def initialize_agents(self):
        """Instantiate all agents, connect message bus, and start them."""
        from crypto_swarm_trading.agents.analysis.ml_prediction_agent import MLPredictionAgent
        from crypto_swarm_trading.agents.analysis.sentiment_analysis_agent import SentimentAnalysisAgent
        from crypto_swarm_trading.agents.analysis.technical_analysis_agent import TechnicalAnalysisAgent
        from crypto_swarm_trading.agents.analysis.volatility_regime_agent import VolatilityRegimeAgent

        from crypto_swarm_trading.agents.strategy.breakout_agent import BreakoutAgent
        from crypto_swarm_trading.agents.strategy.mean_reversion_agent import MeanReversionAgent
        from crypto_swarm_trading.agents.strategy.scalping_strategy_agent import ScalpingStrategyAgent
        from crypto_swarm_trading.agents.strategy.swing_strategy_agent import SwingStrategyAgent
        
        self.agents['ml_prediction'] = MLPredictionAgent(self.config)
        self.agents['sentiment'] = SentimentAnalysisAgent(self.config)
        self.agents['technical'] = TechnicalAnalysisAgent(self.config)
        self.agents['volatility'] = VolatilityRegimeAgent(self.config)
        
        self.agents['breakout'] = BreakoutAgent(self.config)
        self.agents['mean_reversion'] = MeanReversionAgent(self.config)
        self.agents['scalping'] = ScalpingStrategyAgent(self.config)
        self.agents['swing'] = SwingStrategyAgent(self.config)
        
        for name, agent in self.agents.items():
            agent.message_bus = self.message_bus
            agent.start()

    def run_tick(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous pipeline per-tick for the EA."""
        # 1. Run Analysis Layer
        analysis_results = {}
        for name in ['technical', 'ml_prediction', 'sentiment', 'volatility']:
            if name in self.agents and name not in self._disabled_agents:
                try:
                    res = self.agents[name].process(data)
                    if res:
                        analysis_results[name] = res
                except Exception as e:
                    self.logger.error(f"Analysis agent {name} failed: {e}")
                    
        # Merge analysis results into the context for strategies
        context = dict(data)
        context['analysis'] = analysis_results

        # 2. Run Strategy Layer
        strategy_outputs = {}
        for name in ['scalping', 'swing', 'mean_reversion', 'breakout']:
            if name in self.agents and name not in self._disabled_agents:
                try:
                    res = self.agents[name].process(context)
                    if res and 'signal' in res:
                        strategy_outputs[name] = res
                except Exception as e:
                    self.logger.error(f"Strategy agent {name} failed: {e}")

        # 3. Build Consensus
        consensus = self.build_consensus(strategy_outputs)
        
        # Periodic health check
        now = time.time()
        if not hasattr(self, '_last_health_check') or now - self._last_health_check > 30:
            self._check_agent_health()
            self._maybe_reenable_agents()
            self._last_health_check = now

        return consensus

    def run(self):
        self._running = True
        self.message_bus.start()
        self.logger.info("Swarm Orchestrator started.")
        
        last_weight_update = time.time()
        last_health_check = time.time()

        while self._running:
            try:
                # 1. Health checks & re-enable
                if time.time() - last_health_check > 30:
                    self._check_agent_health()
                    self._maybe_reenable_agents()
                    last_health_check = time.time()

                # 2. Weight updates (e.g. hourly)
                if time.time() - last_weight_update > 3600:
                    self._update_agent_weights()
                    last_weight_update = time.time()

                # 3. Main Loop Logic (Pseudo)
                # Fetch Data -> Analyze -> Strategize -> Risk -> Execute
                # This logic depends heavily on how the specific layers trigger.
                # In a purely message-driven system, agents react to messages.
                # The orchestrator might just oversee or send the initial "TICK" message.
                
                time.sleep(3) # Main loop delay

            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received.")
                self.emergency_shutdown()
                break
            except Exception as e:
                self.logger.error(f"Orchestrator encountered error: {e}", exc_info=True)
                time.sleep(5)

    def build_consensus(self, strategy_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Weighted voting consensus mechanism."""
        votes = {'LONG': 0.0, 'SHORT': 0.0, 'HOLD': 0.0}
        
        for agent_name, output in strategy_outputs.items():
            signal = output.get('signal', 'HOLD')
            confidence = output.get('confidence', 0.0)
            weight = self._agent_weights.get(agent_name, 1.0)
            
            if signal in votes:
                votes[signal] += confidence * weight
                
        total_votes = sum(votes.values())
        if total_votes == 0:
            return {'signal': 'HOLD', 'confidence': 0.0, 'agreement_level': 0.0, 'vote_distribution': votes}
            
        normalized_votes = {k: v / total_votes for k, v in votes.items()}
        winner = max(normalized_votes, key=normalized_votes.get)
        
        # Calculate entropy for agreement level
        entropy = -sum(p * math.log2(p) for p in normalized_votes.values() if p > 0)
        max_entropy = math.log2(len(votes))
        agreement = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        
        return {
            'signal': winner,
            'confidence': normalized_votes[winner],
            'agreement_level': agreement,
            'vote_distribution': normalized_votes
        }

    def _update_agent_weights(self):
        """Update weights based on rolling accuracy from DB."""
        self.logger.info("Updating agent weights (placeholder).")
        pass

    def _check_agent_health(self):
        """Check all agents, disable if error count > threshold."""
        for name, agent in self.agents.items():
            if name in self._disabled_agents:
                continue
            health = agent.health_check()
            if health['consecutive_errors'] >= self._error_threshold:
                self._disable_agent(name)
            
            # Upsert health to DB
            self.db.upsert_agent_health(
                name, health['state'], health['uptime_seconds'], 
                health['consecutive_errors'], agent.performance.success_count, 
                health['avg_exec_time_ms']
            )

    def _disable_agent(self, name: str):
        self.logger.warning(f"Disabling agent {name} due to repeated errors.")
        self._disabled_agents[name] = time.time()
        if name in self.agents:
            self.agents[name].stop()

    def _maybe_reenable_agents(self):
        current_time = time.time()
        for name, disabled_at in list(self._disabled_agents.items()):
            if current_time - disabled_at > self._cooldown_seconds:
                self.logger.info(f"Re-enabling agent {name} after cooldown.")
                del self._disabled_agents[name]
                if name in self.agents:
                    self.agents[name].start()

    def _get_active_agents(self, layer_agents: List[str]) -> List[str]:
        return [name for name in layer_agents if name not in self._disabled_agents]

    def emergency_shutdown(self):
        self.logger.critical("Initiating emergency shutdown!")
        self._running = False
        for agent in self.agents.values():
            agent.stop()
        self.message_bus.stop()
        self.logger.info("Shutdown complete.")

    def get_system_status(self) -> Dict[str, Any]:
        return {
            'disabled_agents': list(self._disabled_agents.keys()),
            'weights': self._agent_weights,
            'agent_status': {name: agent.state.name for name, agent in self.agents.items()}
        }
