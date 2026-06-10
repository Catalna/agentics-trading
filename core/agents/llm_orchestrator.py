import json
import logging
import concurrent.futures
from typing import Dict, Any

import config
from core.agents.ollama_client import OllamaClient
from core.agents import prompts

logger = logging.getLogger(__name__)

class LLMOrchestrator:
    def __init__(self):
        self.client = OllamaClient()
        
        # Determine the correct models from config or use defaults
        # To match PRD: DeepSeek R1, Qwen 2.5, Mistral
        # However, fallback to config definitions if user hasn't updated them
        models = config.OLLAMA_MODELS.keys()
        
        self.model_chief = "deepseek-r1:7b" if "deepseek-r1:7b" in models else "llama3.1:latest"
        self.model_tech  = "qwen2.5:7b"     if "qwen2.5:7b" in models else "qwen2.5:3b"
        self.model_sent  = "mistral:7b"     if "mistral:7b" in models else "llama3.2:3b"

    def _query_agent(self, model: str, prompt_template: str, input_data: str) -> Dict[str, Any]:
        """Helper to query a sub-agent."""
        prompt = f"Data:\n{input_data}\n\nTask:\nProvide your analysis in JSON format."
        res = self.client.generate(model, prompt, system=prompt_template)
        return res if res else {"error": "LLM failed to respond"}

    def run_sub_agents(self, market_data: str) -> Dict[str, Any]:
        """Runs the 5 technical sub-agents in parallel."""
        # Using the tech model (Qwen) for the technical sub-agents since it's the Tech Manager's domain
        # In a real heavy-weight setup, sub-agents might be smaller models.
        # We will use the tech manager model to simulate the sub-agents for now.
        
        tasks = {
            "Macro":      (self.model_tech, prompts.MACRO_AGENT_PROMPT),
            "Trend":      (self.model_tech, prompts.TREND_AGENT_PROMPT),
            "Scalp":      (self.model_tech, prompts.SCALP_AGENT_PROMPT),
            "Teknikal":   (self.model_tech, prompts.TEKNIKAL_AGENT_PROMPT),
            "Orderflow":  (self.model_tech, prompts.ORDERFLOW_AGENT_PROMPT),
            "Sentiment":  (self.model_sent, prompts.SENTIMEN_AGENT_PROMPT)
        }
        
        results = {}
        # Sequential execution — all models share one local GPU, parallel causes timeouts
        for name, (model, sys_prompt) in tasks.items():
            logger.debug(f"Running {name} agent...")
            results[name] = self._query_agent(model, sys_prompt, market_data)
                    
        return results

    def run_managers(self, sub_agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Runs Qwen and Mistral to compile the sub-agent reports."""
        tech_inputs = {k: v for k, v in sub_agent_results.items() if k != "Sentiment"}
        sent_inputs = sub_agent_results.get("Sentiment", {})
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_tech = executor.submit(
                self._query_agent, 
                self.model_tech, 
                prompts.TECHNICAL_MANAGER_PROMPT, 
                json.dumps(tech_inputs)
            )
            future_sent = executor.submit(
                self._query_agent, 
                self.model_sent, 
                prompts.SENTIMENT_MANAGER_PROMPT, 
                json.dumps(sent_inputs)
            )
            
            tech_report = future_tech.result()
            sent_report = future_sent.result()
            
        return {"Technical": tech_report, "Sentiment": sent_report}

    def get_final_decision(self, manager_reports: Dict[str, Any], ml_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Runs DeepSeek R1 to make the final decision."""
        combined_data = {
            "ManagerReports": manager_reports,
            "ML_Predictions": ml_predictions
        }
        
        final_decision = self._query_agent(
            self.model_chief, 
            prompts.CHIEF_SUPERVISOR_PROMPT, 
            json.dumps(combined_data)
        )
        
        return final_decision

    def orchestrate(self, market_data: str, ml_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point.
        1. Sub-agents (Parallel)
        2. Managers (Parallel)
        3. Chief Supervisor
        """
        logger.info("Starting LLM Multi-Agent Orchestration...")
        
        # 1. Sub-Agents
        logger.info("Running sub-agents...")
        sub_results = self.run_sub_agents(market_data)
        
        # 2. Managers
        logger.info("Running managers...")
        manager_results = self.run_managers(sub_results)
        
        # 3. Chief
        logger.info("Running chief supervisor...")
        final_decision = self.get_final_decision(manager_results, ml_predictions)
        
        logger.info(f"Final Decision Reached: {final_decision.get('final_decision', 'UNKNOWN')}")
        
        return {
            "sub_agents": sub_results,
            "managers": manager_results,
            "chief": final_decision
        }
