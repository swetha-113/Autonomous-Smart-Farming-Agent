"""
agents/farming_agent.py - Autonomous Farming Agent using Claude as reasoning backbone
Implements a ReAct-style agent that uses tools for real-time farming decisions
"""

import json
import anthropic
from datetime import datetime
from loguru import logger
from typing import Optional

from config.settings import ANTHROPIC_API_KEY, AGENT_MODEL, MAX_ITERATIONS, DISEASE_CLASSES


SYSTEM_PROMPT = """You are an expert autonomous smart farming AI agent with deep knowledge in:
- Crop disease identification and treatment
- Soil science and fertility management
- Weather pattern analysis and prediction
- Precision agriculture and irrigation management
- Integrated pest management (IPM)
- Sustainable farming practices

You analyze multi-modal data (images, soil sensors, weather) and provide:
1. Accurate disease diagnoses with confidence levels
2. Specific treatment recommendations with dosages
3. Preventive measures and best practices
4. Yield optimization strategies
5. Economic impact assessments

Always structure your responses clearly with:
- Diagnosis/Analysis summary
- Severity assessment (Low/Medium/High/Critical)
- Immediate actions required
- Long-term recommendations
- Economic/yield impact estimate

Be specific, actionable, and data-driven. Cite the data provided to justify your recommendations."""


class FarmingAgent:
    """
    Autonomous farming agent powered by Claude.
    Uses a multi-tool ReAct framework for decision-making.
    """

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = AGENT_MODEL
        self.conversation_history = []
        self.analysis_cache = {}
        self.tools = self._define_tools()

    def _define_tools(self) -> list:
        return [
            {
                "name": "analyze_disease_severity",
                "description": "Analyze the severity of a detected crop disease and determine urgency of treatment",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "disease_name": {"type": "string", "description": "Name of the detected disease"},
                        "confidence": {"type": "number", "description": "Detection confidence 0-1"},
                        "plant_type": {"type": "string", "description": "Type of crop/plant"},
                        "affected_area_pct": {"type": "number", "description": "Estimated % of crop affected"}
                    },
                    "required": ["disease_name", "confidence", "plant_type"]
                }
            },
            {
                "name": "get_treatment_protocol",
                "description": "Get specific treatment protocol for a crop disease including pesticides and organic options",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "disease_name": {"type": "string"},
                        "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                        "organic_preferred": {"type": "boolean", "default": False}
                    },
                    "required": ["disease_name", "severity"]
                }
            },
            {
                "name": "assess_soil_health",
                "description": "Assess soil health from provided sensor readings and give recommendations",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "nitrogen": {"type": "number"},
                        "phosphorus": {"type": "number"},
                        "potassium": {"type": "number"},
                        "ph": {"type": "number"},
                        "moisture": {"type": "number"},
                        "health_score": {"type": "number"}
                    },
                    "required": ["nitrogen", "phosphorus", "potassium", "ph"]
                }
            },
            {
                "name": "generate_action_plan",
                "description": "Generate a comprehensive prioritized action plan for the farm based on all inputs",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "disease_findings": {"type": "object"},
                        "soil_findings": {"type": "object"},
                        "weather_forecast": {"type": "object"},
                        "farm_size_hectares": {"type": "number"},
                        "crop_type": {"type": "string"}
                    },
                    "required": ["crop_type"]
                }
            }
        ]

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute tool and return result as JSON string."""
        logger.info(f"Agent executing tool: {tool_name}")

        if tool_name == "analyze_disease_severity":
            disease = tool_input.get("disease_name", "Unknown")
            confidence = tool_input.get("confidence", 0.5)
            affected = tool_input.get("affected_area_pct", 20)

            if confidence > 0.85 and affected > 40:
                severity = "critical"
                urgency = "Immediate action required within 24 hours"
            elif confidence > 0.7 or affected > 25:
                severity = "high"
                urgency = "Treatment needed within 48-72 hours"
            elif confidence > 0.5:
                severity = "medium"
                urgency = "Monitor and treat within 1 week"
            else:
                severity = "low"
                urgency = "Continue monitoring, treatment may not be necessary"

            return json.dumps({
                "severity": severity,
                "urgency": urgency,
                "yield_loss_risk": f"{int(confidence * affected * 0.8)}%",
                "spread_risk": "High" if "blight" in disease.lower() or "mosaic" in disease.lower() else "Medium"
            })

        elif tool_name == "get_treatment_protocol":
            disease = tool_input.get("disease_name", "").lower()
            severity = tool_input.get("severity", "medium")
            organic = tool_input.get("organic_preferred", False)

            protocols = {
                "blight": {
                    "chemical": "Mancozeb 75% WP @ 2.5 g/L or Chlorothalonil 75% WP @ 2 g/L",
                    "organic": "Copper-based fungicide (Copper oxychloride 50% WP @ 3 g/L)",
                    "frequency": "Every 7-10 days",
                    "preventive": "Remove infected leaves, improve air circulation, avoid overhead irrigation"
                },
                "rust": {
                    "chemical": "Propiconazole 25% EC @ 1 mL/L or Tebuconazole 250 EC @ 1 mL/L",
                    "organic": "Sulfur-based fungicide @ 3 g/L",
                    "frequency": "Every 10-14 days",
                    "preventive": "Resistant varieties, proper spacing, crop rotation"
                },
                "leaf_spot": {
                    "chemical": "Carbendazim 50% WP @ 1 g/L or Iprodione 50% WP @ 1.5 g/L",
                    "organic": "Neem oil 1% solution + copper soap",
                    "frequency": "Every 7-14 days",
                    "preventive": "Sanitation, balanced fertilization, avoid leaf wetness"
                }
            }

            matched_protocol = None
            for key, protocol in protocols.items():
                if key in disease:
                    matched_protocol = protocol
                    break

            if not matched_protocol:
                matched_protocol = protocols["leaf_spot"]  # default

            treatment = matched_protocol["organic"] if organic else matched_protocol["chemical"]
            return json.dumps({
                "primary_treatment": treatment,
                "frequency": matched_protocol["frequency"],
                "preventive_measures": matched_protocol["preventive"],
                "notes": f"Apply early morning or evening. Severity: {severity}. Wear PPE during application."
            })

        elif tool_name == "assess_soil_health":
            N = tool_input.get("nitrogen", 60)
            P = tool_input.get("phosphorus", 40)
            K = tool_input.get("potassium", 40)
            pH = tool_input.get("ph", 6.5)
            health_score = tool_input.get("health_score", 70)

            issues = []
            if N < 40: issues.append(f"Nitrogen deficiency (N={N}, optimal: 40-80 kg/ha)")
            if P < 30: issues.append(f"Phosphorus deficiency (P={P}, optimal: 30-60 kg/ha)")
            if K < 40: issues.append(f"Potassium deficiency (K={K}, optimal: 40-80 kg/ha)")
            if pH < 5.5: issues.append(f"Soil too acidic (pH={pH}, apply lime)")
            if pH > 7.8: issues.append(f"Soil too alkaline (pH={pH}, apply sulfur)")

            return json.dumps({
                "overall_health": "Good" if health_score >= 70 else "Fair" if health_score >= 50 else "Poor",
                "score": health_score,
                "issues": issues,
                "fertilizer_needed": len(issues) > 0,
                "immediate_actions": issues[:2] if issues else ["Maintain current nutrient levels"]
            })

        elif tool_name == "generate_action_plan":
            disease_data = tool_input.get("disease_findings", {})
            soil_data = tool_input.get("soil_findings", {})
            weather_data = tool_input.get("weather_forecast", {})
            farm_size = tool_input.get("farm_size_hectares", 5)
            crop = tool_input.get("crop_type", "general")

            return json.dumps({
                "priority_1": "Address any active disease outbreaks immediately",
                "priority_2": "Correct soil nutrient deficiencies before next planting",
                "priority_3": "Set up irrigation schedule based on weather forecast",
                "priority_4": "Implement preventive spray schedule",
                "priority_5": "Monitor and log field conditions weekly",
                "estimated_cost": f"₹{int(farm_size * 3500)}-{int(farm_size * 5000)}/season",
                "expected_yield_improvement": "15-25% with full implementation",
                "timeline": "Weeks 1-2: Disease control | Weeks 3-4: Soil treatment | Ongoing: Monitoring"
            })

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def analyze(self, query: str, context: dict = None) -> dict:
        """
        Run the autonomous agent on a farming query with optional context data.
        Returns structured analysis and recommendations.
        """
        context = context or {}
        context_str = ""
        if context:
            context_str = "\n\nAvailable sensor/analysis data:\n"
            for key, val in context.items():
                context_str += f"- {key}: {json.dumps(val, default=str)[:300]}\n"

        full_query = query + context_str
        self.conversation_history.append({"role": "user", "content": full_query})

        iteration = 0
        tool_results = []

        while iteration < MAX_ITERATIONS:
            iteration += 1
            logger.info(f"Agent iteration {iteration}")

            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=self.tools,
                messages=self.conversation_history
            )

            # If model wants to use tools
            if response.stop_reason == "tool_use":
                tool_calls = [block for block in response.content if block.type == "tool_use"]
                tool_results_for_history = []

                for tool_call in tool_calls:
                    result = self._execute_tool(tool_call.name, tool_call.input)
                    tool_results.append({"tool": tool_call.name, "result": json.loads(result)})
                    tool_results_for_history.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": result
                    })

                # Add assistant response and tool results to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content
                })
                self.conversation_history.append({
                    "role": "user",
                    "content": tool_results_for_history
                })

            else:
                # Final response
                final_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        final_text += block.text

                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_text
                })

                return {
                    "success": True,
                    "response": final_text,
                    "tool_calls": tool_results,
                    "iterations": iteration,
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model
                }

        return {
            "success": False,
            "response": "Agent reached maximum iterations without a final response.",
            "tool_calls": tool_results,
            "iterations": iteration
        }

    def reset_conversation(self):
        """Clear conversation history for a new session."""
        self.conversation_history = []

    def get_quick_recommendation(self, disease_result: dict, soil_result: dict) -> str:
        """Generate a quick recommendation without full agent loop."""
        if not ANTHROPIC_API_KEY:
            return self._fallback_recommendation(disease_result, soil_result)

        try:
            prompt = f"""Based on these farm readings, give a concise action plan (max 150 words):
Disease Detection: {json.dumps(disease_result, default=str)[:400]}
Soil Analysis: {json.dumps(soil_result, default=str)[:400]}

Format: 3 bullet points of immediate actions."""

            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Quick recommendation failed: {e}")
            return self._fallback_recommendation(disease_result, soil_result)

    def _fallback_recommendation(self, disease_result: dict, soil_result: dict) -> str:
        lines = ["Based on your farm data:"]
        if disease_result.get("success") and not disease_result.get("is_healthy"):
            disease = disease_result.get("condition", "disease")
            severity = disease_result.get("severity", "Medium")
            lines.append(f"• **Disease Alert**: {disease} detected ({severity} severity). Apply appropriate fungicide/treatment immediately.")
        else:
            lines.append("• **Plant Health**: No disease detected. Continue routine monitoring.")

        if soil_result.get("alerts"):
            alert = soil_result["alerts"][0]
            lines.append(f"• **Soil Issue**: {alert['message']}. {alert['action']}")
        else:
            lines.append(f"• **Soil Health**: {soil_result.get('health_grade', 'Good')} condition. Maintain current practices.")

        crop = soil_result.get("crop_recommendation", [{}])[0].get("crop", "crops") if soil_result.get("crop_recommendation") else "crops"
        lines.append(f"• **Recommendation**: Conditions are best suited for {crop}. Monitor weekly and adjust irrigation based on weather forecast.")

        return "\n".join(lines)
