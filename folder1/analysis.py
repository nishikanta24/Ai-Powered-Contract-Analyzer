import json
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from prompts.templates import ANALYSIS_PROMPT

load_dotenv()

class ContractAnalysisChain:
    def __init__(self):
        self.llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0,
            max_tokens=8000
        )
        # Store the prompt template
        self.prompt = ANALYSIS_PROMPT

    def analyze_contract(self, contract_text: str) -> dict:
        """
        Analyzes the contract text using the combined prompt.
        Returns a dictionary with summary, clauses, flags, and overall_risk_score.
        """
        try:
            # Format the prompt properly
            if isinstance(self.prompt, PromptTemplate):
                formatted_prompt = self.prompt.format(contract_text=contract_text)
            else:
                # If it's already a string template
                formatted_prompt = self.prompt.format(contract_text=contract_text)
            
            # Invoke the LLM
            response = self.llm.invoke(formatted_prompt)
            response_text = response.content.strip()
            
            # Llama sometimes wraps JSON in markdown code blocks
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.rfind("```")
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.rfind("```")
                response_text = response_text[start:end].strip()
            
            # Handle empty responses
            if not response_text or response_text == "":
                return {
                    "summary": "Unable to analyze contract - empty response from model.",
                    "clauses": [],
                    "flags": ["Empty response from LLM"],
                    "overall_risk_score": None
                }
            
            # Fix truncated JSON
            if not response_text.endswith('}'):
                last_complete_clause = response_text.rfind('    }')
                if last_complete_clause != -1:
                    clauses_end = response_text.rfind('  ]', last_complete_clause)
                    if clauses_end != -1:
                        response_text = response_text[:clauses_end + 3] + ',\n  "flags": [],\n  "overall_risk_score": null\n}'
            
            # Parse the JSON response
            result = json.loads(response_text)
            
            # Define Clause Weights for Weighted Average Calculation
            CLAUSE_WEIGHTS = {
                "liability": 3.0,
                "indemnity": 3.0,
                "intellectual property": 3.0,
                "termination": 2.5,
                "fees and payment terms": 2.5,
                "scope of services": 2.0,
                "confidentiality": 1.5,
                "amendments": 1.0,
                "force majeure": 1.0,
                "default": 1.5,
            }

            # Calculate overall risk score as weighted average
            clauses = result.get("clauses", [])
            total_weighted_score = 0.0
            total_weight = 0.0
            
            for clause in clauses:
                score = clause.get("risk_score", 0)
                name = clause.get("clause_name", "").lower()
                weight = CLAUSE_WEIGHTS["default"]
                
                # Find matching weight
                for key, w in CLAUSE_WEIGHTS.items():
                    if key != "default" and key in name:
                        weight = w
                        break
                
                if isinstance(score, (int, float)) and score > 0:
                    total_weighted_score += (score * weight)
                    total_weight += weight

            if total_weight > 0:
                result["overall_risk_score"] = round(total_weighted_score / total_weight, 1)
            else:
                result["overall_risk_score"] = None
            
            return result
        
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed. Error: {str(e)}")
            print(f"Raw response (first 500 chars): {response_text[:500]}...")
            return {
                "summary": "Error: Could not parse LLM response. The model may not have returned valid JSON.",
                "clauses": [],
                "flags": ["JSON parse error - check if model supports structured output"],
                "overall_risk_score": None
            }
        except Exception as e:
            raise RuntimeError(f"Error during contract analysis: {str(e)}") from e

# Example usage (for testing)
if __name__ == "__main__":
    chain = ContractAnalysisChain()
    sample_text = "Sample contract: Party A agrees to pay Party B $1000 for services within 30 days."
    analysis = chain.analyze_contract(sample_text)
    print(json.dumps(analysis, indent=2))