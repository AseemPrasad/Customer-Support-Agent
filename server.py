"""
Standalone HTTP Web Server for Neo-Brutalist UAT Workbench.
100% Non-destructive: Imports existing src modules in a read-only manner.
"""

import http.server
import json
import socketserver
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Read-only imports from existing codebase
from src.pipeline import AmazonSupportAgent
from src.vector_store import KnowledgeBase

PORT = 8000

# Initialize Agent lazy-loaded
AGENT = None


def get_agent():
    global AGENT
    if AGENT is None:
        print("Initializing KnowledgeBase & AmazonSupportAgent for Web Server...")
        kb = KnowledgeBase()
        AGENT = AmazonSupportAgent(kb=kb)
    return AGENT


class UATHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)

        # Serve UI static files
        if parsed_path.path == "/" or parsed_path.path == "/index.html":
            self.path = "/static/index.html"
            return super().do_GET()
        elif parsed_path.path.startswith("/static/"):
            return super().do_GET()
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        parsed_path = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if parsed_path.path == "/api/process_ticket":
            self.handle_process_ticket(payload)
        elif parsed_path.path == "/api/run_eval":
            self.handle_run_eval(payload)
        else:
            self.send_error(404, "Endpoint Not Found")

    def handle_process_ticket(self, payload):
        tweet = payload.get("tweet", "").strip()
        if not tweet:
            self.send_json_response({"error": "Tweet text is required"}, status=400)
            return

        try:
            agent = get_agent()
            decision = agent.process_ticket(tweet)

            # Retrieve RAG snippets for display
            rag_snippets = getattr(decision, "rag_snippets", [])

            response_data = {
                "decision": {
                    "intent": decision.intent,
                    "decision": decision.decision,
                    "escalation_reason": decision.escalation_reason,
                    "draft_reply": decision.draft_reply,
                    "detected_language": decision.detected_language,
                },
                "rag_snippets": rag_snippets,
            }
            self.send_json_response(response_data)

        except Exception as e:
            self.send_json_response({"error": str(e)}, status=500)

    def handle_run_eval(self, payload):
        sample_size = payload.get("sample_size", 20)
        try:
            import random
            from pathlib import Path
            from evaluation.automated_metrics import compute_intent_metrics, compute_escalation_fnr, check_pii_leakage
            
            golden_path = Path(__file__).resolve().parent / "data" / "golden_set" / "golden_eval_500.gold.jsonl"
            if not golden_path.exists():
                self.send_json_response({"error": "Golden set file not found"}, status=404)
                return

            all_records = []
            with open(golden_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_records.append(json.loads(line.strip()))

            # Stratified sampling: 50% AUTO_REPLY, 50% ESCALATE for balanced evaluation
            auto_recs = [r for r in all_records if r.get("gold_decision") == "AUTO_REPLY"]
            esc_recs = [r for r in all_records if r.get("gold_decision") == "ESCALATE"]

            random.seed(42)  # Deterministic seed for reproducible batch runs
            n_each = sample_size // 2
            sampled_auto = random.sample(auto_recs, min(n_each, len(auto_recs)))
            sampled_esc = random.sample(esc_recs, min(n_each, len(esc_recs)))
            records = sampled_auto + sampled_esc
            random.shuffle(records)

            agent = get_agent()
            y_true_intent = []
            y_pred_intent = []
            y_true_dec = []
            y_pred_dec = []
            pii_clean_count = 0

            for rec in records:
                gold_intent = rec.get("gold_intent") or rec.get("source_intent") or "general_inquiry"
                gold_dec = rec.get("gold_decision") or "AUTO_REPLY"
                inquiry = rec.get("input_tweet") or rec.get("inquiry", "")

                decision = agent.process_ticket(inquiry)
                
                y_true_intent.append(gold_intent)
                y_pred_intent.append(decision.intent)
                y_true_dec.append(gold_dec)
                y_pred_dec.append(decision.decision)

                # Check PII leakage on reply
                if decision.draft_reply:
                    leaks = check_pii_leakage(decision.draft_reply)
                    if not leaks:
                        pii_clean_count += 1
                else:
                    pii_clean_count += 1

            intent_res = compute_intent_metrics(y_true_intent, y_pred_intent)
            esc_res = compute_escalation_fnr(y_true_dec, y_pred_dec)

            acc = sum(1 for t, p in zip(y_true_dec, y_pred_dec) if t == p) / max(len(y_true_dec), 1) * 100
            fnr = esc_res["fnr"] * 100
            pii_rate = (pii_clean_count / max(len(records), 1)) * 100
            macro_f1 = intent_res["macro_avg"]["f1"] * 100

            response_data = {
                "total_evaluated": len(records),
                "auto_reply_accuracy": round(acc, 1),
                "escalation_fnr": round(fnr, 1),
                "pii_pass_rate": round(pii_rate, 1),
                "intent_f1_score": round(macro_f1, 1),
            }
            self.send_json_response(response_data)

        except Exception as e:
            self.send_json_response({"error": str(e)}, status=500)

    def send_json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

if __name__ == "__main__":
    print(f"Starting Neo-Brutalist UAT Web Server at http://localhost:{PORT}")
    print("Press Ctrl+C to stop.")
    with ReusableTCPServer(("", PORT), UATHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
