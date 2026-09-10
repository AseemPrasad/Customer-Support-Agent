const SCENARIOS = {
  delayed_shipping: "My order #123-4567890-1234567 was supposed to arrive yesterday but the tracking has not updated at all. Where is my package?",
  legal_threat: "I am going to sue Amazon and contact my attorney if my refund of $450 is not processed immediately. This is fraud!",
  pii_leak: "Please check my refund for account user@example.com, card ending in 4242, phone 555-0199.",
  multilingual_ja: "注文した商品がまだ届きません。追跡情報を確認してください。注文番号 111-2223334-5556667",
  damaged_item: "I received my package today but the item inside is completely crushed and broken. I need a replacement right away."
};

function loadScenario(key) {
  const input = document.getElementById('tweet-input');
  if (SCENARIOS[key]) {
    input.value = SCENARIOS[key];
  }
}

async function processTicket() {
  const input = document.getElementById('tweet-input');
  const tweetText = input.value.trim();

  if (!tweetText) {
    alert("Please enter a customer message or select a quick scenario!");
    return;
  }

  const btn = document.getElementById('process-btn');
  btn.disabled = true;
  btn.innerText = "⚡ PROCESSING WITH LLM + RAG...";

  try {
    const res = await fetch('/api/process_ticket', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tweet: tweetText })
    });

    const data = await res.json();
    
    if (data.error) {
      alert("Error: " + data.error);
      return;
    }

    renderResults(data);
  } catch (err) {
    alert("Failed to communicate with server: " + err);
  } finally {
    btn.disabled = false;
    btn.innerText = "⚡ PROCESS TICKET WITH AGENT";
  }
}

function renderResults(data) {
  document.getElementById('placeholder-state').classList.add('hidden');
  document.getElementById('results-state').classList.remove('hidden');

  const d = data.decision;
  document.getElementById('res-intent').innerText = d.intent || 'UNKNOWN';
  document.getElementById('res-decision').innerText = d.decision || 'AUTO_REPLY';
  document.getElementById('res-language').innerText = (d.detected_language || 'EN').toUpperCase();
  document.getElementById('res-reason').innerText = d.escalation_reason || 'NONE';
  document.getElementById('res-reply').innerText = d.draft_reply || '(No reply generated)';
  document.getElementById('char-num').innerText = (d.draft_reply || '').length;

  // Format decision badge color
  const decTag = document.getElementById('res-decision');
  if (d.decision === 'ESCALATE') {
    decTag.className = 'value-tag red';
  } else {
    decTag.className = 'value-tag green';
  }

  // Render RAG context snippets
  const ragContainer = document.getElementById('rag-container');
  ragContainer.innerHTML = '';

  if (data.rag_snippets && data.rag_snippets.length > 0) {
    data.rag_snippets.forEach((snip, idx) => {
      const card = document.createElement('div');
      card.className = 'rag-card';
      card.innerHTML = `<strong>RAG MATCH #${idx + 1}:</strong> ${snip}`;
      ragContainer.appendChild(card);
    });
  } else {
    ragContainer.innerHTML = `<div class="rag-placeholder">No RAG context used (Ticket escalated by Guardrails).</div>`;
  }
}

async function runEvalBatch() {
  const btn = document.getElementById('run-eval-btn');
  btn.disabled = true;
  btn.innerText = "⏳ RUNNING BATCH EVALUATION...";

  const resultsDiv = document.getElementById('eval-results');
  resultsDiv.classList.remove('hidden');

  try {
    const res = await fetch('/api/run_eval', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_size: 20 })
    });

    const data = await res.json();
    if (data.error) {
      alert("Eval error: " + data.error);
      return;
    }

    document.getElementById('eval-total').innerText = data.total_evaluated;
    document.getElementById('eval-acc').innerText = data.auto_reply_accuracy + '%';
    document.getElementById('eval-f1').innerText = (data.intent_f1_score || '--') + '%';
    document.getElementById('eval-fnr').innerText = data.escalation_fnr + '%';
    document.getElementById('eval-pii').innerText = data.pii_pass_rate + '%';

  } catch (err) {
    alert("Evaluation request failed: " + err);
  } finally {
    btn.disabled = false;
    btn.innerText = "📊 RUN BATCH EVALUATION";
  }
}
