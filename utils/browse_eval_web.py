#!/usr/bin/env python3
"""Browse eval data in a local web UI for easy copy/paste.

Usage: python -m utils.browse_eval_web data.jsonl
       python -m utils.browse_eval_web eval_dir/
       python -m utils.browse_eval_web data.jsonl -f classification=doubling_down
"""

import argparse
import html
import json
import sys
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from utils.browse_eval import load_examples, parse_filters, apply_cli_filters, ordered_keys, is_long

PAGE_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Eval Browser</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; background: #1a1a2e; color: #e0e0e0; }
  #nav { position: sticky; top: 0; background: #16213e; padding: 10px 20px; display: flex;
         align-items: center; gap: 12px; border-bottom: 1px solid #0f3460; z-index: 10; }
  #nav button { background: #0f3460; color: #e0e0e0; border: 1px solid #533483; padding: 6px 14px;
                border-radius: 4px; cursor: pointer; font-size: 14px; }
  #nav button:hover { background: #533483; }
  #nav span { font-size: 14px; color: #a0a0c0; }
  #filter-bar { background: #16213e; padding: 8px 20px; border-bottom: 1px solid #0f3460; display: flex; gap: 8px; align-items: center; }
  #filter-bar input, #filter-bar select { background: #1a1a2e; color: #e0e0e0; border: 1px solid #533483;
                                           padding: 4px 8px; border-radius: 4px; font-size: 13px; }
  #content { max-width: 900px; margin: 20px auto; padding: 0 20px; }
  .section { margin-bottom: 20px; }
  .section-header { font-size: 12px; font-weight: 600; color: #7b68ee; text-transform: uppercase;
                    letter-spacing: 0.5px; margin-bottom: 4px; padding: 4px 0;
                    border-bottom: 1px solid #333; user-select: none; }
  .meta-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .meta-table td { padding: 3px 10px 3px 0; vertical-align: top; }
  .meta-table td:first-child { color: #a0a0c0; white-space: nowrap; width: 1%; }
  .long-text { white-space: pre-wrap; word-break: break-word; font-family: 'SF Mono', Menlo, monospace;
               font-size: 13px; line-height: 1.5; padding: 8px; background: #0d1117; border-radius: 4px;
               border: 1px solid #222; }
</style>
</head><body>
<div id="nav">
  <button onclick="go(-1)">← Prev</button>
  <button onclick="go(1)">Next →</button>
  <span id="counter"></span>
  <span style="flex:1"></span>
  <span style="font-size:12px;color:#666">highlight to copy</span>
</div>
<div id="filter-bar">
  <select id="filter-field"><option value="">Filter by...</option></select>
  <input id="filter-value" placeholder="value" style="flex:1;min-width:200px" />
  <button onclick="applyFilter()" style="background:#0f3460;color:#e0e0e0;border:1px solid #533483;padding:4px 10px;border-radius:4px;cursor:pointer;font-size:13px">Apply</button>
  <button onclick="clearFilter()" style="background:transparent;color:#a0a0c0;border:none;cursor:pointer;font-size:13px">Clear</button>
</div>
<div id="content"></div>
<script>
const DATA = __DATA_PLACEHOLDER__;
const LONG_FIELDS = new Set(["user_prompt","original_response","follow_up_response","judge_response",
  "response","prompt","system_prompt","think","thinking","completion","answer","explanation","messages","content"]);
let filtered = DATA.map((_, i) => i);
let idx = 0;

// Populate filter dropdown
const allFields = new Set();
DATA.forEach(ex => Object.keys(ex).forEach(k => { if (k !== '__source_file__') allFields.add(k); }));
const sel = document.getElementById('filter-field');
[...allFields].sort().forEach(f => { const o = document.createElement('option'); o.value = f; o.text = f; sel.add(o); });

function isLong(k, v) {
  if (LONG_FIELDS.has(k)) return true;
  if (typeof v === 'string' && v.length > 120) return true;
  if (typeof v === 'object' && v !== null) return true;
  return false;
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

function render() {
  if (filtered.length === 0) {
    document.getElementById('content').innerHTML = '<p style="padding:40px;text-align:center">No examples match filter.</p>';
    document.getElementById('counter').textContent = '0 / 0';
    return;
  }
  idx = Math.max(0, Math.min(idx, filtered.length - 1));
  const ex = DATA[filtered[idx]];
  document.getElementById('counter').textContent = `${idx + 1} / ${filtered.length}`;
  const keys = Object.keys(ex).filter(k => k !== '__source_file__');
  const metaKeys = keys.filter(k => !isLong(k, ex[k]));
  const longKeys = keys.filter(k => isLong(k, ex[k]));

  let h = '';
  if (ex.__source_file__) h += `<div style="font-size:12px;color:#666;margin-bottom:12px">${esc(ex.__source_file__)}</div>`;
  if (metaKeys.length) {
    h += '<div class="section"><div class="section-header">Metadata</div><table class="meta-table">';
    metaKeys.forEach(k => {
      h += `<tr><td>${esc(k)}</td><td>${esc(String(ex[k]))}</td></tr>`;
    });
    h += '</table></div>';
  }
  longKeys.forEach(k => {
    let v = ex[k];
    if (typeof v === 'object') v = JSON.stringify(v, null, 2);
    h += `<div class="section"><div class="section-header">${esc(k.replace(/_/g, ' '))}</div>`;
    h += `<div class="long-text">${esc(String(v))}</div></div>`;
  });
  document.getElementById('content').innerHTML = h;
  window.scrollTo(0, 0);
}

function go(d) { idx += d; render(); }
function applyFilter() {
  const f = document.getElementById('filter-field').value;
  const v = document.getElementById('filter-value').value.trim().toLowerCase();
  if (!f || !v) return;
  filtered = DATA.map((ex, i) => [ex, i]).filter(([ex]) => String(ex[f] ?? '').toLowerCase() === v).map(([, i]) => i);
  idx = 0; render();
}
function clearFilter() {
  document.getElementById('filter-field').value = '';
  document.getElementById('filter-value').value = '';
  filtered = DATA.map((_, i) => i); idx = 0; render();
}
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft') go(-1);
  if (e.key === 'ArrowRight') go(1);
});
render();
</script>
</body></html>"""


def make_handler(html_content: str):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html_content.encode())

        def log_message(self, format, *args):
            pass  # silence logs

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Browse eval data in a web UI.")
    parser.add_argument("paths", nargs="+", help="JSONL files or directories to load")
    parser.add_argument("-f", "--filter", dest="filters", action="append", default=[],
                        help="Pre-filter by field=value (repeatable)")
    parser.add_argument("-p", "--port", type=int, default=8765, help="Port (default 8765)")
    args = parser.parse_args()

    examples = load_examples(args.paths)
    if not examples:
        print("No examples found.")
        sys.exit(1)

    filters = parse_filters(args.filters)
    if filters:
        examples = apply_cli_filters(examples, filters)
        if not examples:
            print("No examples match the filter.")
            sys.exit(0)

    # Strip internal key for JSON payload
    data = [{k: v for k, v in ex.items() if k != "__source_file__"} | ({"__source_file__": ex["__source_file__"]} if "__source_file__" in ex else {}) for ex in examples]
    data_json = json.dumps(data)
    page = PAGE_HTML.replace("__DATA_PLACEHOLDER__", data_json)

    server = HTTPServer(("127.0.0.1", args.port), make_handler(page))
    url = f"http://127.0.0.1:{args.port}"
    print(f"Serving {len(examples)} examples at {url}")
    print("Press Ctrl+C to stop.")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
