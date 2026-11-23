"""Exercise the dashboard's inline rendering without server or ML dependencies."""

import json
from pathlib import Path
import shutil
import subprocess
import unittest


NODE = shutil.which("node")
DASHBOARD = Path(__file__).resolve().parents[1] / "research" / "dashboard.html"


@unittest.skipUnless(NODE, "Node.js is required to execute dashboard JavaScript")
class DashboardRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        html = DASHBOARD.read_text(encoding="utf-8")
        script = html.split("<script>", 1)[1].split("</script>", 1)[0]
        # Load the helpers without installing handlers or starting API polling.
        cls.script = script.split("$('refresh').onclick=refresh;", 1)[0]

    def run_js(self, assertions):
        harness = """
const assert = require('node:assert/strict');
const vm = require('node:vm');
class Element {
  constructor(tagName) {
    this.tagName = tagName;
    this.children = [];
    this.textContent = '';
    this.open = false;
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = [...children]; }
}
const elements = new Map(['model', 'graph'].map(id => [id, new Element('div')]));
const document = {
  createElement: tag => new Element(tag),
  getElementById: id => elements.get(id),
};
const context = {document, location: {search: ''}, URLSearchParams};
const tables = id => elements.get(id).children.filter(el => el.tagName === 'table');
const rows = table => table.children.filter(el => el.tagName === 'tr')
  .map(tr => tr.children.map(td => td.textContent));
function checkJSON(id, data) {
  const details = elements.get(id).children.at(-1);
  assert.equal(details.tagName, 'details');
  assert.equal(details.open, false);
  assert.equal(details.children[0].textContent, '詳細データ（JSON）');
  assert.equal(details.children[1].tagName, 'pre');
  assert.deepEqual(JSON.parse(details.children[1].textContent), data ?? null);
}
"""
        result = subprocess.run(
            [NODE, "-e", harness + "\nvm.createContext(context);\n"
             + "vm.runInContext(" + json.dumps(self.script) + ", context);\n"
             + assertions],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_renderers_tolerate_null_and_missing_fields(self):
        self.run_js("""
assert.equal(typeof context.renderModel, 'function');
assert.equal(typeof context.renderGraph, 'function');
for (const model of [null, undefined, {}, {
  status: null, state_answers: 3, required_days: 5,
  data_mode: null, training_record_count: null,
}]) {
  context.renderModel(model);
  assert.deepEqual(rows(tables('model')[0]), []);
  checkJSON('model', model);
}
for (const graph of [null, undefined, {}, {nodes: null, edges: null}]) {
  context.renderGraph(graph);
  assert.deepEqual(rows(tables('graph')[0]), [['kind', '件数']]);
  assert.deepEqual(rows(tables('graph')[1]), [['関連', '件数'], ['合計', '0']]);
  assert.deepEqual(rows(tables('graph')[2]), [['id', 'kind']]);
  checkJSON('graph', graph);
}
""")

    def test_model_summary_formats_present_fields_and_preserves_json(self):
        self.run_js("""
const model = {
  status: 'insufficient_labels', state_answers: 8, required_answers: 20,
  days: 2, required_days: 5, data_mode: 'combined', contains_sample: true,
  training_record_count: 0, extra: 'retained in JSON',
};
context.renderModel(model);
assert.deepEqual(rows(tables('model')[0]), [
  ['学習状態', 'データ不足（学習開始前）'],
  ['回答数', '8 / 20 件（あと12件）'],
  ['対象日数', '2 / 5 日'],
  ['データモード', 'combined（サンプル含む）'],
  ['学習レコード数', '0 件'],
]);
checkJSON('model', model);
for (const [status, label] of [
  ['trained', '学習済み'], ['training', '学習中'],
  ['untrained', 'untrained'], ['constructor', 'constructor'],
]) {
  context.renderModel({status, state_answers: 20, required_answers: 20,
                       data_mode: 'real', contains_sample: false});
  assert.deepEqual(rows(tables('model')[0]), [
    ['学習状態', label], ['回答数', '20 / 20 件'], ['データモード', 'real'],
  ]);
  assert.equal(elements.get('model').children.length, 2);
}
context.renderModel({state_answers: 0, required_answers: 20, days: 0, required_days: 5});
assert.deepEqual(rows(tables('model')[0]), [
  ['回答数', '0 / 20 件（あと20件）'], ['対象日数', '0 / 5 日'],
]);
""")

    def test_graph_counts_node_limit_and_edge_total_fallback(self):
        self.run_js("""
const graph = {
  nodes: Array.from({length: 103}, (_, i) => ({
    id: i === 0 ? '<script>alert(1)</script>' : 'node:' + i,
    kind: i < 3 ? 'Day' : 'Observation',
  })),
  edges: [{kind: 'provenance'}, {kind: 'part_of'}, {kind: 'part_of'}],
};
context.renderGraph(graph);
assert.deepEqual(rows(tables('graph')[0]), [
  ['kind', '件数'], ['Observation', '100'], ['Day', '3'],
]);
assert.deepEqual(rows(tables('graph')[1]), [
  ['kind', '件数'], ['part_of', '2'], ['provenance', '1'],
]);
const nodeRows = rows(tables('graph')[2]);
assert.equal(nodeRows.length, 102);
assert.deepEqual(nodeRows[1], [graph.nodes[0].id, 'Day']);
assert.deepEqual(nodeRows[100], ['node:99', 'Observation']);
assert.deepEqual(nodeRows[101], ['他 3 件']);
checkJSON('graph', graph);
context.renderGraph({nodes: graph.nodes.slice(0, 100),
                     edges: [{source: 'a', target: 'b'}]});
assert.equal(tables('graph').length, 3);
assert.equal(rows(tables('graph')[2]).length, 101);
assert.deepEqual(rows(tables('graph')[1]), [['関連', '件数'], ['合計', '1']]);
""")


if __name__ == "__main__":
    unittest.main()
