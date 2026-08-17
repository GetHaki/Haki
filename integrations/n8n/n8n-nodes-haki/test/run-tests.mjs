// Test harness for the compiled Haki nodes, outside n8n, against a real
// Haki API.
//
// Prerequisite: Haki API running (default http://localhost:8100, override
// via HAKI_BASE_URL). Runs both nodes' execute() functions with a minimal
// IExecuteFunctions mock (native fetch instead of the httpRequest helper).
//
//   npm run build && node --test test/
//
// Provider note: with HAKI_LLM_PROVIDER=fake, consolidation produces no
// facts (the fake extractor extracts nothing) -- the wait_consolidation
// test checks that the job is processed, not that a fact is created. The
// real LLM path is demonstrated separately (live OpenRouter demo).

import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

// n8n-workflow ships an ESM build with extensionless imports that native
// Node can't resolve: go through the CJS build instead.
const require = createRequire(import.meta.url);
const { NodeOperationError } = require('n8n-workflow');
const { HakiContext } = require('../dist/nodes/HakiContext/HakiContext.node.js');
const { HakiCapture } = require('../dist/nodes/HakiCapture/HakiCapture.node.js');

const BASE_URL = (process.env.HAKI_BASE_URL ?? 'http://localhost:8100').replace(/\/+$/, '');
const PROJECT_ID = 'prj_n8n_harness';
const SUBJECT_ID = 'usr_n8n_harness';

// -- Minimal IExecuteFunctions mock ------------------------------------------

function mockExecuteFunctions(params, credentials) {
	return {
		getInputData: () => [{ json: {} }],
		getNodeParameter: (name, _index, fallback) =>
			params[name] !== undefined && params[name] !== '' ? params[name] : fallback,
		getCredentials: async () => credentials,
		getNode: () => ({
			name: 'HarnessNode',
			type: 'harness',
			typeVersion: 1,
			position: [0, 0],
			parameters: {},
		}),
		continueOnFail: () => false,
		helpers: {
			httpRequest: async ({ method, url, headers = {}, body }) => {
				const response = await fetch(url, {
					method,
					headers: { 'content-type': 'application/json', ...headers },
					body: body === undefined ? undefined : JSON.stringify(body),
				});
				if (!response.ok) {
					const error = new Error(`HTTP ${response.status}`);
					error.status = response.status;
					error.body = await response.text();
					throw error;
				}
				return response.json();
			},
		},
	};
}

const CREDENTIALS = { base_url: BASE_URL, api_key: '' };

// -- Haki Context -------------------------------------------------------------

test('Haki Context: valid subject -> packet + trace_id', async () => {
	const node = new HakiContext();
	const result = await node.execute.call(
		mockExecuteFunctions(
			{
				project_id: PROJECT_ID,
				subject_id: SUBJECT_ID,
				query: 'which language to reply in?',
				budget_tokens: 2000,
			},
			CREDENTIALS,
		),
	);
	const item = result[0][0].json;
	assert.ok(item.trace_id, 'expected a trace_id');
	assert.ok(item.context_text.includes('<haki_memory>'), 'expected a formatted memory block');
	assert.equal(item.subject_id, SUBJECT_ID);
	assert.equal(typeof item.token_count, 'number');
	console.log('  context_text =', JSON.stringify(item.context_text.slice(0, 120)) + '...');
	console.log('  trace_id =', item.trace_id, '| token_count =', item.token_count);
});

test('Haki Context: empty subject -> NodeOperationError', async () => {
	const node = new HakiContext();
	await assert.rejects(
		node.execute.call(
			mockExecuteFunctions(
				{ project_id: PROJECT_ID, subject_id: '', query: 'q', budget_tokens: 2000 },
				CREDENTIALS,
			),
		),
		(error) => {
			assert.ok(error instanceof NodeOperationError, 'expected a NodeOperationError');
			assert.match(error.message, /subject_id/);
			return true;
		},
	);
});

test('Haki Context: "default" subject -> NodeOperationError', async () => {
	const node = new HakiContext();
	await assert.rejects(
		node.execute.call(
			mockExecuteFunctions(
				{ project_id: PROJECT_ID, subject_id: 'default', query: 'q', budget_tokens: 2000 },
				CREDENTIALS,
			),
		),
		NodeOperationError,
	);
});

// -- Haki Capture -------------------------------------------------------------

test('Haki Capture: captured turn -> visible via /v1/timeline', async () => {
	const node = new HakiCapture();
	const userMessage = `harness capture ${Date.now()}`;
	const result = await node.execute.call(
		mockExecuteFunctions(
			{
				project_id: PROJECT_ID,
				subject_id: SUBJECT_ID,
				user_message: userMessage,
				assistant_message: 'harness reply',
				thread_id: 'thread_harness',
			},
			CREDENTIALS,
		),
	);
	const item = result[0][0].json;
	assert.equal(item.events.length, 1);
	assert.match(item.idempotency_key, /^n8n-turn-thread_harness-[0-9a-f]{16}$/);
	console.log('  event_id =', item.events[0].id, '| key =', item.idempotency_key);

	const timeline = await (
		await fetch(`${BASE_URL}/v1/timeline?project_id=${PROJECT_ID}&subject_id=${SUBJECT_ID}`)
	).json();
	const found = timeline.events.find((e) => e.idempotency_key === item.idempotency_key);
	assert.ok(found, 'expected the event in the timeline');
	assert.equal(found.payload.messages[0].content, userMessage);
	console.log('  timeline: event found, kind =', found.kind);
});

test('Haki Capture: idempotency -- same run replayed -> deduplicated', async () => {
	const node = new HakiCapture();
	const params = {
		project_id: PROJECT_ID,
		subject_id: SUBJECT_ID,
		user_message: 'idempotent turn',
		assistant_message: 'same reply',
		run_id: `run_${Date.now()}`,
	};
	const first = await node.execute.call(mockExecuteFunctions(params, CREDENTIALS));
	const second = await node.execute.call(mockExecuteFunctions(params, CREDENTIALS));
	assert.equal(first[0][0].json.events[0].deduplicated, false);
	assert.equal(second[0][0].json.events[0].deduplicated, true);
	assert.equal(second[0][0].json.consolidation_job_id, null);
	console.log('  replay: deduplicated = true, no new job');
});

test('Haki Capture: wait_consolidation -> job processed (fake provider: no fact)', async () => {
	const node = new HakiCapture();
	const result = await node.execute.call(
		mockExecuteFunctions(
			{
				project_id: PROJECT_ID,
				subject_id: SUBJECT_ID,
				user_message: 'I prefer French (harness wait)',
				assistant_message: 'noted',
				wait_consolidation: true,
			},
			CREDENTIALS,
		),
	);
	const item = result[0][0].json;
	assert.equal(typeof item.processed, 'number');
	assert.ok(item.processed >= 1, 'expected at least the job created by this capture');
	console.log('  consolidation: processed =', item.processed);
	console.log('  (HAKI_LLM_PROVIDER=fake -> 0 facts extracted, expected in dev)');
});

test('Haki Capture: empty subject -> NodeOperationError', async () => {
	const node = new HakiCapture();
	await assert.rejects(
		node.execute.call(
			mockExecuteFunctions(
				{
					project_id: PROJECT_ID,
					subject_id: 'default',
					user_message: 'u',
					assistant_message: 'a',
				},
				CREDENTIALS,
			),
		),
		NodeOperationError,
	);
});
