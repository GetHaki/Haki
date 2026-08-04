// Harness de test des nœuds Haki compilés, hors n8n, contre la vraie API Haki.
//
// Prérequis : API Haki lancée (défaut http://localhost:8100, override via
// HAKI_BASE_URL). Exécute les fonctions execute() des deux nœuds avec un mock
// minimal d'IExecuteFunctions (fetch natif à la place du helper httpRequest).
//
//   npm run build && node --test test/
//
// Note provider : avec HAKI_LLM_PROVIDER=fake, la consolidation ne produit
// aucun fait (le fake n'extrait rien) — le test wait_consolidation vérifie
// donc le traitement du job, pas la création d'un fait. Le chemin LLM réel
// est démontré séparément (démo live OpenRouter).

import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

// n8n-workflow publie un build ESM aux imports sans extension, irrésolvable
// par Node natif : on passe par le build CJS.
const require = createRequire(import.meta.url);
const { NodeOperationError } = require('n8n-workflow');
const { HakiContext } = require('../dist/nodes/HakiContext/HakiContext.node.js');
const { HakiCapture } = require('../dist/nodes/HakiCapture/HakiCapture.node.js');

const BASE_URL = (process.env.HAKI_BASE_URL ?? 'http://localhost:8100').replace(/\/+$/, '');
const PROJECT_ID = 'prj_n8n_harness';
const SUBJECT_ID = 'usr_n8n_harness';

// -- Mock minimal d'IExecuteFunctions ---------------------------------------

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

// -- Haki Context ------------------------------------------------------------

test('Haki Context : subject valide → packet + trace_id', async () => {
	const node = new HakiContext();
	const result = await node.execute.call(
		mockExecuteFunctions(
			{
				project_id: PROJECT_ID,
				subject_id: SUBJECT_ID,
				query: 'dans quelle langue répondre ?',
				budget_tokens: 900,
			},
			CREDENTIALS,
		),
	);
	const item = result[0][0].json;
	assert.ok(item.trace_id, 'trace_id attendu');
	assert.ok(item.context_text.includes('<haki_memory>'), 'bloc mémoire formaté attendu');
	assert.equal(item.subject_id, SUBJECT_ID);
	assert.equal(typeof item.token_count, 'number');
	console.log('  context_text =', JSON.stringify(item.context_text.slice(0, 120)) + '…');
	console.log('  trace_id =', item.trace_id, '| token_count =', item.token_count);
});

test('Haki Context : subject vide → NodeOperationError', async () => {
	const node = new HakiContext();
	await assert.rejects(
		node.execute.call(
			mockExecuteFunctions(
				{ project_id: PROJECT_ID, subject_id: '', query: 'q', budget_tokens: 900 },
				CREDENTIALS,
			),
		),
		(error) => {
			assert.ok(error instanceof NodeOperationError, 'NodeOperationError attendu');
			assert.match(error.message, /subject_id/);
			return true;
		},
	);
});

test('Haki Context : subject « default » → NodeOperationError', async () => {
	const node = new HakiContext();
	await assert.rejects(
		node.execute.call(
			mockExecuteFunctions(
				{ project_id: PROJECT_ID, subject_id: 'default', query: 'q', budget_tokens: 900 },
				CREDENTIALS,
			),
		),
		NodeOperationError,
	);
});

// -- Haki Capture ------------------------------------------------------------

test('Haki Capture : tour capturé → visible via /v1/timeline', async () => {
	const node = new HakiCapture();
	const userMessage = `harness capture ${Date.now()}`;
	const result = await node.execute.call(
		mockExecuteFunctions(
			{
				project_id: PROJECT_ID,
				subject_id: SUBJECT_ID,
				user_message: userMessage,
				assistant_message: 'réponse du harness',
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
	assert.ok(found, 'événement attendu dans la timeline');
	assert.equal(found.payload.messages[0].content, userMessage);
	console.log('  timeline : événement retrouvé, kind =', found.kind);
});

test('Haki Capture : idempotence — même run rejoué → dédupliqué', async () => {
	const node = new HakiCapture();
	const params = {
		project_id: PROJECT_ID,
		subject_id: SUBJECT_ID,
		user_message: 'tour idempotent',
		assistant_message: 'même réponse',
		run_id: `run_${Date.now()}`,
	};
	const first = await node.execute.call(mockExecuteFunctions(params, CREDENTIALS));
	const second = await node.execute.call(mockExecuteFunctions(params, CREDENTIALS));
	assert.equal(first[0][0].json.events[0].deduplicated, false);
	assert.equal(second[0][0].json.events[0].deduplicated, true);
	assert.equal(second[0][0].json.consolidation_job_id, null);
	console.log('  replay : deduplicated = true, pas de nouveau job');
});

test('Haki Capture : wait_consolidation → job traité (provider fake : pas de fait)', async () => {
	const node = new HakiCapture();
	const result = await node.execute.call(
		mockExecuteFunctions(
			{
				project_id: PROJECT_ID,
				subject_id: SUBJECT_ID,
				user_message: 'je préfère le français (harness wait)',
				assistant_message: 'noté',
				wait_consolidation: true,
			},
			CREDENTIALS,
		),
	);
	const item = result[0][0].json;
	assert.equal(typeof item.processed, 'number');
	assert.ok(item.processed >= 1, 'au moins le job créé par cette capture');
	console.log('  consolidation : processed =', item.processed);
	console.log('  (HAKI_LLM_PROVIDER=fake → 0 fait extrait, comportement attendu en dev)');
});

test('Haki Capture : subject vide → NodeOperationError', async () => {
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
