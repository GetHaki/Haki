import { createHash, randomUUID } from 'crypto';
import type {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
} from 'n8n-workflow';
import {
	apiError,
	authHeaders,
	baseUrl,
	requireSubject,
	type HakiCredentials,
} from '../utils';

interface CaptureApiResponse {
	status: string;
	events: { id: string; deduplicated: boolean }[];
	consolidation_job_id: string | null;
	policy: string;
}

/** Idempotency key derived from the run/thread when available: re-running the
 * same n8n execution never duplicates the captured turn. */
function idempotencyKey(
	runId: string,
	threadId: string,
	userMessage: string,
	assistantMessage: string,
): string {
	const anchor = runId || threadId;
	const digest = createHash('sha256')
		.update(`${userMessage}\n${assistantMessage}`)
		.digest('hex')
		.slice(0, 16);
	return anchor ? `n8n-turn-${anchor}-${digest}` : `n8n-turn-${randomUUID()}`;
}

export class HakiCapture implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Haki Capture',
		name: 'hakiCapture',
		group: ['transform'],
		version: 1,
		description:
			"Records the user/assistant turn AFTER the LLM call. Always place after the AI Agent.",
		defaults: { name: 'Haki Capture' },
		inputs: ['main'],
		outputs: ['main'],
		credentials: [{ name: 'hakiApi', required: true }],
		properties: [
			{
				displayName: 'Project ID',
				name: 'project_id',
				type: 'string',
				default: '',
				required: true,
				placeholder: 'prj_support',
				description: 'Haki project (must match the Haki Context node).',
			},
			{
				displayName: 'Subject ID',
				name: 'subject_id',
				type: 'string',
				default: '',
				required: true,
				placeholder: "{{ $('Haki Context').item.json.subject_id }}",
				description: "Stable identifier — reuse the one from the Haki Context node.",
			},
			{
				displayName: 'User Message',
				name: 'user_message',
				type: 'string',
				default: '',
				required: true,
				description: "The turn's user message.",
			},
			{
				displayName: 'Assistant Message',
				name: 'assistant_message',
				type: 'string',
				default: '',
				required: true,
				description: "The agent's reply for this turn.",
			},
			{
				displayName: 'Thread ID',
				name: 'thread_id',
				type: 'string',
				default: '',
				description: 'Optional: conversation thread (used for idempotency).',
			},
			{
				displayName: 'Run ID',
				name: 'run_id',
				type: 'string',
				default: '',
				description: "Optional: execution ID (takes priority for idempotency).",
			},
			{
				displayName: 'Wait Consolidation',
				name: 'wait_consolidation',
				type: 'boolean',
				default: false,
				description:
					'When on, also calls POST /v1/consolidate so the memory is recallable immediately (dev/demo).',
			},
			{
				displayName: 'Org ID',
				name: 'org_id',
				type: 'string',
				default: 'org_default',
				description: 'Haki organization (contract B.1).',
			},
			{
				displayName: 'Origin Trust',
				name: 'origin_trust',
				type: 'options',
				options: [
					{ name: 'Trusted (direct message from the subject)', value: 'trusted' },
					{ name: 'Semi-trusted (agent/tool output)', value: 'semi_trusted' },
					{ name: 'Third-party (someone else in the conversation)', value: 'third_party' },
					{ name: 'Untrusted (ingested content)', value: 'untrusted' },
				],
				default: 'trusted',
				description:
					"Authority level of the captured turn's origin. In a group (the sender is not the tracked subject), choose third-party — the fact will be attributed to that third party, never to the subject.",
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const returnData: INodeExecutionData[] = [];
		const credentials = (await this.getCredentials('hakiApi')) as unknown as HakiCredentials;
		const root = baseUrl(credentials);
		const headers = authHeaders(credentials);

		for (let i = 0; i < items.length; i++) {
			const projectId = this.getNodeParameter('project_id', i) as string;
			const subjectId = requireSubject(
				this.getNode(),
				this.getNodeParameter('subject_id', i) as string,
				i,
			);
			const userMessage = this.getNodeParameter('user_message', i) as string;
			const assistantMessage = this.getNodeParameter('assistant_message', i) as string;
			const threadId = (this.getNodeParameter('thread_id', i) as string) || '';
			const runId = (this.getNodeParameter('run_id', i) as string) || '';
			const waitConsolidation = this.getNodeParameter('wait_consolidation', i) as boolean;
			const orgId = (this.getNodeParameter('org_id', i) as string) || 'org_default';
			const originTrust = this.getNodeParameter('origin_trust', i) as string;

			const key = idempotencyKey(runId, threadId, userMessage, assistantMessage);
			const event = {
				org_id: orgId,
				project_id: projectId,
				subject_type: 'user',
				subject_id: subjectId,
				agent_id: 'n8n',
				origin_trust: originTrust,
				...(threadId ? { thread_id: threadId } : {}),
				...(runId ? { run_id: runId } : {}),
				kind: 'conversation.turn',
				occurred_at: new Date().toISOString(),
				payload: {
					messages: [
						{ role: 'user', content: userMessage },
						{ role: 'assistant', content: assistantMessage },
					],
				},
				source: { integration: 'n8n', node: 'HakiCapture' },
				idempotency_key: key,
			};

			let capture: CaptureApiResponse;
			try {
				// Per-event key only: a batch-level key would be suffixed
				// with a content hash on the Ledger side and would break
				// dedup on replay (occurred_at changes on every run).
				capture = (await this.helpers.httpRequest({
					method: 'POST',
					url: `${root}/v1/capture`,
					headers,
					body: { events: [event] },
					json: true,
				})) as CaptureApiResponse;
			} catch (error) {
				throw apiError(this.getNode(), error, i);
			}

			let processed: number | null = null;
			if (waitConsolidation) {
				try {
					const consolidation = (await this.helpers.httpRequest({
						method: 'POST',
						url: `${root}/v1/consolidate`,
						headers,
						json: true,
					})) as { processed: number };
					processed = consolidation.processed;
				} catch (error) {
					throw apiError(this.getNode(), error, i);
				}
			}

			returnData.push({
				json: {
					status: capture.status,
					events: capture.events,
					consolidation_job_id: capture.consolidation_job_id,
					policy: capture.policy,
					idempotency_key: key,
					processed,
					subject_id: subjectId,
					project_id: projectId,
				},
				pairedItem: { item: i },
			});
		}
		return [returnData];
	}
}
