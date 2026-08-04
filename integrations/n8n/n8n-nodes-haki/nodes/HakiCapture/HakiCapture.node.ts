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

/** Idempotency key dérivée du run/thread quand disponible : relancer la même
 * exécution n8n ne duplique jamais le tour capturé. */
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
			"Enregistre le tour user/assistant APRÈS l'appel LLM. Toujours placer après l'AI Agent.",
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
				description: 'Projet Haki (doit être le même que le nœud Haki Context).',
			},
			{
				displayName: 'Subject ID',
				name: 'subject_id',
				type: 'string',
				default: '',
				required: true,
				placeholder: "{{ $('Haki Context').item.json.subject_id }}",
				description: "Identifiant stable — reprendre celui du nœud Haki Context.",
			},
			{
				displayName: 'User Message',
				name: 'user_message',
				type: 'string',
				default: '',
				required: true,
				description: 'Le message utilisateur du tour.',
			},
			{
				displayName: 'Assistant Message',
				name: 'assistant_message',
				type: 'string',
				default: '',
				required: true,
				description: "La réponse de l'agent pour ce tour.",
			},
			{
				displayName: 'Thread ID',
				name: 'thread_id',
				type: 'string',
				default: '',
				description: 'Optionnel : fil de conversation (utilisé pour l’idempotence).',
			},
			{
				displayName: 'Run ID',
				name: 'run_id',
				type: 'string',
				default: '',
				description: "Optionnel : identifiant d'exécution (prioritaire pour l’idempotence).",
			},
			{
				displayName: 'Wait Consolidation',
				name: 'wait_consolidation',
				type: 'boolean',
				default: false,
				description:
					'Si actif, appelle aussi POST /v1/consolidate pour que la mémoire soit rappelable immédiatement (dev/démo).',
			},
			{
				displayName: 'Org ID',
				name: 'org_id',
				type: 'string',
				default: 'org_default',
				description: 'Organisation Haki (contrat B.1).',
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

			const key = idempotencyKey(runId, threadId, userMessage, assistantMessage);
			const event = {
				org_id: orgId,
				project_id: projectId,
				subject_type: 'user',
				subject_id: subjectId,
				agent_id: 'n8n',
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
				// Clé par événement uniquement : une clé de batch serait
				// suffixée du hash de contenu côté Ledger et casserait la
				// déduplication d'un replay (occurred_at change à chaque run).
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
