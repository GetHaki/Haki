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
	formatPacket,
	requireSubject,
	type ContextApiResponse,
	type HakiCredentials,
} from '../utils';

export class HakiContext implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Haki Context',
		name: 'hakiContext',
		group: ['transform'],
		version: 1,
		description:
			"Fetches the subject's long-term memory (ContextPacket) BEFORE the LLM call. Always place before the AI Agent.",
		defaults: { name: 'Haki Context' },
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
				description: 'Haki project (memory scope). Never chosen by the model.',
			},
			{
				displayName: 'Subject ID',
				name: 'subject_id',
				type: 'string',
				default: '',
				required: true,
				placeholder: '{{ $json.body.subject_id }}',
				description:
					"Stable identifier for the person (sessionId, email, webhook ID...). Empty or \"default\" is an error.",
			},
			{
				displayName: 'Query',
				name: 'query',
				type: 'string',
				default: '',
				required: true,
				placeholder: '{{ $json.body.message }}',
				description: 'The current message — used to rerank relevant facts.',
			},
			{
				displayName: 'Budget Tokens',
				name: 'budget_tokens',
				type: 'number',
				default: 2000,
				description: 'Token budget for the ContextPacket.',
			},
			{
				displayName: 'Purpose',
				name: 'purpose',
				type: 'string',
				default: '',
				description: 'Optional: task type (support, onboarding...), recorded in the trace.',
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const returnData: INodeExecutionData[] = [];
		const credentials = (await this.getCredentials('hakiApi')) as unknown as HakiCredentials;
		const url = `${baseUrl(credentials)}/v1/context`;
		const headers = authHeaders(credentials);

		for (let i = 0; i < items.length; i++) {
			const projectId = this.getNodeParameter('project_id', i) as string;
			const subjectId = requireSubject(
				this.getNode(),
				this.getNodeParameter('subject_id', i) as string,
				i,
			);
			const query = this.getNodeParameter('query', i) as string;
			const budgetTokens = this.getNodeParameter('budget_tokens', i) as number;
			const purpose = (this.getNodeParameter('purpose', i) as string) || null;

			let response: ContextApiResponse;
			try {
				response = (await this.helpers.httpRequest({
					method: 'POST',
					url,
					headers,
					body: {
						project_id: projectId,
						subject_id: subjectId,
						query,
						purpose,
						budget_tokens: budgetTokens,
					},
					json: true,
				})) as ContextApiResponse;
			} catch (error) {
				throw apiError(this.getNode(), error, i);
			}

			returnData.push({
				json: {
					// Text ready to inject into the agent's system prompt.
					context_text: formatPacket(response.packet),
					packet: response.packet,
					warnings: response.packet.warnings ?? [],
					token_count: response.token_count,
					trace_id: response.trace_id,
					subject_id: subjectId,
					project_id: projectId,
				},
				pairedItem: { item: i },
			});
		}
		return [returnData];
	}
}
