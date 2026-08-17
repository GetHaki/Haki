import type { ICredentialType, INodeProperties } from 'n8n-workflow';

export class HakiApi implements ICredentialType {
	name = 'hakiApi';

	displayName = 'Haki API';

	documentationUrl = 'https://github.com/GetHaki/Haki';

	properties: INodeProperties[] = [
		{
			displayName: 'Base URL',
			name: 'base_url',
			type: 'string',
			default: 'http://localhost:8100',
			required: true,
			placeholder: 'http://localhost:8100',
			description:
				"The Haki API URL, no trailing slash. From an n8n Docker container to a Haki API on the host machine: http://host.docker.internal:8100",
		},
		{
			displayName: 'API Key',
			name: 'api_key',
			type: 'string',
			typeOptions: { password: true },
			default: '',
			description:
				"Haki bearer key. Optional in local development (the API runs in open mode), required as soon as HAKI_ADMIN_KEY or an API key is configured server-side.",
		},
	];
}
