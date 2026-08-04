import type { ICredentialType, INodeProperties } from 'n8n-workflow';

export class HakiApi implements ICredentialType {
	name = 'hakiApi';

	displayName = 'Haki API';

	documentationUrl = 'https://github.com/BigOD2307/Haki';

	properties: INodeProperties[] = [
		{
			displayName: 'Base URL',
			name: 'base_url',
			type: 'string',
			default: 'http://localhost:8100',
			required: true,
			placeholder: 'http://localhost:8100',
			description:
				"URL de l'API Haki, sans slash final. Depuis un n8n Docker vers une API Haki sur la machine hôte : http://host.docker.internal:8100",
		},
		{
			displayName: 'API Key',
			name: 'api_key',
			type: 'string',
			typeOptions: { password: true },
			default: '',
			description:
				"Clé Bearer Haki. Optionnelle en développement local (l'API tourne en mode ouvert), obligatoire dès que HAKI_API_KEY est configurée côté serveur.",
		},
	];
}
