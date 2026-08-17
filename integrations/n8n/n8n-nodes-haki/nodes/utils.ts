import { NodeOperationError } from 'n8n-workflow';
import type { INode } from 'n8n-workflow';

export interface HakiCredentials {
	base_url: string;
	api_key?: string;
}

export interface PacketFact {
	id: string;
	predicate: string;
	value: Record<string, unknown>;
	confidence: number | null;
	valid_from: string | null;
	source_event_ids: string[];
}

export interface ContextApiResponse {
	packet: { facts: PacketFact[]; warnings: string[] };
	token_count: number;
	trace_id: string;
}

/** Validates the memory scope: a durable memory without a stable subject is
 * a dangerous memory — empty or `default` subjects are rejected hard. */
export function requireSubject(node: INode, subjectId: string, itemIndex: number): string {
	const subject = (subjectId ?? '').trim();
	if (!subject || subject === 'default') {
		throw new NodeOperationError(
			node,
			"subject_id is required and cannot be empty or \"default\". " +
				"Map a stable identifier (sessionId, email, webhook ID...) from the trigger.",
			{ itemIndex },
		);
	}
	return subject;
}

export function authHeaders(credentials: HakiCredentials): Record<string, string> {
	const apiKey = (credentials.api_key ?? '').trim();
	return apiKey ? { Authorization: `Bearer ${apiKey}` } : {};
}

export function baseUrl(credentials: HakiCredentials): string {
	return (credentials.base_url ?? '').replace(/\/+$/, '');
}

/** Formats a ContextPacket as an injectable system-prompt block. Mirrors the
 * formatting of the Python SDK runtime so all integrations read the same. */
export function formatPacket(packet: ContextApiResponse['packet']): string {
	const lines = [
		'<haki_memory>',
		'The following are verified long-term memories about the user. Prefer them over your own assumptions when relevant (cite them when you use them):',
	];
	for (const fact of packet.facts ?? []) {
		const validFrom = fact.valid_from ?? 'unknown date';
		const sources = (fact.source_event_ids ?? []).join(',') || 'no-source';
		lines.push(
			`- ${fact.predicate}: ${JSON.stringify(fact.value)} (valid from ${validFrom}; sources: ${sources})`,
		);
	}
	for (const warning of packet.warnings ?? []) {
		lines.push(`! ${warning}`);
	}
	lines.push('</haki_memory>');
	return lines.join('\n');
}

/** Turns any HTTP/fetch failure into a readable n8n execution error. */
export function apiError(node: INode, error: unknown, itemIndex: number): NodeOperationError {
	const err = error as {
		message?: string;
		status?: number;
		body?: unknown;
	};
	const status = err.status ? `HTTP ${err.status}` : 'connection failed';
	let detail = '';
	if (typeof err.body === 'string') {
		detail = err.body.slice(0, 500);
	} else if (err.body != null) {
		try {
			detail = JSON.stringify(err.body).slice(0, 500);
		} catch {
			detail = '';
		}
	}
	return new NodeOperationError(
		node,
		`Haki API error (${status})${detail ? `: ${detail}` : ''}. Check the credential's Base URL and that the Haki API is running.`,
		{ itemIndex },
	);
}
