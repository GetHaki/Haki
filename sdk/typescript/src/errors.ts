/** Typed errors raised by the Haki SDK (parity with sdk/python/haki/errors.py). */

/** Base class for every SDK error. */
export class HakiError extends Error {
  constructor(message: string) {
    super(message);
    this.name = new.target.name;
  }
}

/** The API is unreachable (network error, timeout, DNS...). */
export class HakiConnectionError extends HakiError {}

/** The API returned an error payload {"error": {type, message, field}}. */
export class HakiApiError extends HakiError {
  readonly statusCode: number;
  readonly errorType?: string;
  readonly field?: string;
  readonly payload: Record<string, unknown>;

  constructor(
    message: string,
    options: {
      statusCode: number;
      errorType?: string;
      field?: string;
      payload?: Record<string, unknown>;
    },
  ) {
    super(message);
    this.statusCode = options.statusCode;
    this.errorType = options.errorType;
    this.field = options.field;
    this.payload = options.payload ?? {};
    // Same rendering as the Python SDK: "[401 unauthorized] message".
    this.message = this.errorType
      ? `[${this.statusCode} ${this.errorType}] ${message}`
      : `[${this.statusCode}] ${message}`;
  }
}
