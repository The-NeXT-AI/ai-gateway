import { describe, expect, it } from 'vitest';
import { formatErrorWithCause } from './utils';

describe('formatErrorWithCause', () => {
  it('formats nested Error causes and Node-style error codes', () => {
    const socketError = Object.assign(new Error('connect ECONNREFUSED 127.0.0.1:443'), {
      code: 'ECONNREFUSED'
    });
    const requestError = new Error('request failed', { cause: socketError });
    const fetchError = new TypeError('fetch failed', { cause: requestError });

    expect(formatErrorWithCause(fetchError)).toBe(
      'fetch failed => request failed => connect ECONNREFUSED 127.0.0.1:443 (ECONNREFUSED)'
    );
  });

  it('handles non-Error values and circular cause chains', () => {
    expect(formatErrorWithCause('connection failed')).toBe('connection failed');

    const circularError = new Error('circular failure');
    Object.assign(circularError, { cause: circularError });
    expect(formatErrorWithCause(circularError)).toBe('circular failure => [circular cause]');
  });
});
