import type { SourceAdapter } from '../../../types';
import { ok } from '../../../types';
import { asBoolean } from '../../../utils';
import { buildOpenAIHeaders } from '../common';
import { formatOpenAIChatCompletionsResponse } from './formatters';
import { parseOpenAIChatCompletionsRequest } from './parsers';

export const openAIChatCompletionsSourceAdapter: SourceAdapter = {
  key: 'openai_chat',
  provider: 'openai',
  toStandardRequest(input) {
    return parseOpenAIChatCompletionsRequest(input.body);
  },
  fromStandardResponse(input) {
    return formatOpenAIChatCompletionsResponse(input.response);
  },
  isStreamingRequest(input) {
    return asBoolean(input.body.stream) === true;
  },
  buildPassthroughRequest(input) {
    const headersResult = buildOpenAIHeaders(input.request.headers, input.config);
    if (!headersResult.ok) {
      return headersResult;
    }

    return ok({
      url: `${input.config.openaiBaseUrl}/chat/completions`,
      headers: headersResult.value,
      body: input.body
    });
  }
};
