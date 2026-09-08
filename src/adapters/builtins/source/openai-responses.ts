import type { SourceAdapter } from '../../../types';
import { ok } from '../../../types';
import { asBoolean } from '../../../utils';
import { buildOpenAIPassthroughHeaders, normalizeOpenAIResponsesCompletedResponse } from '../common';
import { parseOpenAIResponsesRequest } from './parsers';
import { addNamespaceFieldsToStandardResponse } from '../target/tools';

export const openAIResponsesSourceAdapter: SourceAdapter = {
  key: 'openai_responses',
  provider: 'openai',
  toStandardRequest(input) {
    return parseOpenAIResponsesRequest(input.body);
  },
  fromStandardResponse(input) {
    return normalizeOpenAIResponsesCompletedResponse({
      ...addNamespaceFieldsToStandardResponse(input.response, input.standardRequest?.tools)
    });
  },
  isStreamingRequest(input) {
    return asBoolean(input.body.stream) === true;
  },
  buildPassthroughRequest(input) {
    const headersResult = buildOpenAIPassthroughHeaders(input.request.headers, input.config);
    if (!headersResult.ok) {
      return headersResult;
    }

    return ok({
      url: `${input.config.openaiBaseUrl}/responses`,
      headers: headersResult.value,
      body: input.body
    });
  }
};
