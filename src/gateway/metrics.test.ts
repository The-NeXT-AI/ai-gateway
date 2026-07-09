import { afterEach, describe, expect, it } from 'vitest';
import { parseGatewayConfigFromRaw } from '../config';
import {
  recordGatewayBillingDelivery,
  recordGatewayHttpRequest,
  recordGatewayStreamConversion,
  recordGatewayToolExecution,
  renderGatewayMetrics,
  resetGatewayMetricsForTests
} from './metrics';

describe('gateway metrics', () => {
  afterEach(() => {
    resetGatewayMetricsForTests();
  });

  it('renders HTTP request counters and duration metrics', () => {
    const config = parseGatewayConfigFromRaw({
      metrics: {
        enabled: true,
        includeProviderHealth: false
      }
    });

    recordGatewayHttpRequest({
      method: 'post',
      route: '/v1/responses',
      statusCode: 200,
      durationMs: 12.3456
    });
    recordGatewayHttpRequest({
      method: 'POST',
      route: '/v1/responses',
      statusCode: 200,
      durationMs: 7
    });

    const metrics = renderGatewayMetrics(config);

    expect(metrics).toContain(
      'gateway_http_requests_total{method="POST",route="/v1/responses",status_class="2xx",status_code="200"} 2'
    );
    expect(metrics).toContain(
      'gateway_http_request_duration_ms_sum{method="POST",route="/v1/responses",status_class="2xx",status_code="200"} 19.346'
    );
    expect(metrics).toContain(
      'gateway_http_request_duration_ms_count{method="POST",route="/v1/responses",status_class="2xx",status_code="200"} 2'
    );
    expect(metrics).toContain(
      'gateway_http_request_duration_ms_max{method="POST",route="/v1/responses",status_class="2xx",status_code="200"} 12.346'
    );
    expect(metrics).toContain(
      'gateway_http_request_duration_seconds_bucket{le="0.05",method="POST",route="/v1/responses",status_class="2xx",status_code="200"} 2'
    );
    expect(metrics).toContain(
      'gateway_http_request_duration_seconds_bucket{le="+Inf",method="POST",route="/v1/responses",status_class="2xx",status_code="200"} 2'
    );
    expect(metrics).toContain(
      'gateway_http_request_duration_seconds_sum{method="POST",route="/v1/responses",status_class="2xx",status_code="200"} 0.019'
    );
    expect(metrics).toContain(
      'gateway_http_request_duration_seconds_count{method="POST",route="/v1/responses",status_class="2xx",status_code="200"} 2'
    );
    expect(metrics).not.toContain('gateway_provider_info');
  });

  it('renders provider health gauges when enabled', () => {
    const config = parseGatewayConfigFromRaw({
      metrics: {
        enabled: true,
        includeProviderHealth: true
      },
      providers: [
        {
          name: 'openai-main',
          type: 'openai_responses',
          models: ['gpt-4.1-mini'],
          health: {
            status: 'degraded',
            available: true,
            latencyMs: 123.4
          }
        }
      ]
    });

    const metrics = renderGatewayMetrics(config);

    expect(metrics).toContain(
      'gateway_provider_info{provider="openai",provider_name="openai-main",type="openai_responses"} 1'
    );
    expect(metrics).toContain(
      'gateway_provider_health_status{provider="openai",provider_name="openai-main",status="degraded",type="openai_responses"} 1'
    );
    expect(metrics).toContain(
      'gateway_provider_health_status{provider="openai",provider_name="openai-main",status="healthy",type="openai_responses"} 0'
    );
    expect(metrics).toContain(
      'gateway_provider_available{provider="openai",provider_name="openai-main",type="openai_responses"} 1'
    );
    expect(metrics).toContain(
      'gateway_provider_latency_ms{provider="openai",provider_name="openai-main",type="openai_responses"} 123.4'
    );
  });

  it('renders transparent tool execution and stream conversion counters', () => {
    const config = parseGatewayConfigFromRaw({
      metrics: {
        enabled: true,
        includeProviderHealth: false
      }
    });

    recordGatewayToolExecution({
      provider: 'openai',
      providerName: 'openai-main',
      sourceAdapter: 'openai_chat',
      outcome: 'success'
    });
    recordGatewayToolExecution({
      provider: 'openai',
      providerName: 'openai-main',
      sourceAdapter: 'openai_chat',
      outcome: 'success'
    });
    recordGatewayStreamConversion({
      sourceAdapter: 'openai_responses',
      targetProvider: 'anthropic',
      targetProviderName: 'anthropic-main',
      mode: 'buffered'
    });

    const metrics = renderGatewayMetrics(config);

    expect(metrics).toContain(
      'gateway_tool_executions_total{outcome="success",provider="openai",provider_name="openai-main",source_adapter="openai_chat"} 2'
    );
    expect(metrics).toContain(
      'gateway_stream_conversions_total{mode="buffered",source_adapter="openai_responses",target_provider="anthropic",target_provider_name="anthropic-main"} 1'
    );
  });

  it('renders billing delivery counters by outcome and transport', () => {
    const config = parseGatewayConfigFromRaw({
      metrics: {
        enabled: true,
        includeProviderHealth: false
      }
    });

    recordGatewayBillingDelivery({
      outcome: 'delivered',
      transport: 'http'
    });
    recordGatewayBillingDelivery({
      outcome: 'failed',
      transport: 'http'
    });
    recordGatewayBillingDelivery({
      outcome: 'failed',
      transport: 'http'
    });

    const metrics = renderGatewayMetrics(config);

    expect(metrics).toContain(
      'gateway_billing_events_total{outcome="delivered",transport="http"} 1'
    );
    expect(metrics).toContain(
      'gateway_billing_events_total{outcome="failed",transport="http"} 2'
    );
  });
});
