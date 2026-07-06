import type { GatewayConfig, ProviderConfig, ProviderHealthStatus } from '../types';
import { providerFromProviderType } from '../utils';

interface GatewayHttpMetricKey {
  method: string;
  route: string;
  statusCode: number;
  statusClass: string;
}

interface GatewayHttpMetricValue extends GatewayHttpMetricKey {
  count: number;
  durationMsSum: number;
  durationMsMax: number;
  durationBuckets: number[];
}

interface GatewayToolExecutionMetricKey {
  provider: string;
  providerName: string;
  sourceAdapter: string;
  outcome: string;
}

interface GatewayToolExecutionMetricValue extends GatewayToolExecutionMetricKey {
  count: number;
}

interface GatewayStreamConversionMetricKey {
  sourceAdapter: string;
  targetProvider: string;
  targetProviderName: string;
  mode: string;
}

interface GatewayStreamConversionMetricValue extends GatewayStreamConversionMetricKey {
  count: number;
}

interface GatewayBillingDeliveryMetricKey {
  outcome: string;
  transport: string;
}

interface GatewayBillingDeliveryMetricValue extends GatewayBillingDeliveryMetricKey {
  count: number;
}

export interface GatewayHttpMetricInput {
  method: string;
  route?: string;
  statusCode: number;
  durationMs: number;
}

export interface GatewayToolExecutionMetricInput {
  provider: string;
  providerName?: string;
  sourceAdapter: string;
  outcome: 'success' | 'error';
}

export interface GatewayStreamConversionMetricInput {
  sourceAdapter: string;
  targetProvider: string;
  targetProviderName?: string;
  mode: 'passthrough' | 'live' | 'buffered';
}

export interface GatewayBillingDeliveryMetricInput {
  outcome: 'delivered' | 'failed' | 'not_configured' | 'not_delivered';
  transport?: string;
}

const httpMetrics = new Map<string, GatewayHttpMetricValue>();
const toolExecutionMetrics = new Map<string, GatewayToolExecutionMetricValue>();
const streamConversionMetrics = new Map<string, GatewayStreamConversionMetricValue>();
const billingDeliveryMetrics = new Map<string, GatewayBillingDeliveryMetricValue>();
const providerHealthStatuses: ProviderHealthStatus[] = ['healthy', 'degraded', 'unknown', 'down'];
const httpDurationBucketsSeconds = [0.05, 0.1, 0.25, 0.5, 1, 3, 10, 30];

export function recordGatewayHttpRequest(input: GatewayHttpMetricInput): void {
  const statusCode = normalizeStatusCode(input.statusCode);
  const key: GatewayHttpMetricKey = {
    method: normalizeMethod(input.method),
    route: normalizeRoute(input.route),
    statusCode,
    statusClass: statusClass(statusCode)
  };
  const metricKey = serializeHttpMetricKey(key);
  const durationMs = normalizeDurationMs(input.durationMs);
  const metric = httpMetrics.get(metricKey) || {
    ...key,
    count: 0,
    durationMsSum: 0,
    durationMsMax: 0,
    durationBuckets: new Array(httpDurationBucketsSeconds.length).fill(0)
  };

  metric.count += 1;
  metric.durationMsSum += durationMs;
  metric.durationMsMax = Math.max(metric.durationMsMax, durationMs);
  const durationSeconds = durationMs / 1000;
  for (let index = 0; index < httpDurationBucketsSeconds.length; index += 1) {
    if (durationSeconds <= httpDurationBucketsSeconds[index]) {
      metric.durationBuckets[index] += 1;
    }
  }
  httpMetrics.set(metricKey, metric);
}

export function recordGatewayToolExecution(input: GatewayToolExecutionMetricInput): void {
  const key: GatewayToolExecutionMetricKey = {
    provider: normalizeLabel(input.provider),
    providerName: normalizeLabel(input.providerName || 'default'),
    sourceAdapter: normalizeLabel(input.sourceAdapter),
    outcome: normalizeLabel(input.outcome)
  };
  const metricKey = serializeToolExecutionMetricKey(key);
  const metric = toolExecutionMetrics.get(metricKey) || {
    ...key,
    count: 0
  };

  metric.count += 1;
  toolExecutionMetrics.set(metricKey, metric);
}

export function recordGatewayStreamConversion(input: GatewayStreamConversionMetricInput): void {
  const key: GatewayStreamConversionMetricKey = {
    sourceAdapter: normalizeLabel(input.sourceAdapter),
    targetProvider: normalizeLabel(input.targetProvider),
    targetProviderName: normalizeLabel(input.targetProviderName || 'default'),
    mode: normalizeLabel(input.mode)
  };
  const metricKey = serializeStreamConversionMetricKey(key);
  const metric = streamConversionMetrics.get(metricKey) || {
    ...key,
    count: 0
  };

  metric.count += 1;
  streamConversionMetrics.set(metricKey, metric);
}

export function recordGatewayBillingDelivery(input: GatewayBillingDeliveryMetricInput): void {
  const key: GatewayBillingDeliveryMetricKey = {
    outcome: normalizeLabel(input.outcome),
    transport: normalizeLabel(input.transport || 'none')
  };
  const metricKey = serializeBillingDeliveryMetricKey(key);
  const metric = billingDeliveryMetrics.get(metricKey) || {
    ...key,
    count: 0
  };

  metric.count += 1;
  billingDeliveryMetrics.set(metricKey, metric);
}

export function renderGatewayMetrics(config: GatewayConfig): string {
  const lines: string[] = [];

  lines.push('# HELP gateway_http_requests_total Total HTTP requests handled by the gateway.');
  lines.push('# TYPE gateway_http_requests_total counter');
  for (const metric of sortedHttpMetrics()) {
    lines.push(
      `gateway_http_requests_total${formatLabels(httpMetricLabels(metric))} ${metric.count}`
    );
  }

  lines.push('# HELP gateway_http_request_duration_ms_sum Sum of gateway HTTP request durations in milliseconds.');
  lines.push('# TYPE gateway_http_request_duration_ms_sum counter');
  for (const metric of sortedHttpMetrics()) {
    lines.push(
      `gateway_http_request_duration_ms_sum${formatLabels(httpMetricLabels(metric))} ${formatMetricNumber(metric.durationMsSum)}`
    );
  }

  lines.push('# HELP gateway_http_request_duration_ms_count Count of gateway HTTP request durations.');
  lines.push('# TYPE gateway_http_request_duration_ms_count counter');
  for (const metric of sortedHttpMetrics()) {
    lines.push(
      `gateway_http_request_duration_ms_count${formatLabels(httpMetricLabels(metric))} ${metric.count}`
    );
  }

  lines.push('# HELP gateway_http_request_duration_ms_max Max gateway HTTP request duration in milliseconds.');
  lines.push('# TYPE gateway_http_request_duration_ms_max gauge');
  for (const metric of sortedHttpMetrics()) {
    lines.push(
      `gateway_http_request_duration_ms_max${formatLabels(httpMetricLabels(metric))} ${formatMetricNumber(metric.durationMsMax)}`
    );
  }

  lines.push('# HELP gateway_http_request_duration_seconds Gateway HTTP request duration in seconds.');
  lines.push('# TYPE gateway_http_request_duration_seconds histogram');
  for (const metric of sortedHttpMetrics()) {
    const labels = httpMetricLabels(metric);
    for (let index = 0; index < httpDurationBucketsSeconds.length; index += 1) {
      lines.push(
        `gateway_http_request_duration_seconds_bucket${formatLabels({ ...labels, le: String(httpDurationBucketsSeconds[index]) })} ${metric.durationBuckets[index]}`
      );
    }
    lines.push(
      `gateway_http_request_duration_seconds_bucket${formatLabels({ ...labels, le: '+Inf' })} ${metric.count}`
    );
    lines.push(
      `gateway_http_request_duration_seconds_sum${formatLabels(labels)} ${formatMetricNumber(metric.durationMsSum / 1000)}`
    );
    lines.push(
      `gateway_http_request_duration_seconds_count${formatLabels(labels)} ${metric.count}`
    );
  }

  lines.push('# HELP gateway_tool_executions_total Total transparent gateway tool executions.');
  lines.push('# TYPE gateway_tool_executions_total counter');
  for (const metric of sortedToolExecutionMetrics()) {
    lines.push(
      `gateway_tool_executions_total${formatLabels(toolExecutionMetricLabels(metric))} ${metric.count}`
    );
  }

  lines.push('# HELP gateway_stream_conversions_total Total streaming requests by gateway conversion mode.');
  lines.push('# TYPE gateway_stream_conversions_total counter');
  for (const metric of sortedStreamConversionMetrics()) {
    lines.push(
      `gateway_stream_conversions_total${formatLabels(streamConversionMetricLabels(metric))} ${metric.count}`
    );
  }

  lines.push('# HELP gateway_billing_events_total Gateway billing event delivery attempts by outcome.');
  lines.push('# TYPE gateway_billing_events_total counter');
  for (const metric of sortedBillingDeliveryMetrics()) {
    lines.push(
      `gateway_billing_events_total${formatLabels(billingDeliveryMetricLabels(metric))} ${metric.count}`
    );
  }

  if (config.metrics.includeProviderHealth) {
    appendProviderHealthMetrics(lines, config.providers);
  }

  return `${lines.join('\n')}\n`;
}

export function resetGatewayMetricsForTests(): void {
  httpMetrics.clear();
  toolExecutionMetrics.clear();
  streamConversionMetrics.clear();
  billingDeliveryMetrics.clear();
}

function appendProviderHealthMetrics(lines: string[], providers: ProviderConfig[]): void {
  lines.push('# HELP gateway_provider_info Configured gateway provider metadata.');
  lines.push('# TYPE gateway_provider_info gauge');
  lines.push('# HELP gateway_provider_health_status Current provider health status, represented as one-hot status labels.');
  lines.push('# TYPE gateway_provider_health_status gauge');
  lines.push('# HELP gateway_provider_available Current provider availability, where unknown is omitted.');
  lines.push('# TYPE gateway_provider_available gauge');
  lines.push('# HELP gateway_provider_latency_ms Last observed provider latency in milliseconds.');
  lines.push('# TYPE gateway_provider_latency_ms gauge');

  for (const providerConfig of providers) {
    const provider = providerFromProviderType(providerConfig.type);
    const baseLabels = {
      provider,
      provider_name: providerConfig.name,
      type: providerConfig.type
    };
    const healthStatus = providerConfig.health?.status || 'unknown';

    lines.push(`gateway_provider_info${formatLabels(baseLabels)} 1`);
    for (const status of providerHealthStatuses) {
      lines.push(
        `gateway_provider_health_status${formatLabels({ ...baseLabels, status })} ${healthStatus === status ? 1 : 0}`
      );
    }

    if (typeof providerConfig.health?.available === 'boolean') {
      lines.push(
        `gateway_provider_available${formatLabels(baseLabels)} ${providerConfig.health.available ? 1 : 0}`
      );
    }

    if (Number.isFinite(providerConfig.health?.latencyMs)) {
      lines.push(
        `gateway_provider_latency_ms${formatLabels(baseLabels)} ${formatMetricNumber(providerConfig.health?.latencyMs || 0)}`
      );
    }
  }
}

function sortedHttpMetrics(): GatewayHttpMetricValue[] {
  return Array.from(httpMetrics.values()).sort((left, right) => {
    const leftKey = serializeHttpMetricKey(left);
    const rightKey = serializeHttpMetricKey(right);
    return leftKey.localeCompare(rightKey);
  });
}

function sortedToolExecutionMetrics(): GatewayToolExecutionMetricValue[] {
  return Array.from(toolExecutionMetrics.values()).sort((left, right) => {
    const leftKey = serializeToolExecutionMetricKey(left);
    const rightKey = serializeToolExecutionMetricKey(right);
    return leftKey.localeCompare(rightKey);
  });
}

function sortedStreamConversionMetrics(): GatewayStreamConversionMetricValue[] {
  return Array.from(streamConversionMetrics.values()).sort((left, right) => {
    const leftKey = serializeStreamConversionMetricKey(left);
    const rightKey = serializeStreamConversionMetricKey(right);
    return leftKey.localeCompare(rightKey);
  });
}

function sortedBillingDeliveryMetrics(): GatewayBillingDeliveryMetricValue[] {
  return Array.from(billingDeliveryMetrics.values()).sort((left, right) => {
    const leftKey = serializeBillingDeliveryMetricKey(left);
    const rightKey = serializeBillingDeliveryMetricKey(right);
    return leftKey.localeCompare(rightKey);
  });
}

function httpMetricLabels(metric: GatewayHttpMetricValue): Record<string, string> {
  return {
    method: metric.method,
    route: metric.route,
    status_code: String(metric.statusCode),
    status_class: metric.statusClass
  };
}

function toolExecutionMetricLabels(metric: GatewayToolExecutionMetricValue): Record<string, string> {
  return {
    outcome: metric.outcome,
    provider: metric.provider,
    provider_name: metric.providerName,
    source_adapter: metric.sourceAdapter
  };
}

function streamConversionMetricLabels(metric: GatewayStreamConversionMetricValue): Record<string, string> {
  return {
    mode: metric.mode,
    source_adapter: metric.sourceAdapter,
    target_provider: metric.targetProvider,
    target_provider_name: metric.targetProviderName
  };
}

function billingDeliveryMetricLabels(metric: GatewayBillingDeliveryMetricValue): Record<string, string> {
  return {
    outcome: metric.outcome,
    transport: metric.transport
  };
}

function serializeHttpMetricKey(key: GatewayHttpMetricKey): string {
  return `${key.method}\n${key.route}\n${key.statusCode}\n${key.statusClass}`;
}

function serializeToolExecutionMetricKey(key: GatewayToolExecutionMetricKey): string {
  return `${key.provider}\n${key.providerName}\n${key.sourceAdapter}\n${key.outcome}`;
}

function serializeStreamConversionMetricKey(key: GatewayStreamConversionMetricKey): string {
  return `${key.sourceAdapter}\n${key.targetProvider}\n${key.targetProviderName}\n${key.mode}`;
}

function serializeBillingDeliveryMetricKey(key: GatewayBillingDeliveryMetricKey): string {
  return `${key.outcome}\n${key.transport}`;
}

function normalizeMethod(value: string): string {
  const normalized = value.trim().toUpperCase();
  return normalized || 'UNKNOWN';
}

function normalizeRoute(value: string | undefined): string {
  const normalized = value?.trim();
  return normalized || 'unknown';
}

function normalizeLabel(value: string): string {
  const normalized = value.trim();
  return normalized || 'unknown';
}

function normalizeStatusCode(value: number): number {
  if (!Number.isFinite(value) || value < 100 || value > 999) {
    return 0;
  }

  return Math.trunc(value);
}

function statusClass(statusCode: number): string {
  if (statusCode < 100) {
    return 'unknown';
  }

  return `${Math.trunc(statusCode / 100)}xx`;
}

function normalizeDurationMs(value: number): number {
  if (!Number.isFinite(value) || value < 0) {
    return 0;
  }

  return value;
}

function formatLabels(labels: Record<string, string>): string {
  const entries = Object.entries(labels).sort(([left], [right]) => left.localeCompare(right));
  if (entries.length === 0) {
    return '';
  }

  return `{${entries.map(([key, value]) => `${key}="${escapeLabelValue(value)}"`).join(',')}}`;
}

function escapeLabelValue(value: string): string {
  return value.replace(/\\/g, '\\\\').replace(/\n/g, '\\n').replace(/"/g, '\\"');
}

function formatMetricNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return '0';
  }

  return Number(value.toFixed(3)).toString();
}
