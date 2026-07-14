import type {
  GatewayBillingTrace,
  BillingQueueConfig,
  BillingWebhookConfig,
  GatewayRequestClientContext,
  GatewayRequestIdentity,
  Provider
} from '../types';
import { publishJsonEventToExternalSink } from '../external-event-sink';
import { recordGatewayBillingDelivery } from '../gateway/metrics';
import type { BillingResult } from './calculate';

export interface BillingPublisherLogger {
  info(payload: unknown, message?: string): void;
  warn(payload: unknown, message?: string): void;
}

export interface BillingQueueEvent {
  eventId: string;
  emittedAt: string;
  requestId: string;
  clientIp?: string;
  attempt?: {
    kind?: 'upstream_attempt';
    sequence?: number;
  };
  route: {
    method: string;
    url: string;
  };
  source: {
    provider: Provider;
    adapterKey: string;
  };
  target: {
    provider: Provider;
    model?: string;
    providerName?: string;
  };
  fallback: {
    used: boolean;
    attempts: number;
  };
  performance?: {
    latency_ms?: number;
  };
  identity?: GatewayRequestIdentity;
  clientContext?: GatewayRequestClientContext;
  trace?: GatewayBillingTrace;
  outcome?: {
    status: 'success' | 'error' | 'timeout' | 'rate-limited';
    statusCode?: number;
    errorMessage?: string;
  };
  attempts?: Array<{
    provider: Provider;
    providerName?: string;
    stage: string;
    message: string;
    status?: number;
    details?: unknown;
  }>;
  billing: BillingResult;
}

let queueConfig: BillingQueueConfig | undefined;
let webhookConfig: BillingWebhookConfig | undefined;
let logger: BillingPublisherLogger | undefined;

export async function initializeBillingPublisher(
  queuePublisherConfig: BillingQueueConfig,
  webhookPublisherConfig: BillingWebhookConfig,
  log?: BillingPublisherLogger
): Promise<void> {
  logger = log;
  initializeWebhookPublisher(webhookPublisherConfig);
  initializeQueuePublisher(queuePublisherConfig);
}

function initializeWebhookPublisher(config: BillingWebhookConfig): void {
  if (!config.enabled) {
    webhookConfig = config;
    logger?.info(
      {
        enabled: false
      },
      'Billing webhook publisher is disabled.'
    );
    return;
  }

  const target = normalizeWebhookTarget(config);
  if (!target) {
    webhookConfig = {
      ...config,
      enabled: false
    };
    logger?.warn(
      {
        configuredEnabled: true
      },
      'Billing webhook publisher is enabled but target is missing. Webhook delivery is disabled.'
    );
    return;
  }

  webhookConfig = {
    ...config,
    endpoint: config.endpoint?.trim(),
    command: config.command?.trim()
  };
  logger?.info(
    {
      target,
      transport: config.transport,
      timeoutMs: config.timeoutMs,
      maxAttempts: config.maxAttempts,
      headerKeys: Object.keys(config.headers)
    },
    'Billing webhook publisher initialized.'
  );
}

function initializeQueuePublisher(config: BillingQueueConfig): void {
  queueConfig = {
    ...config,
    enabled: false
  };
  if (!config.enabled) {
    logger?.info(
      {
        enabled: false
      },
      'Billing queue publisher is disabled.'
    );
    return;
  }

  logger?.warn(
    {
      queueName: config.queueName,
      jobName: config.jobName
    },
    'Billing queue publisher is disabled because the gateway does not open database or Redis connections. Use billingWebhook or an external protocol adapter for delivery.'
  );
}

export async function publishBillingEvent(event: BillingQueueEvent): Promise<boolean> {
  const deliveries: Array<Promise<boolean>> = [];

  if (webhookConfig?.enabled && normalizeWebhookTarget(webhookConfig)) {
    deliveries.push(publishJsonEventToExternalSink(event, webhookConfig));
  }

  if (deliveries.length === 0) {
    recordGatewayBillingDelivery({
      outcome: 'not_configured',
      transport: 'none'
    });
    return false;
  }

  const settled = await Promise.allSettled(deliveries);
  let delivered = false;
  const failures: string[] = [];

  for (const result of settled) {
    if (result.status === 'fulfilled') {
      delivered = delivered || result.value;
      continue;
    }

    failures.push(toErrorMessage(result.reason));
  }

  if (failures.length > 0) {
    recordGatewayBillingDelivery({
      outcome: 'failed',
      transport: webhookConfig?.transport || 'unknown'
    });
    logger?.warn(
      {
        details: failures
      },
      'One or more billing publishers failed to deliver event.'
    );
  }

  if (delivered) {
    recordGatewayBillingDelivery({
      outcome: 'delivered',
      transport: webhookConfig?.transport || 'unknown'
    });
    return true;
  }

  if (failures.length > 0) {
    throw new Error(failures.join(' | '));
  }

  recordGatewayBillingDelivery({
    outcome: 'not_delivered',
    transport: webhookConfig?.transport || 'unknown'
  });
  return false;
}

export async function closeBillingPublisher(): Promise<void> {
  webhookConfig = undefined;
  queueConfig = undefined;
}

function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function normalizeWebhookTarget(config: BillingWebhookConfig): string | undefined {
  if (config.transport === 'stdio') {
    return config.command?.trim() || undefined;
  }

  return config.endpoint?.trim() || undefined;
}
