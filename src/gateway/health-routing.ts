import type { GatewayConfig, Provider, ProviderConfig, ProviderHealthStatus } from '../types';
import { findDefaultProviderConfig } from '../utils';

export interface HealthAwareProviderRoute {
  provider: Provider;
  providerConfig?: ProviderConfig;
}

interface AnnotatedRoute<T extends HealthAwareProviderRoute> {
  route: T;
  index: number;
  providerConfig?: ProviderConfig;
}

export function applyHealthAwareRouting<T extends HealthAwareProviderRoute>(
  routes: T[],
  config: GatewayConfig
): T[] {
  const routingConfig = config.healthAwareRouting;
  if (!routingConfig?.enabled || routes.length <= 1) {
    return routes;
  }

  const annotated = routes.map((route, index) => ({
    route,
    index,
    providerConfig: route.providerConfig || findProviderConfigByType(config.providers, route.provider)
  }));

  const filtered =
    routingConfig.skipUnavailable
      ? annotated.filter((item) => !isUnavailable(item, config))
      : annotated;
  const candidates = filtered.length > 0 ? filtered : annotated;

  if (!routingConfig.preferHealthy && !routingConfig.preferLowerLatency) {
    return candidates.map((item) => item.route);
  }

  return [...candidates]
    .sort((left, right) => compareHealthAwareRoutes(left, right, config))
    .map((item) => item.route);
}

function compareHealthAwareRoutes<T extends HealthAwareProviderRoute>(
  left: AnnotatedRoute<T>,
  right: AnnotatedRoute<T>,
  config: GatewayConfig
): number {
  if (config.healthAwareRouting.preferHealthy) {
    const statusDelta = healthScore(left) - healthScore(right);
    if (statusDelta !== 0) {
      return statusDelta;
    }
  }

  const priorityDelta = routePriority(left) - routePriority(right);
  if (priorityDelta !== 0) {
    return priorityDelta;
  }

  if (config.healthAwareRouting.preferLowerLatency) {
    const latencyDelta = routeLatency(left) - routeLatency(right);
    if (latencyDelta !== 0) {
      return latencyDelta;
    }
  }

  return left.index - right.index;
}

function isUnavailable<T extends HealthAwareProviderRoute>(
  item: AnnotatedRoute<T>,
  config: GatewayConfig
): boolean {
  const health = item.providerConfig?.health;
  if (!health) {
    return false;
  }

  return (
    health.available === false ||
    config.healthAwareRouting.unhealthyStatuses.includes(health.status)
  );
}

function healthScore<T extends HealthAwareProviderRoute>(item: AnnotatedRoute<T>): number {
  const health = item.providerConfig?.health;
  if (!health) {
    return 2;
  }

  if (health.available === false) {
    return 4;
  }

  if (health.available === true && health.status === 'unknown') {
    return 1;
  }

  return statusScore(health.status);
}

function statusScore(status: ProviderHealthStatus): number {
  if (status === 'healthy') {
    return 0;
  }

  if (status === 'degraded') {
    return 1;
  }

  if (status === 'unknown') {
    return 2;
  }

  return 3;
}

function routePriority<T extends HealthAwareProviderRoute>(item: AnnotatedRoute<T>): number {
  const priority = item.providerConfig?.health?.priority;
  return typeof priority === 'number' && Number.isFinite(priority) ? priority : 0;
}

function routeLatency<T extends HealthAwareProviderRoute>(item: AnnotatedRoute<T>): number {
  const latencyMs = item.providerConfig?.health?.latencyMs;
  return typeof latencyMs === 'number' && Number.isFinite(latencyMs)
    ? latencyMs
    : Number.POSITIVE_INFINITY;
}

function findProviderConfigByType(
  providers: ProviderConfig[],
  provider: Provider
): ProviderConfig | undefined {
  return findDefaultProviderConfig(providers, provider);
}
