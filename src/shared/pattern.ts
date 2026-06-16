export function matchesAnyPattern(value: string, patterns: string[]): boolean {
  if (patterns.length === 0) {
    return false;
  }

  return patterns.some((pattern) => matchesPattern(value, pattern));
}

export function matchesPattern(value: string, pattern: string): boolean {
  const normalizedPattern = pattern.trim();
  if (!normalizedPattern) {
    return false;
  }

  if (normalizedPattern === '*') {
    return true;
  }

  if (!normalizedPattern.includes('*')) {
    return value === normalizedPattern;
  }

  const escaped = normalizedPattern.replace(/[.+?^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*');
  const matcher = new RegExp(`^${escaped}$`);
  return matcher.test(value);
}

export function toLowerSet(values: string[]): Set<string> {
  const normalized = values
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean);

  return new Set(normalized);
}
