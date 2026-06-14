#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';

const usage = `
Usage:
  npm run release -- <version> [options]

Examples:
  npm run release -- 1.2.3
  npm run release -- 1.2.3-beta.1 --tag next
  npm run release -- 1.2.3 --dry-run

Options:
  --tag <tag>          npm dist-tag. Defaults to latest, or next for prerelease versions.
  --access <access>    npm access level: public or restricted. Defaults to public.
  --otp <code>         npm two-factor authentication code.
  --provenance         Publish with npm provenance. Intended for GitHub Actions.
  --dry-run            Run npm publish in dry-run mode.
  --skip-tests         Skip npm test before publishing.
`;

const semverPattern =
  /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$/;

function parseArgs(argv) {
  const options = {
    access: 'public',
    dryRun: false,
    otp: undefined,
    provenance: false,
    skipTests: false,
    tag: undefined,
    version: undefined
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if (arg === '--help' || arg === '-h') {
      console.log(usage.trim());
      process.exit(0);
    }

    if (arg === '--dry-run') {
      options.dryRun = true;
      continue;
    }

    if (arg === '--provenance') {
      options.provenance = true;
      continue;
    }

    if (arg === '--skip-tests') {
      options.skipTests = true;
      continue;
    }

    if (arg === '--tag' || arg === '--access' || arg === '--otp') {
      const value = argv[index + 1];
      if (!value || value.startsWith('--')) {
        throw new Error(`${arg} requires a value.`);
      }

      if (arg === '--tag') {
        options.tag = value;
      } else if (arg === '--access') {
        options.access = value;
      } else {
        options.otp = value;
      }

      index += 1;
      continue;
    }

    if (arg.startsWith('--')) {
      throw new Error(`Unknown option: ${arg}`);
    }

    if (options.version) {
      throw new Error(`Unexpected extra argument: ${arg}`);
    }

    options.version = arg;
  }

  return options;
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    stdio: 'inherit',
    shell: false,
    ...options
  });

  if (result.error) {
    throw result.error;
  }

  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(' ')} failed with exit code ${result.status}.`);
  }
}

function readPackageJson() {
  return JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
}

try {
  const options = parseArgs(process.argv.slice(2));
  const targetVersion = options.version?.replace(/^v/, '');

  if (!targetVersion || !semverPattern.test(targetVersion)) {
    throw new Error('A valid semver version is required, for example 1.2.3 or 1.2.3-beta.1.');
  }

  if (!['public', 'restricted'].includes(options.access)) {
    throw new Error('--access must be public or restricted.');
  }

  const packageJson = readPackageJson();
  if (options.access === 'restricted' && !packageJson.name.startsWith('@')) {
    throw new Error('--access restricted requires a scoped package name, for example @scope/package.');
  }

  const distTag = options.tag ?? (targetVersion.includes('-') ? 'next' : 'latest');

  if (!options.skipTests) {
    run('npm', ['test']);
  }

  if (packageJson.version !== targetVersion) {
    run('npm', ['version', targetVersion, '--no-git-tag-version']);
  } else {
    console.log(`package.json is already at ${targetVersion}.`);
  }

  run('npm', ['pack', '--dry-run']);

  const publishArgs = ['publish', '--tag', distTag, '--access', options.access];
  if (options.otp) {
    publishArgs.push('--otp', options.otp);
  }
  if (options.provenance && !options.dryRun) {
    publishArgs.push('--provenance');
  }
  if (options.dryRun) {
    publishArgs.push('--dry-run');
  }

  run('npm', publishArgs);
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  console.error('');
  console.error(usage.trim());
  process.exit(1);
}
