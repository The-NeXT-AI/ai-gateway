const esbuild = require('esbuild');

async function build() {
  const watchMode = process.argv.includes('--watch');
  const buildOptions = {
    entryPoints: ['src/index.ts'],
    bundle: true,
    platform: 'node',
    target: 'node20',
    outfile: 'dist/index.js',
    minify: !watchMode,
    sourcemap: true,
    external: ['fastify', 'undici', 'ws'],
  };

  try {
    if (watchMode) {
      const context = await esbuild.context(buildOptions);
      await context.watch();
      console.log('👀 Gateway build watch 已启动');

      const shutdown = async () => {
        await context.dispose();
        process.exit(0);
      };

      process.on('SIGINT', shutdown);
      process.on('SIGTERM', shutdown);
      return;
    }

    await esbuild.build(buildOptions);
    console.log('✅ 构建成功!');
  } catch (error) {
    console.error('❌ 构建失败:', error);
    process.exit(1);
  }
}

build();
