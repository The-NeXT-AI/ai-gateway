# 发布与 CI/CD

## 包内容

npm 包通过 `.npmignore` 控制发布内容，包含运行所需的 `dist/`、`bin/`、`package.json`、`README.md`、`LICENSE` 和 `docs/`，并排除源码、测试、CI、Docker 与本地开发配置。

`npm pack` 或 `npm publish` 前会自动执行：

```bash
npm run build
```

## 本地发布

发布指定版本：

```bash
npm run release -- 1.2.3
```

预发布版本建议使用 `next` tag：

```bash
npm run release -- 1.2.3-beta.1 --tag next
```

常用选项：

```bash
npm run release -- 1.2.3 --dry-run
npm run release -- 1.2.3 --otp 123456
npm run release -- 1.2.3 --skip-tests
```

release 脚本会依次执行：

1. `npm test`
2. `npm version <version> --no-git-tag-version`
3. `npm pack --dry-run`
4. `npm publish`

## GitHub Actions

CI workflow 会在 push、pull request 和手动触发时执行：

```bash
npm ci
npm test
npm run build
npm pack --dry-run
```

Release workflow 通过 `workflow_dispatch` 手动触发，输入 npm version、dist-tag、access 和 dry-run 选项后发布到 npm。

发布前需要在 GitHub 仓库中配置 secret：

```text
NPM_TOKEN
```

该 token 需要具备发布 `next-ai-gateway` 包的权限。若 npm 包名不可用，请先调整 `package.json` 中的 `name` 字段并同步 lockfile。
