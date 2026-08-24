# SDK Publishing Guide

`sdk/python` (PyPI) + `sdk/typescript` (npm) 외부 배포 절차. 처음 배포 시 1회, 이후 버전 bump 시마다 반복.

## 사전 결정 (운영자)

- **패키지명**: 현재 `aicess-tts` — PyPI/npm에 동일 이름 사용 가능한지 검증 필요 (선점 시 다른 이름)
- **라이선스**: 현재 `Proprietary`/`UNLICENSED` — 외부 공개 시 MIT/Apache-2.0 등으로 변경 (사내 전용이면 private registry)
- **버전 정책**: SemVer 권장. API breaking change 시 major bump
- **저장소 URL**: `homepage`, `repository` 필드 추가 (PyPI/npm 페이지에 노출)

## Python SDK (PyPI)

### 1. 메타데이터 보강

[sdk/python/pyproject.toml](../sdk/python/pyproject.toml)에 다음 필드 추가 검토:

```toml
[project]
name = "aicess-tts"          # PyPI 선점 확인: pip search 또는 https://pypi.org/project/aicess-tts/
version = "0.1.0"             # SemVer
license = { text = "MIT" }   # 또는 운영 정책
keywords = ["tts", "qwen3", "supertonic", "voice-cloning", "icl"]

[project.urls]
Homepage = "https://github.com/your-org/qwen3-tts-api"
Documentation = "https://your-domain.com/docs"
Repository = "https://github.com/your-org/qwen3-tts-api.git"
Issues = "https://github.com/your-org/qwen3-tts-api/issues"
```

### 2. 빌드 도구

```bash
pip install build twine
```

### 3. 빌드

```bash
cd sdk/python
rm -rf dist/ build/ *.egg-info
python -m build
# → dist/aicess_tts-0.1.0-py3-none-any.whl
# → dist/aicess_tts-0.1.0.tar.gz
```

### 4. 검증 (TestPyPI 먼저 권장)

```bash
# TestPyPI 토큰: https://test.pypi.org/manage/account/token/
twine upload --repository testpypi dist/*
pip install -i https://test.pypi.org/simple/ aicess-tts
# 동작 확인 후 ↓
```

### 5. 운영 배포

```bash
# PyPI 토큰: https://pypi.org/manage/account/token/
twine upload dist/*
# 약 1~2분 후 https://pypi.org/project/aicess-tts/ 에 노출
```

### 6. 검증

```bash
pip install aicess-tts
python -c "from aicess_tts import Client; print(Client.__doc__)"
```

## TypeScript SDK (npm)

### 1. 메타데이터 보강

[sdk/typescript/package.json](../sdk/typescript/package.json)에 추가:

```json
{
  "license": "MIT",
  "homepage": "https://github.com/your-org/qwen3-tts-api",
  "repository": {
    "type": "git",
    "url": "git+https://github.com/your-org/qwen3-tts-api.git",
    "directory": "sdk/typescript"
  },
  "bugs": "https://github.com/your-org/qwen3-tts-api/issues",
  "publishConfig": {
    "access": "public"
  }
}
```

### 2. 빌드

```bash
cd sdk/typescript
npm install
npm run build
# → dist/index.js, dist/index.d.ts 등 생성 확인
```

### 3. npm 로그인

```bash
npm login
# 또는 token: ~/.npmrc 에 //registry.npmjs.org/:_authToken=npm_xxx
```

### 4. dry-run 검증

```bash
npm publish --dry-run
# 어떤 파일이 패키지에 포함되는지 확인 (package.json "files" 필드 기준)
```

### 5. 운영 배포

```bash
npm publish
# scoped name이면: npm publish --access public
# 약 1분 후 https://www.npmjs.com/package/aicess-tts 에 노출
```

### 6. 검증

```bash
npm install aicess-tts
node -e "const {Client} = require('aicess-tts'); console.log(typeof Client)"
```

## 버전 bump 절차

API 변경 후:

| 변경 종류 | 버전 bump | 예시 |
|---|---|---|
| Patch (호환, 버그 수정) | 0.1.0 → 0.1.1 | 응답 헤더 추가, 내부 최적화 |
| Minor (호환, 신기능) | 0.1.0 → 0.2.0 | 신규 메서드 추가, optional 인자 |
| Major (breaking) | 0.1.0 → 1.0.0 | 메서드 시그너처 변경, 인자 제거 |

```bash
# Python
sed -i 's/version = "0.1.0"/version = "0.1.1"/' sdk/python/pyproject.toml

# TypeScript
cd sdk/typescript && npm version patch  # 자동 increment
```

## 사내 Private Registry (선택)

외부 PyPI/npm 미사용 시:

### Python — devpi 또는 GitLab/GitHub Packages
```bash
twine upload --repository-url https://your-registry.example.com/simple/ dist/*
pip install --index-url https://your-registry.example.com/simple/ aicess-tts
```

### TypeScript — Verdaccio 또는 GitHub Packages
```bash
# ~/.npmrc 에 registry 등록
npm publish --registry https://your-registry.example.com/
```

## CI/CD 자동화 (참고)

`.github/workflows/publish.yml` 예시 (운영자가 작성):

```yaml
on:
  release:
    types: [published]
jobs:
  python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install build twine
      - run: cd sdk/python && python -m build
      - run: twine upload sdk/python/dist/*
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
  npm:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: cd sdk/typescript && npm ci && npm run build && npm publish
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

## 배포 후 체크리스트

- [ ] 패키지 페이지(PyPI/npm)에서 README 렌더링 확인
- [ ] `pip install` / `npm install` 후 quickstart 예제 동작 검증
- [ ] [curl_예제.md](../api/curl_예제.md) 옆에 SDK 사용 예제 1개씩 추가
- [ ] 버전 태그 git push: `git tag sdk-python-v0.1.0 && git push --tags`
- [ ] 릴리즈 노트 작성 (변경 사항, 호환성 메모)

## 보안 주의

- PyPI/npm 토큰은 절대 git commit 금지. `~/.pypirc`, `~/.npmrc` 또는 CI secrets 사용
- `twine upload --skip-existing` 사용 시 동일 버전 재업로드 차단 (PyPI는 1회 업로드 후 삭제/덮어쓰기 불가)
- npm 패키지에 `.env`, `node_modules`, 빌드 산출물 제외 (`.npmignore` 또는 `files` whitelist 사용)
